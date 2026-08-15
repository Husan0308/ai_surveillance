from __future__ import annotations

import os
import queue as pyqueue
import time
from collections import deque
from pathlib import Path

# PP-Human's production MOT pipeline recommends not skipping more than ~3 video
# frames between detector updates. On this GTX 1050 Ti we cannot afford detector
# inference at 20 FPS per camera, so use a 2-frame micro-batch plus an adaptive
# per-camera detection-Hz target.
os.environ.setdefault("CAMERA_V2_DETECT_WIDTH", "640")
os.environ.setdefault("CAMERA_V2_DETECT_HEIGHT", "384")
os.environ.setdefault("CAMERA_V2_MICRO_BATCH", "2")
os.environ.setdefault("CAMERA_V2_DETECT_CONF", "0.12")
os.environ.setdefault("CAMERA_V2_DETECT_IOU", "0.45")
os.environ.setdefault("CAMERA_V2_MAX_DET", "40")
os.environ.setdefault("CAMERA_V2_TRACKER_WIDTH", "640")
os.environ.setdefault("CAMERA_V2_TRACKER_HEIGHT", "384")
os.environ.setdefault("CAMERA_V2_TRACK_BOX_SIDE_MARGIN", "0.02")
os.environ.setdefault("CAMERA_V2_TRACK_BOX_TOP_MARGIN", "0.01")
os.environ.setdefault("CAMERA_V2_TRACK_BOX_BOTTOM_MARGIN", "0.03")
os.environ.setdefault("CAMERA_V2_DEDUP_IOU", "0.48")
os.environ.setdefault("CAMERA_V2_DEDUP_CONTAINMENT", "0.78")

from .detection import INFER_HEIGHT, INFER_WIDTH, MICRO_BATCH
from .detector_latency import DetectorLatencyCompensator, PreparedDetection
from .person_tracking import CameraPersonTrackingV2 as _BaseTracking
from .tracker_profile import prepare_sparse_tracker_config


class CameraPersonTrackingFinal(_BaseTracking):
    """High-cadence YOLO26m + continuous NvDCF pedestrian tracking."""

    def __init__(self) -> None:
        self.detector_frames_applied = 0
        self.detector_target_hz = float(os.environ.get("CAMERA_V2_DETECT_TARGET_HZ", "3.4"))
        self.detector_min_hz = float(os.environ.get("CAMERA_V2_DETECT_MIN_HZ", "2.4"))
        self.detector_max_hz = float(os.environ.get("CAMERA_V2_DETECT_MAX_HZ", "4.2"))
        self.detector_min_idle = float(os.environ.get("CAMERA_V2_DETECT_MIN_IDLE_MS", "18")) / 1000.0
        self.detector_result_age_ms = 0.0
        self.detector_times: dict[str, deque[float]] = {}
        super().__init__()
        self.detector_target_hz = max(self.detector_min_hz, min(self.detector_max_hz, self.detector_target_hz))
        self.detector_times = {cid: deque(maxlen=80) for cid in self.camera_index}
        self.latency_compensator = DetectorLatencyCompensator(self.frame_width, self.frame_height)

    def _resolve_tracker_files(self):
        lib, stock_max_perf = super()._resolve_tracker_files()
        perf = stock_max_perf.with_name("config_tracker_NvDCF_perf.yml")
        stock = perf if perf.exists() else stock_max_perf
        config = prepare_sparse_tracker_config(stock)
        return lib, config

    def _publish_prepared(
        self,
        cid: str,
        captured_t: float,
        prepared: list[PreparedDetection],
    ) -> None:
        with self.pending_lock:
            self.pending_seq += 1
            self.pending[cid] = (self.pending_seq, float(captured_t), list(prepared))

    def _inject_detector_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK

        now = time.monotonic()
        boxes_added = 0
        frames_applied = 0
        max_age_ms = 0.0
        with self.pending_lock:
            pending = dict(self.pending)

        for cid, source_id in self.camera_index.items():
            row = pending.get(cid)
            if row is None:
                continue
            seq, captured_t, prepared = row
            if seq <= self.injected_seq.get(cid, 0):
                continue

            boxes, age_ms = self.latency_compensator.project(prepared, captured_t, now)
            result = self.bridge.apply_detector_result(buffer, source_id, boxes)
            if result == -2:
                continue
            if result < 0:
                continue

            self.injected_seq[cid] = seq
            frames_applied += 1
            boxes_added += result
            max_age_ms = max(max_age_ms, age_ms)

        if frames_applied or boxes_added:
            with self.det_lock:
                self.detector_frames_applied += frames_applied
                self.meta_boxes += boxes_added
                self.detector_result_age_ms = max_age_ms
        return self.Gst.PadProbeReturn.OK

    def _tracker_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is not None:
            # Native bridge first promotes official current-frame NvDCF shadow-track
            # metadata, then applies display-only lead/limb margin. NvDCF remains
            # the only temporal tracker.
            count = self.bridge.style_and_count_tracked(buffer)
            if count >= 0:
                with self.det_lock:
                    self.tracked_now = count
                    self.tracker_frames += 1
        return self.Gst.PadProbeReturn.OK

    def _scheduler(self) -> None:
        assert self.result_q is not None and self.job_q is not None
        try:
            ready = self.result_q.get(timeout=40.0)
        except pyqueue.Empty:
            with self.det_lock:
                self.det_error = "YOLO worker startup timeout"
            return
        if ready.get("type") != "ready":
            with self.det_lock:
                self.det_error = ready.get("error", "YOLO worker failed")
            return

        with self.det_lock:
            self.det_ready = True
        print(
            "CAMERA_TRACK_FINAL ready: "
            f"YOLO26m micro_batch={MICRO_BATCH} input={INFER_WIDTH}x{INFER_HEIGHT} "
            f"target={self.detector_target_hz:.1f}Hz/cam "
            f"range={self.detector_min_hz:.1f}-{self.detector_max_hz:.1f}Hz/cam "
            f"NvDCF={self.tracker_width}x{self.tracker_height} "
            f"device={ready.get('device')} cuda={ready.get('cuda')} "
            "timestamp_compensation=1 nvdcf_per_frame=1 shadow_output=1",
            flush=True,
        )

        ids = [camera.camera_id for camera in self.cameras]
        groups = [ids[i : i + MICRO_BATCH] for i in range(0, len(ids), MICRO_BATCH)]
        versions = {cid: 0 for cid in ids}
        group_index = 0

        while not self.det_stop.is_set():
            cycle_started = time.monotonic()
            group = groups[group_index % len(groups)]
            group_index += 1

            self._request_group(group)
            rows = self.mailbox.wait_group(group, versions, timeout=1.0)
            if rows is None:
                self._clear_requests()
                with self.det_lock:
                    self.capture_timeouts += 1
                self.det_stop.wait(0.04)
                continue

            frames = []
            captured = []
            for cid, row in zip(group, rows):
                version, captured_t, frame = row
                versions[cid] = version
                captured.append(captured_t)
                frames.append(frame)
            self._clear_requests()

            try:
                self.job_q.put(
                    {"cameras": group, "frames": frames, "captured": captured},
                    timeout=0.4,
                )
                result = self.result_q.get(timeout=6.0)
            except pyqueue.Empty:
                with self.det_lock:
                    self.det_error = "YOLO result timeout"
                self.det_stop.wait(0.10)
                continue

            if result.get("type") == "fatal":
                with self.det_lock:
                    self.det_error = result.get("error", "YOLO fatal error")
                return
            if result.get("type") == "batch_error":
                with self.det_lock:
                    self.det_error = result.get("error", "YOLO batch error")
                self.det_stop.wait(0.20)
                continue
            if result.get("type") != "result":
                continue

            completed_t = time.monotonic()
            counts: dict[str, int] = {}
            ages_ms: list[float] = []
            for cid, captured_t in zip(result["cameras"], result["captured"]):
                detections = self._dedup_and_expand(result["boxes"].get(cid, []))
                prepared = self.latency_compensator.prepare(cid, captured_t, detections)
                self._publish_prepared(cid, captured_t, prepared)
                counts[cid] = len(detections)
                ages_ms.append(max(0.0, (completed_t - captured_t) * 1000.0))
                self.detector_times[cid].append(completed_t)

            batch_ms = float(result.get("batch_ms") or 0.0)
            with self.det_lock:
                self.det_calls += 1
                self.det_inputs += len(group)
                self.det_batch_ms = batch_ms
                self.det_counts.update(counts)
                if ages_ms:
                    self.detector_result_age_ms = max(ages_ms)
                self.det_error = ""
                target_hz = self.detector_target_hz

            desired_call_interval = 1.0 / max(0.1, target_hz * len(groups))
            elapsed = time.monotonic() - cycle_started
            idle = max(self.detector_min_idle, desired_call_interval - elapsed)
            self.det_stop.wait(min(0.35, idle))

    @staticmethod
    def _recent_rate(times: deque[float], now: float, horizon: float = 5.0) -> float:
        while times and now - times[0] > horizon:
            times.popleft()
        if len(times) < 2:
            return 0.0
        span = max(0.2, times[-1] - times[0])
        return (len(times) - 1) / span

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        p95 = self._p95(self.wall_intervals_ms)
        now = time.monotonic()

        with self.det_lock:
            if p95 is not None:
                if p95 > 88.0:
                    self.detector_target_hz -= 0.60
                elif p95 > 74.0:
                    self.detector_target_hz -= 0.30
                elif p95 < 62.0 and self.det_ready:
                    self.detector_target_hz += 0.15
            self.detector_target_hz = max(
                self.detector_min_hz,
                min(self.detector_max_hz, self.detector_target_hz),
            )
            applied = self.detector_frames_applied
            tracked = self.tracked_now
            age_ms = self.detector_result_age_ms
            target_hz = self.detector_target_hz

        rates = {cid: self._recent_rate(rows, now) for cid, rows in self.detector_times.items()}
        rate_text = " ".join(f"{cid}:{rates.get(cid, 0.0):.1f}" for cid in self.camera_index)
        expected_skip = max(0.0, 20.0 / max(0.1, target_hz) - 1.0)
        shadow_promoted = self.bridge.shadow_promoted_total()
        print(
            "CAMERA_TRACK_FINAL "
            f"detector_frames={applied} tracked_now={tracked} shadow_promoted={shadow_promoted} "
            f"detector={INFER_WIDTH}x{INFER_HEIGHT}/micro{MICRO_BATCH} "
            f"target_hz={target_hz:.1f}/cam approx_skip={expected_skip:.1f}frames "
            f"actual_hz=[{rate_text}] result_age={age_ms:.0f}ms "
            f"tracker={self.tracker_width}x{self.tracker_height} "
            f"config={self.tracker_config} timestamp_comp=1 shadow_output=1",
            flush=True,
        )
        return keep


def main() -> int:
    return CameraPersonTrackingFinal().run()


if __name__ == "__main__":
    raise SystemExit(main())
