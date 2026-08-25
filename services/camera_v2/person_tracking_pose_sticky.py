from __future__ import annotations

"""CAM-01 pose-validated detector + sticky per-frame NvDCF tracking.

YOLO26s-pose refreshes a person on sparse, fresh frames. DeepStream NvDCF owns
the bbox on every display frame. This runtime is deliberately local-camera only;
Global ID/ReID is added only after this detection/tracking baseline is proven.

Important sparse-tracker rule: detector refreshes are much slower than video
frames, so a new NvDCF target must not spend a multi-frame probation period and
be early-terminated before the next detector refresh arrives. We therefore make
pose-seeded targets active immediately (probationAge=0) and let shadow tracking
own continuity between detector observations.
"""

import os
import queue as pyqueue
import time

# Patch only the generated local NvDCF profile used by this runtime. The generic
# tracker profile remains unchanged for the rollback/known-good branches.
from . import tracker_profile as _tracker_profile

_tracker_profile._REQUIRED_PATCHES.update(
    {
        "minDetectorConfidence": "0.05",
        "minTrackerConfidence": "0.12",
        "probationAge": "0",
        "maxShadowTrackingAge": "50",
        "earlyTerminationAge": "6",
    }
)
_tracker_profile._OPTIONAL_PATCHES.update(
    {
        "minTrackingConfidenceDuringInactive": "0.08",
        "tentativeDetectorConfidence": "0.05",
    }
)

from .yolo_pose_backend import install as _install_pose_backend

# Must happen before CameraPersonTrackingFinal imports the detector worker target.
_install_pose_backend()

from .detection import INFER_HEIGHT, INFER_WIDTH, MICRO_BATCH
from .person_tracking_final import CameraPersonTrackingFinal


class CameraPersonTrackingPoseSticky(CameraPersonTrackingFinal):
    """YOLO26s-pose refreshes; NvDCF owns camera-local temporal tracking."""

    def __init__(self) -> None:
        super().__init__()

        # Sparse appsinks must not block PAUSED->PLAYING waiting for preroll.
        sparse_sinks: list[int] = []
        for index, _camera in enumerate(self.cameras):
            sink = self.pipeline.get_by_name(f"detect_sink_{index}")
            if sink is None:
                continue
            self._set_if(sink, "async", False)
            self._set_if(sink, "sync", False)
            self._set_if(sink, "qos", False)
            sparse_sinks.append(index)

        print(
            "CAMERA_POSE_SPARSE_APPSINK "
            f"async=0 sync=0 qos=0 sinks={sparse_sinks}",
            flush=True,
        )
        print(
            "CAMERA_POSE_NVDCF_PROFILE "
            "probationAge=0 earlyTerminationAge=6 maxShadowTrackingAge=50 "
            "minDetectorConfidence=0.05 minTrackerConfidence=0.12 "
            "inactiveOutputConfidence=0.08 outputShadowTracks=1",
            flush=True,
        )

        # Pose inference on Pascal can finish several hundred milliseconds after
        # capture. Project a stable detector observation toward the live frame;
        # NvDCF remains authoritative between refreshes.
        self.latency_compensator.max_projection_s = float(
            os.environ.get("CAMERA_V2_POSE_MAX_PROJECTION_S", "0.45")
        )
        self.latency_compensator.projection_gain = float(
            os.environ.get("CAMERA_V2_POSE_PROJECTION_GAIN", "0.82")
        )

        self.empty_confirm_misses = max(
            2,
            int(os.environ.get("CAMERA_V2_EMPTY_CONFIRM_MISSES", "3")),
        )
        self._empty_detector_streak = {
            camera.camera_id: 0 for camera in self.cameras
        }

    def _publish_prepared(self, cid: str, captured_t: float, prepared) -> None:
        if prepared:
            self._empty_detector_streak[cid] = 0
            return super()._publish_prepared(cid, captured_t, prepared)

        streak = self._empty_detector_streak.get(cid, 0) + 1
        self._empty_detector_streak[cid] = streak
        if streak < self.empty_confirm_misses:
            print(
                "CAMERA_POSE_EMPTY_HOLD "
                f"cid={cid} miss={streak}/{self.empty_confirm_misses} action=keep-nvdcf",
                flush=True,
            )
            return

        self._empty_detector_streak[cid] = 0
        print(
            "CAMERA_POSE_EMPTY_CONFIRM "
            f"cid={cid} misses={self.empty_confirm_misses} action=publish-empty",
            flush=True,
        )
        return super()._publish_prepared(cid, captured_t, prepared)

    def _active_detector_ids(self) -> list[str]:
        configured = [
            value.strip()
            for value in os.environ.get(
                "CAMERA_V2_DETECT_ACTIVE_CAMERAS", ""
            ).split(",")
            if value.strip()
        ]
        all_ids = [camera.camera_id for camera in self.cameras]
        if not configured:
            return all_ids
        allowed = set(configured)
        active = [cid for cid in all_ids if cid in allowed]
        if not active:
            raise RuntimeError(
                "CAMERA_V2_DETECT_ACTIVE_CAMERAS selected no configured cameras"
            )
        return active

    def _scheduler(self) -> None:
        assert self.result_q is not None and self.job_q is not None
        try:
            ready = self.result_q.get(timeout=60.0)
        except pyqueue.Empty:
            with self.det_lock:
                self.det_error = "YOLO pose worker startup timeout"
            return

        if ready.get("type") != "ready":
            with self.det_lock:
                self.det_error = ready.get("error", "YOLO pose worker failed")
            return

        active_ids = self._active_detector_ids()
        with self.det_lock:
            self.det_ready = True

        print(
            "CAMERA_POSE_NVDCF ready: "
            f"backend={ready.get('backend', 'YOLO26s-pose')} "
            f"model={ready.get('model')} active={active_ids} "
            f"capture={INFER_WIDTH}x{INFER_HEIGHT} pose_imgsz={ready.get('imgsz')} "
            f"conf={ready.get('threshold')} target={self.detector_target_hz:.2f}Hz/cam "
            f"tracker={self.tracker_width}x{self.tracker_height} "
            f"max_result_age={self.max_detector_result_age_ms:.0f}ms "
            f"empty_confirm={self.empty_confirm_misses} "
            f"device={ready.get('device')} cuda={ready.get('cuda')} "
            "policy=pose-refreshes-nvdcf nvdcf-per-frame=1 jit-no-prefetch=1",
            flush=True,
        )

        groups = [
            active_ids[i : i + MICRO_BATCH]
            for i in range(0, len(active_ids), MICRO_BATCH)
        ]
        versions = {cid: 0 for cid in active_ids}
        group_index = 0
        consecutive_capture_timeouts = 0

        while not self.det_stop.is_set():
            cycle_started = time.monotonic()
            group = groups[group_index % len(groups)]
            group_index += 1

            # JIT latest frame only. Never request the next frame while the
            # current inference is still running.
            self._request_group(group)
            rows = self.mailbox.wait_group(group, versions, timeout=0.8)
            if rows is None:
                self._clear_requests()
                consecutive_capture_timeouts += 1
                with self.det_lock:
                    self.capture_timeouts += 1
                if consecutive_capture_timeouts in {3, 10, 30}:
                    print(
                        "CAMERA_POSE_CAPTURE_WAIT "
                        f"group={group} consecutive={consecutive_capture_timeouts} "
                        f"mailbox={sorted(self.mailbox.rows)}",
                        flush=True,
                    )
                self.det_stop.wait(0.025)
                continue

            consecutive_capture_timeouts = 0
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
                    timeout=0.3,
                )
                result = self.result_q.get(timeout=8.0)
            except pyqueue.Empty:
                with self.det_lock:
                    self.det_error = "YOLO pose result timeout"
                self.det_stop.wait(0.05)
                continue

            if result.get("type") == "fatal":
                with self.det_lock:
                    self.det_error = result.get("error", "YOLO pose fatal error")
                return
            if result.get("type") == "batch_error":
                with self.det_lock:
                    self.det_error = result.get("error", "YOLO pose batch error")
                self.det_stop.wait(0.10)
                continue
            if result.get("type") != "result":
                continue

            completed_t = time.monotonic()
            counts: dict[str, int] = {}
            ages_ms: list[float] = []

            for cid, captured_t in zip(result["cameras"], result["captured"]):
                detections = self._dedup_and_expand(
                    result["boxes"].get(cid, [])
                )
                prepared = self.latency_compensator.prepare(
                    cid, captured_t, detections
                )
                self._publish_prepared(cid, captured_t, prepared)
                counts[cid] = len(detections)
                ages_ms.append(
                    max(0.0, (completed_t - captured_t) * 1000.0)
                )
                if cid in self.detector_times:
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

            desired_call_interval = 1.0 / max(
                0.1, target_hz * len(groups)
            )
            elapsed = time.monotonic() - cycle_started
            idle = max(
                self.detector_min_idle,
                desired_call_interval - elapsed,
            )
            self.det_stop.wait(idle)

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        with self.det_lock:
            tracked = self.tracked_now
            calls = self.det_calls
            inputs = self.det_inputs
            age = self.detector_result_age_ms
            stale = self.stale_detector_results
            meta = self.meta_boxes
            timeouts = self.capture_timeouts
        print(
            "CAMERA_POSE_STICKY "
            f"calls={calls} inputs={inputs} meta_boxes={meta} tracked_now={tracked} "
            f"result_age={age:.1f}ms stale={stale} timeouts={timeouts} "
            f"empty_confirm={self.empty_confirm_misses} no_prefetch=1",
            flush=True,
        )
        return keep


def main() -> int:
    return CameraPersonTrackingPoseSticky().run()


if __name__ == "__main__":
    raise SystemExit(main())
