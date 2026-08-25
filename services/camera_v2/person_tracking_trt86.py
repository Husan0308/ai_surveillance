from __future__ import annotations

import os
import queue as pyqueue
import time
from collections import deque

# Set geometry/rate before importing detection constants.
os.environ.setdefault("CAMERA_V2_DETECT_WIDTH", "672")
os.environ.setdefault("CAMERA_V2_DETECT_HEIGHT", "384")
os.environ.setdefault("CAMERA_V2_MICRO_BATCH", "1")
os.environ.setdefault("CAMERA_V2_DETECT_CONF", "0.05")
os.environ.setdefault("CAMERA_V2_DETECT_IOU", "0.70")
os.environ.setdefault("CAMERA_V2_MAX_DET", "40")
os.environ.setdefault("CAMERA_V2_DETECT_ACTIVE_CAMERAS", "CAM-01")
os.environ.setdefault("CAMERA_V2_DETECT_TARGET_HZ", "2.0")
os.environ.setdefault("CAMERA_V2_DETECT_MIN_HZ", "1.8")
os.environ.setdefault("CAMERA_V2_DETECT_MAX_HZ", "2.3")
# 160 ms is below the observed 154-190 ms TensorRT round-trip and causes valid
# results to be discarded. Keep a hard correctness floor and adapt above it.
os.environ.setdefault("CAMERA_V2_MAX_DETECT_RESULT_AGE_MS", "350")
os.environ.setdefault("CAMERA_V2_TRACKER_WIDTH", "512")
os.environ.setdefault("CAMERA_V2_TRACKER_HEIGHT", "288")

from . import detection as detection_module
from .yolo_trt86_shm_bridge import yolo_trt86_shm_worker

# CameraDetectionV2.run resolves this module global at runtime.
detection_module._yolo_worker = yolo_trt86_shm_worker

from .person_tracking_final import CameraPersonTrackingFinal


class CameraPersonTrackingTRT86(CameraPersonTrackingFinal):
    """CAM-01 correctness runtime: YOLO26 TRT8.6 SHM + per-frame NvDCF.

    The inference side branch is intentionally sparse: the upstream pad probe only
    lets a frame through when the detector scheduler asks for one. GstBaseSink
    normally waits for a preroll buffer during the PAUSED state transition. That
    is a bad fit for a sparse/gated appsink and can deadlock the capture path:
    display keeps moving, but the detector mailbox never receives a frame. This
    runtime therefore disables async preroll on every detector appsink and primes
    the active camera with one bootstrap capture request.
    """

    def __init__(self) -> None:
        self._result_age_samples: deque[float] = deque(maxlen=120)
        self._configured_result_age_ms = max(
            350.0,
            float(os.environ.get("CAMERA_V2_MAX_DETECT_RESULT_AGE_MS", "350")),
        )
        # _add_camera() is dispatched while parent constructors build the graph,
        # so these sets must exist before super().__init__().
        self._capture_gate_logged: set[str] = set()
        self._capture_sample_logged: set[str] = set()
        self._capture_sink_logged: set[str] = set()
        super().__init__()
        self.max_detector_result_age_ms = self._configured_result_age_ms

    @staticmethod
    def _active_camera_set() -> set[str]:
        values = {
            value.strip()
            for value in os.environ.get(
                "CAMERA_V2_DETECT_ACTIVE_CAMERAS", "CAM-01"
            ).split(",")
            if value.strip()
        }
        return values or {"CAM-01"}

    def _add_camera(self, index, camera) -> None:
        """Build the normal branch, then make its sparse appsink preroll-safe."""
        super()._add_camera(index, camera)
        cid = camera.camera_id
        appsink = self.pipeline.get_by_name(f"detect_sink_{index}")
        if appsink is None:
            raise RuntimeError(f"{cid}: detector appsink was not created")

        # GstBaseSink async=FALSE is the documented mode for sparse streams: the
        # sink enters PAUSED immediately instead of waiting for a preroll buffer.
        # sync=FALSE remains set by CameraDetectionV2, so inference never waits on
        # the presentation clock either.
        appsink.set_property("async", False)
        appsink.set_property("sync", False)
        self._set_if(appsink, "wait-on-eos", False)

        # Prime only detector-active cameras. This lets the first decoded frame
        # traverse the gate as soon as streaming starts and gives the scheduler a
        # fresh mailbox row even before its first request/timeout cycle finishes.
        if cid in self._active_camera_set():
            with self.capture_lock:
                self.capture_requested[cid] = True

        if cid not in self._capture_sink_logged:
            self._capture_sink_logged.add(cid)
            print(
                "CAM01_TRT86_CAPTURE_SETUP "
                f"camera={cid} appsink_async=0 sync=0 "
                f"bootstrap={int(cid in self._active_camera_set())}",
                flush=True,
            )

    def _infer_gate_probe(self, pad, info, cid: str):
        result = super()._infer_gate_probe(pad, info, cid)
        if (
            result == self.Gst.PadProbeReturn.OK
            and cid in self._active_camera_set()
            and cid not in self._capture_gate_logged
        ):
            self._capture_gate_logged.add(cid)
            print(
                f"CAM01_TRT86_CAPTURE_GATE camera={cid} first_buffer_passed=1",
                flush=True,
            )
        return result

    def _on_infer_sample(self, sink, cid: str):
        first = cid not in self._capture_sample_logged
        result = super()._on_infer_sample(sink, cid)
        if first:
            self._capture_sample_logged.add(cid)
            print(
                f"CAM01_TRT86_CAPTURE_SAMPLE camera={cid} first_sample=1",
                flush=True,
            )
        return result

    @staticmethod
    def _p95_local(values: deque[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
        return float(ordered[index])

    def _update_freshness_budget(self, ages_ms: list[float]) -> None:
        self._result_age_samples.extend(float(v) for v in ages_ms)
        p95 = self._p95_local(self._result_age_samples)
        adaptive = p95 * 1.60 + 40.0 if p95 > 0.0 else 350.0
        # Enough headroom for Pascal jitter, but never let old detector results
        # linger for seconds. NvDCF owns continuity between detector refreshes.
        self.max_detector_result_age_ms = min(
            800.0,
            max(self._configured_result_age_ms, adaptive),
        )

    def _scheduler(self) -> None:
        assert self.result_q is not None and self.job_q is not None
        try:
            ready = self.result_q.get(timeout=40.0)
        except pyqueue.Empty:
            with self.det_lock:
                self.det_error = "YOLO TRT86 worker startup timeout"
            return
        if ready.get("type") != "ready":
            with self.det_lock:
                self.det_error = ready.get("error", "YOLO TRT86 worker failed")
            return

        with self.det_lock:
            self.det_ready = True

        all_ids = [camera.camera_id for camera in self.cameras]
        allowed = self._active_camera_set()
        ids = [cid for cid in all_ids if cid in allowed]
        if not ids:
            raise RuntimeError(
                "CAMERA_V2_DETECT_ACTIVE_CAMERAS selected no cameras"
            )

        print(
            "CAMERA_TRACK_FINAL ready: "
            f"model={ready.get('model')} micro_batch=1 "
            f"input=672x384 conf={os.environ.get('CAMERA_V2_DETECT_CONF')} "
            f"iou={os.environ.get('CAMERA_V2_DETECT_IOU')} "
            f"target={self.detector_target_hz:.1f}Hz/cam "
            f"range={self.detector_min_hz:.1f}-{self.detector_max_hz:.1f}Hz/cam "
            f"tracker={self.tracker_width}x{self.tracker_height} "
            f"max_result_age={self.max_detector_result_age_ms:.0f}ms "
            f"device={ready.get('device')} cuda={ready.get('cuda')} "
            f"active={','.join(ids)} backend={ready.get('backend')} "
            "capture=sparse-gate-appsink-async0-bootstrap",
            flush=True,
        )

        groups = [[cid] for cid in ids]
        versions = {cid: 0 for cid in ids}
        group_index = 0
        prefetched_group: tuple[str, ...] | None = None

        while not self.det_stop.is_set():
            cycle_started = time.monotonic()
            group = groups[group_index % len(groups)]
            group_index += 1
            group_key = tuple(group)

            if prefetched_group != group_key:
                self._request_group(group)
            rows = self.mailbox.wait_group(group, versions, timeout=0.8)
            prefetched_group = None
            if rows is None:
                self._clear_requests()
                with self.det_lock:
                    self.capture_timeouts += 1
                    timeout_count = self.capture_timeouts
                if timeout_count <= 3 or timeout_count % 20 == 0:
                    print(
                        "CAM01_TRT86_CAPTURE_TIMEOUT "
                        f"count={timeout_count} waiting={','.join(group)} "
                        f"gate_seen={int(all(cid in self._capture_gate_logged for cid in group))} "
                        f"sample_seen={int(all(cid in self._capture_sample_logged for cid in group))}",
                        flush=True,
                    )
                self.det_stop.wait(0.025)
                continue

            frames = []
            captured = []
            for cid, row in zip(group, rows):
                version, captured_t, frame = row
                versions[cid] = version
                captured.append(captured_t)
                frames.append(frame)
            self._clear_requests()

            next_group = groups[group_index % len(groups)]
            self._request_group(next_group)
            prefetched_group = tuple(next_group)

            try:
                self.job_q.put(
                    {
                        "cameras": group,
                        "frames": frames,
                        "captured": captured,
                    },
                    timeout=0.3,
                )
                result = self.result_q.get(timeout=5.0)
            except pyqueue.Empty:
                with self.det_lock:
                    self.det_error = "YOLO TRT86 result timeout"
                self.det_stop.wait(0.05)
                continue

            if result.get("type") == "fatal":
                with self.det_lock:
                    self.det_error = result.get(
                        "error", "YOLO TRT86 fatal error"
                    )
                return
            if result.get("type") == "batch_error":
                with self.det_lock:
                    self.det_error = result.get(
                        "error", "YOLO TRT86 batch error"
                    )
                self.det_stop.wait(0.10)
                continue
            if result.get("type") != "result":
                continue

            completed_t = time.monotonic()
            counts: dict[str, int] = {}
            ages_ms: list[float] = []

            for cid, captured_t in zip(
                result["cameras"], result["captured"]
            ):
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
                self.detector_times[cid].append(completed_t)

            self._update_freshness_budget(ages_ms)
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
            self.det_stop.wait(min(0.25, idle))


def main() -> int:
    return CameraPersonTrackingTRT86().run()


if __name__ == "__main__":
    raise SystemExit(main())
