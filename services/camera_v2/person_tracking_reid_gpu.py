from __future__ import annotations

"""CAM-01 production test runtime: GPU pose detector + NvDCF + ReID.

The detector only refreshes NvDCF. NvDCF owns the bbox on every live video frame,
so a short detector miss or a slow pose inference does not make the visible box
blink off. This mirrors the stable CAM-01 behaviour used during yesterday's
single-camera tuning.
"""

import os
import queue as pyqueue
import time

# Install the old pose/keypoint detector before importing the NvDCF/ReID stack.
# The install function replaces detection._yolo_worker, which is the spawn target
# CameraDetectionV2 uses for the asynchronous detector process.
from .yolo_pose_backend import install as _install_pose_backend

_install_pose_backend()

from .detection import INFER_HEIGHT, INFER_WIDTH, MICRO_BATCH
from .person_tracking_reid import CameraPersonTrackingReID


class CameraPersonTrackingReIDGpu(CameraPersonTrackingReID):
    """GPU YOLO26s-pose detections feeding sticky camera-local NvDCF tracks."""

    def __init__(self) -> None:
        super().__init__()

        # detection.py deliberately gates the inference branch before appsink so
        # only requested frames are converted/copied. A normal GstBaseSink waits
        # for a preroll buffer during the PAUSED -> PLAYING transition. Because
        # our inference stream is intentionally sparse, that can leave appsink
        # waiting for preroll while the scheduler waits for appsink: mailbox=[]
        # forever. Disable async state changes only on the sparse analysis sinks;
        # the display sink remains unchanged.
        sparse_sinks = []
        for index, _camera in enumerate(self.cameras):
            sink = self.pipeline.get_by_name(f"detect_sink_{index}")
            if sink is None:
                continue
            self._set_if(sink, "async", False)
            self._set_if(sink, "sync", False)
            self._set_if(sink, "qos", False)
            sparse_sinks.append(index)

        print(
            "CAMERA_GPU_SPARSE_APPSINK "
            f"async=0 sync=0 qos=0 sinks={sparse_sinks}",
            flush=True,
        )

        # The current Pascal/driver combination can return a pose refresh several
        # hundred milliseconds after capture. Do not inject that old coordinate
        # unchanged into the live tracker: project it toward the current frame.
        # NvDCF still remains authoritative between detector observations.
        self.latency_compensator.max_projection_s = 0.45
        self.latency_compensator.projection_gain = 0.82

    def _active_detector_ids(self) -> list[str]:
        configured = [
            value.strip()
            for value in os.environ.get(
                "CAMERA_V2_DETECT_ACTIVE_CAMERAS",
                "",
            ).split(",")
            if value.strip()
        ]
        all_ids = [camera.camera_id for camera in self.cameras]
        if not configured:
            return all_ids
        allowed = set(configured)
        ids = [cid for cid in all_ids if cid in allowed]
        if not ids:
            raise RuntimeError(
                "CAMERA_V2_DETECT_ACTIVE_CAMERAS selected no configured cameras"
            )
        return ids

    def _scheduler(self) -> None:
        assert self.result_q is not None and self.job_q is not None
        try:
            ready = self.result_q.get(timeout=40.0)
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
            "CAMERA_GPU_NVDCF ready: "
            f"backend={ready.get('backend', 'YOLO26s-pose')} "
            f"model={ready.get('model')} "
            f"active={active_ids} input={INFER_WIDTH}x{INFER_HEIGHT} "
            f"pose_imgsz={ready.get('imgsz')} conf={ready.get('threshold')} "
            f"target={self.detector_target_hz:.1f}Hz/cam "
            f"tracker={self.tracker_width}x{self.tracker_height} "
            f"max_result_age={self.max_detector_result_age_ms:.0f}ms "
            f"device={ready.get('device')} cuda={ready.get('cuda')} "
            "policy=detector-refreshes-nvdcf nvdcf-per-frame=1 shadow-hold=1",
            flush=True,
        )

        groups = [
            active_ids[i : i + MICRO_BATCH]
            for i in range(0, len(active_ids), MICRO_BATCH)
        ]
        versions = {cid: 0 for cid in active_ids}
        group_index = 0
        prefetched_group: tuple[str, ...] | None = None
        consecutive_capture_timeouts = 0

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
                consecutive_capture_timeouts += 1
                with self.det_lock:
                    self.capture_timeouts += 1
                if consecutive_capture_timeouts in {3, 10, 30}:
                    print(
                        "CAMERA_GPU_CAPTURE_WAIT "
                        f"group={group} consecutive={consecutive_capture_timeouts} "
                        f"mailbox={sorted(self.mailbox.rows)} "
                        "hint=inference-appsink-not-delivering",
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
                    cid,
                    captured_t,
                    detections,
                )
                self._publish_prepared(cid, captured_t, prepared)
                counts[cid] = len(detections)
                ages_ms.append(
                    max(0.0, (completed_t - captured_t) * 1000.0)
                )
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
                0.1,
                target_hz * len(groups),
            )
            elapsed = time.monotonic() - cycle_started
            idle = max(
                self.detector_min_idle,
                desired_call_interval - elapsed,
            )
            self.det_stop.wait(min(0.25, idle))


def main() -> int:
    return CameraPersonTrackingReIDGpu().run()


if __name__ == "__main__":
    raise SystemExit(main())
