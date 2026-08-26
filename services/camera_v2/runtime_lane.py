from __future__ import annotations

import queue as pyqueue
import threading
import time
from collections import deque

from .runtime_quality import QualityCameraRuntime


class SerializedGpuLaneRuntime(QualityCameraRuntime):
    """Serialize NvDCF and TRT8.6 work on one Pascal GPU lane.

    Isolation tests on the GTX 1050 Ti prove that NvDCF alone sustains the six
    20 FPS streams and TRT8.6 alone runs at roughly 15-25 ms, while running both
    CUDA workloads concurrently inflates TRT latency to roughly 160-180 ms and
    pulls several source streams down into the low teens. DeepStream and the TRT
    8.6 sidecar are separate CUDA processes/contexts, so this runtime avoids the
    destructive overlap explicitly instead of lowering camera/display quality.

    Display/NVDEC remain outside this lock. A tracker batch acquires the lane just
    before NvDCF and releases it at the NvDCF source pad. The detector scheduler
    acquires the same lane before requesting a fresh frame and holds it only until
    the TRT result returns. Tracker frames that arrive while TRT owns the lane are
    dropped at the analytics mux; their latest-only input queues prevent backlog.
    """

    def __init__(self) -> None:
        self.gpu_lane = threading.Lock()
        self.lane_stats_lock = threading.Lock()
        self._tracker_lane_held = False
        self.lane_tracker_skips = 0
        self.lane_detector_wait_ms = deque(maxlen=120)
        self.lane_detector_hold_ms = deque(maxlen=120)
        super().__init__()
        print(
            "CAMERA_GPU_LANE mode=serialized tracker=NvDCF detector=TRT86 "
            "display=independent policy=detector-waits-tracker+tracker-drops-during-detector",
            flush=True,
        )

    @staticmethod
    def _p95(values) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
        return float(ordered[index])

    def _inject_detector_probe(self, pad, info):
        if not self.analytics_enabled:
            return super()._inject_detector_probe(pad, info)

        # Never block the tracker streaming branch behind TRT. If the detector
        # owns the GPU lane, drop this analytics batch; the next 10 Hz batch will
        # arrive shortly and display is on a separate tee branch.
        if not self.gpu_lane.acquire(blocking=False):
            with self.lane_stats_lock:
                self.lane_tracker_skips += 1
            return self.Gst.PadProbeReturn.DROP

        self._tracker_lane_held = True
        try:
            return super()._inject_detector_probe(pad, info)
        except Exception:
            self._tracker_lane_held = False
            self.gpu_lane.release()
            raise

    def _tracker_probe(self, pad, info):
        try:
            return super()._tracker_probe(pad, info)
        finally:
            if self._tracker_lane_held:
                self._tracker_lane_held = False
                self.gpu_lane.release()

    def _detector_scheduler(self) -> None:
        assert self.result_q is not None and self.job_q is not None
        try:
            ready = self.result_q.get(timeout=40.0)
        except pyqueue.Empty:
            with self.det_lock:
                self.det_error = "TRT86 startup timeout"
            return
        if ready.get("type") != "ready":
            with self.det_lock:
                self.det_error = ready.get("error", "TRT86 startup failed")
            return
        with self.det_lock:
            self.det_ready = True
        print(
            "CAMERA_CLEAN_DETECT_READY "
            f"model={ready.get('model')} backend={ready.get('backend')} "
            f"target={self.detect_hz:.2f}Hz/cam capture=jit-latest-no-prefetch "
            "gpu_lane=serialized",
            flush=True,
        )

        ids = [camera.camera_id for camera in self.cameras]
        versions = {cid: 0 for cid in ids}
        index = 0

        while not self.det_stop.is_set():
            cycle_started = time.monotonic()
            cid = ids[index % len(ids)]
            index += 1

            if self.stats[cid].frames <= 0:
                self.det_stop.wait(0.03)
                continue

            wait_started = time.monotonic()
            acquired = False
            while not self.det_stop.is_set():
                if self.gpu_lane.acquire(timeout=0.05):
                    acquired = True
                    break
            if not acquired:
                break

            lane_started = time.monotonic()
            lane_wait_ms = (lane_started - wait_started) * 1000.0
            try:
                # Capture only after the detector owns the GPU lane. This keeps the
                # TensorRT input fresh even when we had to wait for an NvDCF batch.
                self._request_capture(cid)
                row = self.mailbox.wait_new(cid, versions[cid], timeout=0.8)
                if row is None:
                    self._clear_capture(cid)
                    with self.det_lock:
                        self.capture_timeouts += 1
                    self.det_stop.wait(0.025)
                    continue

                version, captured, frame = row
                versions[cid] = version
                self._clear_capture(cid)

                try:
                    self.job_q.put(
                        {"cameras": [cid], "frames": [frame], "captured": [captured]},
                        timeout=0.3,
                    )
                    result = self.result_q.get(timeout=5.0)
                except pyqueue.Empty:
                    with self.det_lock:
                        self.det_error = "TRT86 result timeout"
                    self.det_stop.wait(0.05)
                    continue

                if result.get("type") == "fatal":
                    with self.det_lock:
                        self.det_error = result.get("error", "TRT86 fatal")
                    return
                if result.get("type") == "batch_error":
                    with self.det_lock:
                        self.det_error = result.get("error", "TRT86 batch error")
                    self.det_stop.wait(0.10)
                    continue
                if result.get("type") != "result":
                    continue

                completed = time.monotonic()
                rows = result.get("boxes", {}).get(cid, [])
                boxes = self._map_detector_rows(rows)
                self._publish_detector(cid, captured, boxes)
                age_ms = max(0.0, (completed - captured) * 1000.0)
                batch_ms = float(result.get("batch_ms") or 0.0)
                with self.det_lock:
                    self.det_calls += 1
                    self.det_inputs += 1
                    self.det_batch_ms = batch_ms
                    self.det_counts[cid] = len(boxes)
                    self.det_result_age_ms = age_ms
                    self.detector_times[cid].append(completed)
                    self.det_error = ""
            finally:
                lane_hold_ms = (time.monotonic() - lane_started) * 1000.0
                with self.lane_stats_lock:
                    self.lane_detector_wait_ms.append(lane_wait_ms)
                    self.lane_detector_hold_ms.append(lane_hold_ms)
                self.gpu_lane.release()

            desired_interval = 1.0 / max(0.1, self.detect_hz * len(ids))
            elapsed = time.monotonic() - cycle_started
            self.det_stop.wait(max(0.005, desired_interval - elapsed))

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        with self.lane_stats_lock:
            waits = list(self.lane_detector_wait_ms)
            holds = list(self.lane_detector_hold_ms)
            skips = int(self.lane_tracker_skips)
        wait_avg = sum(waits) / len(waits) if waits else 0.0
        hold_avg = sum(holds) / len(holds) if holds else 0.0
        print(
            "CAMERA_GPU_LANE_STATS "
            f"det_wait_avg={wait_avg:.1f}ms det_wait_p95={self._p95(waits):.1f}ms "
            f"det_hold_avg={hold_avg:.1f}ms det_hold_p95={self._p95(holds):.1f}ms "
            f"tracker_skips={skips}",
            flush=True,
        )
        return keep


def main() -> int:
    return SerializedGpuLaneRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
