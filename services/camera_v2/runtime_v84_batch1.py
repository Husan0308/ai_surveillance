from __future__ import annotations

import multiprocessing as mp
import os
import queue as pyqueue
import threading
import time
from collections import deque

from .runtime_v8_pascal import PascalBatchLowLatencyRuntime
from .yolo_trt86_shm_bridge import yolo_trt86_shm_worker


class PascalBatch1LowLatencyRuntime(PascalBatchLowLatencyRuntime):
    """V8.4 Step 2: batch-1 low-latency detector on Pascal.

    Clean-room measurements on the target GTX 1050 Ti showed batch=1 ~=60 ms and
    batch=6 ~=349.5 ms, i.e. almost identical per-image throughput but ~6x larger
    result latency for batch=6. This runtime therefore keeps the proven V8 camera,
    display and NvDCF topology, but replaces the all-camera batch-6 detector with a
    single persistent TRT8.6 batch-1 worker scheduled fairly across cameras.

    Important invariants:
      * no Python GPU lane and no tracker drops for detector work;
      * only one latest-frame capture is requested per detector cycle;
      * detector cadence is a GLOBAL call rate, divided fairly round-robin across
        live cameras, and adapts from measured integrated GPU latency;
      * stale queues cannot build up: job queue depth is one and each cycle waits for
        the previous result before requesting another camera.
    """

    def __init__(self) -> None:
        self.v84_budget = max(
            0.10,
            min(0.45, float(os.environ.get("CAMERA_V84_DETECT_GPU_BUDGET", "0.30"))),
        )
        self.v84_global_min_hz = max(
            0.6,
            min(4.0, float(os.environ.get("CAMERA_V84_GLOBAL_MIN_HZ", "1.50"))),
        )
        self.v84_global_max_hz = max(
            self.v84_global_min_hz,
            min(10.0, float(os.environ.get("CAMERA_V84_GLOBAL_MAX_HZ", "6.00"))),
        )
        self.v84_global_hz = max(
            self.v84_global_min_hz,
            min(
                self.v84_global_max_hz,
                float(os.environ.get("CAMERA_V84_GLOBAL_INITIAL_HZ", "4.00")),
            ),
        )
        self.v84_capture_timeout = max(
            0.05,
            min(0.30, float(os.environ.get("CAMERA_V84_CAPTURE_TIMEOUT", "0.12"))),
        )
        self.v84_ema_alpha = max(
            0.05,
            min(0.50, float(os.environ.get("CAMERA_V84_EMA_ALPHA", "0.20"))),
        )
        self.v84_calls = 0
        self.v84_capture_miss = 0
        self.v84_gpu_ms_ema = 0.0
        self.v84_roundtrip_ms_ema = 0.0
        self.v84_result_age_ms = 0.0
        self.v84_intervals = deque(maxlen=180)
        self.v84_last_complete = 0.0
        self.v84_per_camera_calls: dict[str, int] = {}
        super().__init__()
        self.detect_hz = self.v84_global_hz / max(1, len(self.cameras))
        self.v84_per_camera_calls = {camera.camera_id: 0 for camera in self.cameras}
        print(
            "CAMERA_V84_ARCH "
            f"detector=TRT8.6/batch1-roundrobin cameras={len(self.cameras)} "
            f"global_initial={self.v84_global_hz:.2f}Hz "
            f"per_camera_initial={self.detect_hz:.2f}Hz "
            f"budget={self.v84_budget:.2f} "
            "batch6=disabled coalesced_wait=disabled gpu_lane=0 "
            "tracker_drop_for_detector=0 latest_only=1",
            flush=True,
        )

    def _start_detector(self) -> None:
        if not self.detect_enabled:
            print("CAMERA_V84_DETECT enabled=0", flush=True)
            return
        ctx = mp.get_context("spawn")
        self.job_q = ctx.Queue(maxsize=1)
        self.result_q = ctx.Queue(maxsize=2)
        self.det_process = ctx.Process(
            target=yolo_trt86_shm_worker,
            args=(self.job_q, self.result_q),
            name="camera-v84-trt86-batch1-bridge",
        )
        self.det_process.start()
        self.det_thread = threading.Thread(
            target=self._detector_scheduler_v84,
            name="camera-v84-detector-roundrobin",
            daemon=True,
        )
        self.det_thread.start()

    @staticmethod
    def _ema_v84(previous: float, current: float, alpha: float) -> float:
        if previous <= 0.0:
            return float(current)
        return previous * (1.0 - alpha) + float(current) * alpha

    def _adapt_v84(self) -> None:
        gpu_s = max(0.001, self.v84_gpu_ms_ema / 1000.0)
        wall_s = max(0.001, self.v84_roundtrip_ms_ema / 1000.0)
        # Global call frequency. Keep detector below its assigned GPU duty and leave
        # at least 20% wall-time slack so one slow call cannot self-backlog.
        by_gpu = self.v84_budget / gpu_s
        by_wall = 0.80 / wall_s
        target = min(self.v84_global_max_hz, by_gpu, by_wall)
        self.v84_global_hz = max(self.v84_global_min_hz, target)
        self.detect_hz = self.v84_global_hz / max(1, len(self.cameras))

    def _detector_scheduler_v84(self) -> None:
        assert self.result_q is not None and self.job_q is not None
        try:
            ready = self.result_q.get(timeout=60.0)
        except pyqueue.Empty:
            with self.det_lock:
                self.det_error = "V84 TRT86 batch1 startup timeout"
            return
        if ready.get("type") != "ready":
            with self.det_lock:
                self.det_error = ready.get("error", "V84 TRT86 batch1 startup failed")
            return
        with self.det_lock:
            self.det_ready = True
        print(
            "CAMERA_V84_DETECT_READY "
            f"model={ready.get('model')} backend={ready.get('backend')} batch=1 "
            f"global={self.v84_global_hz:.2f}Hz "
            f"per_camera={self.detect_hz:.2f}Hz "
            "capture=one-latest-camera-at-a-time queue_depth=1 gpu_lane=none",
            flush=True,
        )

        ids = [camera.camera_id for camera in self.cameras]
        versions = {cid: 0 for cid in ids}
        rr = 0

        while not self.det_stop.is_set():
            cycle_started = time.monotonic()
            # Find the next camera that has produced at least one source frame. Do not
            # let one offline camera stall detection for the other five.
            selected = None
            for offset in range(len(ids)):
                cid = ids[(rr + offset) % len(ids)]
                if self.stats[cid].frames > 0:
                    selected = cid
                    rr = (rr + offset + 1) % len(ids)
                    break
            if selected is None:
                self.det_stop.wait(0.03)
                continue

            cid = selected
            self._request_capture(cid)
            row = self.mailbox.wait_new(
                cid,
                versions[cid],
                timeout=self.v84_capture_timeout,
            )
            self._clear_capture(cid)
            if row is None:
                self.v84_capture_miss += 1
                with self.det_lock:
                    self.capture_timeouts += 1
                self.det_stop.wait(0.003)
                continue

            version, captured_at, frame = row
            versions[cid] = version
            try:
                self.job_q.put(
                    {"cameras": [cid], "frames": [frame], "captured": [captured_at]},
                    timeout=0.10,
                )
                result = self.result_q.get(timeout=5.0)
            except pyqueue.Empty:
                with self.det_lock:
                    self.det_error = "V84 TRT86 batch1 result timeout"
                self.det_stop.wait(0.03)
                continue

            if result.get("type") == "fatal":
                with self.det_lock:
                    self.det_error = result.get("error", "V84 TRT86 batch1 fatal")
                return
            if result.get("type") != "result":
                continue

            completed = time.monotonic()
            gpu_ms = float(result.get("batch_ms") or 0.0)
            roundtrip_ms = float(result.get("total_ms") or gpu_ms)
            self.v84_gpu_ms_ema = self._ema_v84(
                self.v84_gpu_ms_ema, gpu_ms, self.v84_ema_alpha
            )
            self.v84_roundtrip_ms_ema = self._ema_v84(
                self.v84_roundtrip_ms_ema, roundtrip_ms, self.v84_ema_alpha
            )
            self._adapt_v84()

            if self.v84_last_complete > 0.0:
                self.v84_intervals.append(completed - self.v84_last_complete)
            self.v84_last_complete = completed
            self.v84_calls += 1
            self.v84_per_camera_calls[cid] += 1

            detector_rows = result.get("boxes", {}).get(cid, [])
            boxes = self._map_detector_rows(detector_rows)
            self._publish_detector(cid, captured_at, boxes)
            age_ms = max(0.0, (completed - captured_at) * 1000.0)
            self.v84_result_age_ms = age_ms

            with self.det_lock:
                self.det_counts[cid] = len(boxes)
                self.detector_times[cid].append(completed)
                self.det_calls += 1
                self.det_inputs += 1
                self.det_batch_ms = gpu_ms
                self.det_result_age_ms = age_ms
                self.det_error = ""

            if self.v84_calls <= 8 or self.v84_calls % 20 == 0:
                duty = self.v84_global_hz * max(0.0, self.v84_gpu_ms_ema) / 1000.0
                print(
                    "CAMERA_V84_DETECT "
                    f"call={self.v84_calls} camera={cid} gpu={gpu_ms:.1f}ms "
                    f"gpu_ema={self.v84_gpu_ms_ema:.1f}ms "
                    f"roundtrip_ema={self.v84_roundtrip_ms_ema:.1f}ms "
                    f"age={age_ms:.0f}ms global_hz={self.v84_global_hz:.2f} "
                    f"per_camera_hz={self.detect_hz:.2f} duty_est={duty:.2f} "
                    f"boxes={len(boxes)}",
                    flush=True,
                )

            desired = 1.0 / max(0.1, self.v84_global_hz)
            elapsed = time.monotonic() - cycle_started
            self.det_stop.wait(max(0.001, desired - elapsed))

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        intervals = list(self.v84_intervals)
        actual_global_hz = 0.0
        if intervals:
            avg = sum(intervals) / len(intervals)
            actual_global_hz = 1.0 / avg if avg > 0.0 else 0.0
        duty = self.v84_global_hz * max(0.0, self.v84_gpu_ms_ema) / 1000.0
        fairness = ",".join(
            f"{cid}:{self.v84_per_camera_calls.get(cid, 0)}" for cid in sorted(self.v84_per_camera_calls)
        )
        print(
            "CAMERA_V84_STATS "
            f"calls={self.v84_calls} global_target={self.v84_global_hz:.2f}Hz "
            f"global_actual={actual_global_hz:.2f}Hz per_camera_target={self.detect_hz:.2f}Hz "
            f"gpu_ema={self.v84_gpu_ms_ema:.1f}ms "
            f"roundtrip_ema={self.v84_roundtrip_ms_ema:.1f}ms "
            f"gpu_duty_est={duty:.2f} result_age={self.v84_result_age_ms:.0f}ms "
            f"capture_miss={self.v84_capture_miss} fairness=[{fairness}] "
            "batch=1 queue_depth=1 gpu_lane=0",
            flush=True,
        )
        return keep


def main() -> int:
    return PascalBatch1LowLatencyRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
