from __future__ import annotations

import os
import sys
import time

import numpy as np

from .batch1_trt86 import Batch1TRT86Client
from .step2_detector_only import DETECT_H, DETECT_W, V11Step2DetectorOnly


class V11Step2DetectorB1V5(V11Step2DetectorOnly):
    """Latency-first Step2 detector using one batch-1 inference at a time.

    Frozen Step1 V7 display and the existing demand-gated detector branches are kept.
    Only one camera is requested at a time, then one fresh 672x384 frame is sent to
    TensorRT 8.6 batch1. Cameras are serviced round-robin, so detector conversion
    bursts never involve all six cameras simultaneously and there is no historical
    frame queue.
    """

    def __init__(self) -> None:
        self.b1_target_hz_per_camera = max(
            1.0,
            min(4.0, float(os.environ.get("V11_DETECT_B1_HZ_PER_CAMERA", "3.0"))),
        )
        self.b1_global_target_hz = self.b1_target_hz_per_camera * 6.0
        self.b1_global_period = 1.0 / self.b1_global_target_hz
        self.b1_capture_timeout_ms = max(
            80.0,
            min(220.0, float(os.environ.get("V11_DETECT_B1_CAPTURE_TIMEOUT_MS", "140"))),
        )
        self.b1_started_mono = 0.0
        super().__init__()
        self.det_prefetch_batches = 0
        print(
            "CAMERA_V11_STEP2V5_ARCH "
            "base=step2-latest-branch display=step1-v7-frozen detector=trt86-batch1 "
            "scheduler=round-robin-jit conversions_inflight_max=1 tracker=0 osd=0 reid=0 face=0",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP2V5_POLICY "
            f"per_camera_target={self.b1_target_hz_per_camera:.2f}Hz "
            f"global_target={self.b1_global_target_hz:.2f}Hz batch=1 prefetch=0 "
            f"capture_timeout_ms={self.b1_capture_timeout_ms:.0f} latest_only=1",
            flush=True,
        )

    def _detector_scheduler(self) -> None:
        client: Batch1TRT86Client | None = None
        try:
            client = Batch1TRT86Client()
            warm = np.full((DETECT_H, DETECT_W, 3), 114, dtype=np.uint8)
            for _ in range(10):
                client.infer("WARMUP", warm, conf=self.det_conf, max_det=self.det_max_det)

            ids = [camera.camera_id for camera in self.cameras]
            with self.det_lock:
                self.det_ready = True
                self.b1_started_mono = time.monotonic()
            print(
                "CAMERA_V11_STEP2V5_DETECT_READY "
                f"engine={client.engine} worker={client.worker.name} batch=1 tensorrt=8.6.1 "
                f"per_camera_target={self.b1_target_hz_per_camera:.2f}Hz "
                f"global_target={self.b1_global_target_hz:.2f}Hz warmup=10",
                flush=True,
            )

            while not self.det_stop.is_set():
                with self.lock:
                    source_ready = all(self.stats[cid].decoded > 0 for cid in ids)
                if source_ready:
                    break
                self.det_stop.wait(0.02)
            if self.det_stop.is_set():
                return

            versions = {cid: self.mailbox.version(cid) for cid in ids}
            camera_index = 0
            next_slot = time.monotonic()

            while not self.det_stop.is_set():
                now = time.monotonic()
                if now < next_slot:
                    self.det_stop.wait(next_slot - now)
                    if self.det_stop.is_set():
                        break
                cycle_started = time.monotonic()
                # Do not accumulate schedule debt. A late cycle always schedules from
                # the actual start time, preserving latest-only semantics.
                next_slot = cycle_started + self.b1_global_period

                cid = ids[camera_index]
                camera_index = (camera_index + 1) % len(ids)
                request_started = time.monotonic()
                with self.det_lock:
                    self.capture_requested[cid] = True

                row = self.mailbox.wait_new(
                    cid,
                    versions[cid],
                    self.b1_capture_timeout_ms / 1000.0,
                )
                if row is None:
                    with self.det_lock:
                        self.capture_requested[cid] = False
                        self.det_timeouts += 1
                    versions[cid] = self.mailbox.version(cid)
                    continue

                version, captured_at, frame = row
                versions[cid] = int(version)
                captured_at = float(captured_at)
                capture_wait = max(0.0, (time.monotonic() - request_started) * 1000.0)

                result = client.infer(
                    cid,
                    frame,
                    conf=self.det_conf,
                    max_det=self.det_max_det,
                )
                completed = time.monotonic()
                result_age = max(0.0, (completed - captured_at) * 1000.0)

                with self.det_lock:
                    if self.last_batch_completed > 0.0:
                        self.batch_interval_ms.append(
                            (completed - self.last_batch_completed) * 1000.0
                        )
                    self.last_batch_completed = completed
                    self.det_batches += 1
                    self.capture_wait_ms.append(capture_wait)
                    self.input_skew_ms.append(0.0)
                    self.shm_copy_ms.append(result.shm_copy_ms)
                    self.prep_ms.append(result.prep_ms)
                    self.trt_ms.append(result.trt_ms)
                    self.roundtrip_ms.append(result.roundtrip_ms)
                    self.result_age_ms.append(result_age)
                    self.det_last_boxes[cid] = len(result.boxes)
                    self.det_result_counts[cid] += 1
                    self.det_result_age_ms[cid].append(result_age)
                    self.det_total_boxes += len(result.boxes)
                    self.det_error = ""
                    n = self.det_batches

                if n <= 12 or n % 60 == 0:
                    print(
                        "CAMERA_V11_STEP2V5_INFER "
                        f"n={n} camera={cid} capture_wait={capture_wait:.1f}ms "
                        f"shm={result.shm_copy_ms:.1f}ms prep={result.prep_ms:.1f}ms "
                        f"trt={result.trt_ms:.1f}ms roundtrip={result.roundtrip_ms:.1f}ms "
                        f"result_age={result_age:.1f}ms boxes={len(result.boxes)}",
                        flush=True,
                    )
        except BaseException as exc:
            with self.det_lock:
                self.det_error = f"{type(exc).__name__}:{exc}"
            print(
                f"CAMERA_V11_STEP2V5_DETECT_FATAL error={type(exc).__name__}:{exc}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            if client is not None:
                client.close()
            with self.det_lock:
                self.det_ready = False

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        if not keep:
            return False
        with self.det_lock:
            intervals = list(self.batch_interval_ms)
            global_hz = 0.0
            if intervals:
                mean_interval = sum(intervals) / len(intervals)
                global_hz = 1000.0 / mean_interval if mean_interval > 0.0 else 0.0
            elapsed = max(0.001, time.monotonic() - self.b1_started_mono) if self.b1_started_mono else 0.001
            per_camera = {
                cid: self.det_result_counts.get(cid, 0) / elapsed for cid in sorted(self.det_result_counts)
            }
            trt = list(self.trt_ms)
            roundtrip = list(self.roundtrip_ms)
            result_age = list(self.result_age_ms)
            capture_wait = list(self.capture_wait_ms)
            batches = self.det_batches
            timeouts = self.det_timeouts
            ready = int(self.det_ready)
            error = self._safe_error(self.det_error)
        rates = ",".join(f"{cid}:{hz:.2f}" for cid, hz in per_camera.items())
        print(
            "CAMERA_V11_STEP2V5_STATS "
            f"ready={ready} target_per_camera={self.b1_target_hz_per_camera:.2f}Hz "
            f"global_actual={global_hz:.2f}Hz rates={rates} "
            f"capture_wait_p95={self._pct(capture_wait, 0.95):.1f}ms "
            f"trt_p50={self._pct(trt, 0.50):.1f}ms trt_p95={self._pct(trt, 0.95):.1f}ms "
            f"roundtrip_p95={self._pct(roundtrip, 0.95):.1f}ms "
            f"result_age_p95={self._pct(result_age, 0.95):.1f}ms "
            f"inferences={batches} timeouts={timeouts} batch=1 prefetch=0 error={error}",
            flush=True,
        )
        return True


def main() -> int:
    return V11Step2DetectorB1V5().run()


if __name__ == "__main__":
    raise SystemExit(main())
