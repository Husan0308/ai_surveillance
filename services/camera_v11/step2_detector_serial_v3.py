from __future__ import annotations

import os
import sys
import time

import numpy as np

from .batch6_trt86 import Batch6TRT86Client
from .step2_detector_only import DETECT_H, DETECT_W
from .step2_detector_protected_v2 import V11Step2DetectorProtectedV2


class V11Step2DetectorSerialV3(V11Step2DetectorProtectedV2):
    """Step2 V3: serialize detector conversion to protect the frozen display.

    V2 removed TensorRT/capture prefetch but still requested all six detector
    conversions at once. On GTX 1050 Ti that created a periodic six-way GPU resize
    burst every detector cycle and regressed Step1 source/render cadence.

    V3 keeps the same main-stream detector input and true TRT8.6 batch=6, but requests
    exactly one camera at a time. The next camera is requested only after the previous
    672x378 BGRx sample reaches the mailbox. Thus at most one detector nvvideoconvert
    is active at once. No historical frame queue is introduced.
    """

    def __init__(self) -> None:
        # A 5 Hz cap gives the display >80% of wall-time headroom on a 20 FPS source.
        # NvDCF in the next stage will track on uninferenced frames.
        self.det_serial_target_hz = max(
            2.0,
            min(8.0, float(os.environ.get("V11_DETECT_SERIAL_TARGET_HZ", "5.0"))),
        )
        self.det_serial_period = 1.0 / self.det_serial_target_hz
        self.det_single_timeout_ms = max(
            80.0,
            min(220.0, float(os.environ.get("V11_DETECT_SINGLE_TIMEOUT_MS", "120"))),
        )
        super().__init__()
        self.det_target_hz = self.det_serial_target_hz
        self.det_target_period = self.det_serial_period
        self.det_prefetch_batches = 0
        print(
            "CAMERA_V11_STEP2V3_POLICY "
            f"target_hz={self.det_serial_target_hz:.2f} prefetch=0 batch=6 "
            f"single_timeout_ms={self.det_single_timeout_ms:.0f} "
            "capture=serial-jit latest_only=1 detector_conversions_inflight_max=1 "
            "display_priority=protected",
            flush=True,
        )

    def _detector_scheduler(self) -> None:
        client: Batch6TRT86Client | None = None
        try:
            client = Batch6TRT86Client()
            warm = np.full((DETECT_H, DETECT_W, 3), 114, dtype=np.uint8)
            ids = [camera.camera_id for camera in self.cameras]
            for _ in range(3):
                client.infer(ids, [warm] * 6, conf=self.det_conf, max_det=self.det_max_det)
            with self.det_lock:
                self.det_ready = True
            print(
                "CAMERA_V11_STEP2V3_DETECT_READY "
                f"engine={client.engine} worker={client.worker.name} batch=6 "
                f"tensorrt=8.6.1 warmup=3 target_hz={self.det_serial_target_hz:.2f} "
                "prefetch=0 capture=serial-jit",
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
            rotate = 0

            while not self.det_stop.is_set():
                cycle_started = time.monotonic()
                ordered = ids[rotate:] + ids[:rotate]
                rotate = (rotate + 1) % len(ids)

                frame_by_cid: dict[str, np.ndarray] = {}
                captured_by_cid: dict[str, float] = {}
                first_request = 0.0
                last_capture = 0.0
                complete = True

                for index, cid in enumerate(ordered):
                    requested = time.monotonic()
                    if index == 0:
                        first_request = requested
                    with self.det_lock:
                        self.capture_requested[cid] = True
                    row = self.mailbox.wait_new(
                        cid,
                        versions[cid],
                        self.det_single_timeout_ms / 1000.0,
                    )
                    if row is None:
                        complete = False
                        with self.det_lock:
                            self.capture_requested[cid] = False
                            self.det_timeouts += 1
                        break
                    version, captured_at, frame = row
                    versions[cid] = int(version)
                    frame_by_cid[cid] = frame
                    captured_by_cid[cid] = float(captured_at)
                    last_capture = time.monotonic()

                if not complete or len(frame_by_cid) != 6:
                    elapsed = time.monotonic() - cycle_started
                    self.det_stop.wait(max(0.005, self.det_serial_period - elapsed))
                    continue

                # Preserve stable batch slot -> camera mapping irrespective of the
                # rotating serial capture order.
                frames = [frame_by_cid[cid] for cid in ids]
                captured = [captured_by_cid[cid] for cid in ids]
                capture_wait = max(0.0, (last_capture - first_request) * 1000.0)
                skew = max(0.0, (max(captured) - min(captured)) * 1000.0)

                result = client.infer(
                    ids,
                    frames,
                    conf=self.det_conf,
                    max_det=self.det_max_det,
                )
                completed = time.monotonic()
                max_age = max((completed - ts) * 1000.0 for ts in captured)

                with self.det_lock:
                    if self.last_batch_completed > 0.0:
                        self.batch_interval_ms.append(
                            (completed - self.last_batch_completed) * 1000.0
                        )
                    self.last_batch_completed = completed
                    self.det_batches += 1
                    self.capture_wait_ms.append(capture_wait)
                    self.input_skew_ms.append(skew)
                    self.shm_copy_ms.append(result.shm_copy_ms)
                    self.prep_ms.append(result.prep_ms)
                    self.trt_ms.append(result.trt_ms)
                    self.roundtrip_ms.append(result.roundtrip_ms)
                    self.result_age_ms.append(max_age)
                    total_boxes = 0
                    for cid, ts in zip(ids, captured):
                        count = len(result.boxes.get(cid, []))
                        self.det_last_boxes[cid] = count
                        self.det_result_counts[cid] += 1
                        self.det_result_age_ms[cid].append(
                            max(0.0, (completed - ts) * 1000.0)
                        )
                        total_boxes += count
                    self.det_total_boxes += total_boxes
                    self.det_error = ""
                    batch_n = self.det_batches

                if batch_n <= 5 or batch_n % 20 == 0:
                    print(
                        "CAMERA_V11_STEP2V3_BATCH "
                        f"n={batch_n} capture_wait={capture_wait:.1f}ms skew={skew:.1f}ms "
                        f"shm={result.shm_copy_ms:.1f}ms prep={result.prep_ms:.1f}ms "
                        f"trt={result.trt_ms:.1f}ms roundtrip={result.roundtrip_ms:.1f}ms "
                        f"result_age={max_age:.1f}ms boxes={total_boxes}",
                        flush=True,
                    )

                elapsed = time.monotonic() - cycle_started
                self.det_stop.wait(max(0.002, self.det_serial_period - elapsed))
        except BaseException as exc:
            with self.det_lock:
                self.det_error = f"{type(exc).__name__}:{exc}"
            print(
                f"CAMERA_V11_STEP2V3_DETECT_FATAL error={type(exc).__name__}:{exc}",
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
            actual_hz = 0.0
            if intervals:
                avg = sum(intervals) / len(intervals)
                actual_hz = 1000.0 / avg if avg > 0.0 else 0.0
            trt = list(self.trt_ms)
            roundtrip = list(self.roundtrip_ms)
            result_age = list(self.result_age_ms)
            capture_wait = list(self.capture_wait_ms)
            skew = list(self.input_skew_ms)
            batches = self.det_batches
            timeouts = self.det_timeouts
            ready = int(self.det_ready)
            error = self._safe_error(self.det_error)
        print(
            "CAMERA_V11_STEP2V3_STATS "
            f"ready={ready} target={self.det_serial_target_hz:.2f}Hz actual={actual_hz:.2f}Hz "
            f"capture_wait_p95={self._pct(capture_wait, 0.95):.1f}ms "
            f"skew_p95={self._pct(skew, 0.95):.1f}ms "
            f"trt_p95={self._pct(trt, 0.95):.1f}ms "
            f"roundtrip_p95={self._pct(roundtrip, 0.95):.1f}ms "
            f"result_age_p95={self._pct(result_age, 0.95):.1f}ms "
            f"batches={batches} timeouts={timeouts} prefetch=0 serial=1 error={error}",
            flush=True,
        )
        return True


def main() -> int:
    return V11Step2DetectorSerialV3().run()


if __name__ == "__main__":
    raise SystemExit(main())
