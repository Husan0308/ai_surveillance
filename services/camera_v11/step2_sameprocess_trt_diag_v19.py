from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np

from services.camera_v11.step1_cam02_lowlat_v7 import V11Step1Cam02LowLatV7
from scripts.yolo26_trt86_step2_worker import Runner


class V11Step2SameProcessTRTDiagV19(V11Step1Cam02LowLatV7):
    """Diagnostic only: frozen Step1 plus synthetic FP32 TRT in one process."""

    def __init__(self) -> None:
        super().__init__()
        self.engine_path = Path(
            os.environ.get(
                "V11_STEP2_ENGINE",
                "artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine",
            )
        ).resolve()
        self.synthetic_hz = max(0.1, min(30.0, float(os.environ.get("V11_DIAG_TRT_HZ", "12.0"))))
        self.delay_sec = max(1.0, float(os.environ.get("V11_DIAG_TRT_DELAY_SEC", "5.0")))
        self.trt_stop = threading.Event()
        self.trt_thread: threading.Thread | None = None
        self.trt_started = False
        self.samples: list[float] = []
        self.count = 0
        print(
            "CAMERA_V11_STEP2_CTXDIAG_ARCH mode=same-process-synthetic-trt "
            "display=frozen-step1-v7 detector_rtsp=0 substream=0 tracker=0",
            flush=True,
        )
        print(
            f"CAMERA_V11_STEP2_CTXDIAG_POLICY hz={self.synthetic_hz:.2f} "
            f"delay_sec={self.delay_sec:.1f} engine={self.engine_path}",
            flush=True,
        )

    @staticmethod
    def _p(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        rows = sorted(values)
        return float(rows[min(len(rows) - 1, int(round((len(rows) - 1) * q)))])

    def _trt_loop(self) -> None:
        runner: Runner | None = None
        frame = np.full((384, 672, 3), 114, dtype=np.uint8)
        try:
            runner = Runner(self.engine_path)
            print(
                "CAMERA_V11_STEP2_CTXDIAG_READY "
                f"priority_least={runner.priority_least} priority_greatest={runner.priority_greatest}",
                flush=True,
            )
            for _ in range(10):
                runner.infer(frame, 0.18, 20)
            print("CAMERA_V11_STEP2_CTXDIAG_WARMUP iterations=10 status=OK", flush=True)
            period = 1.0 / self.synthetic_hz
            next_slot = time.monotonic()
            report_at = next_slot + 5.0
            while not self.trt_stop.is_set() and not self._stopping:
                now = time.monotonic()
                if now < next_slot:
                    self.trt_stop.wait(min(0.005, next_slot - now))
                    continue
                started = time.perf_counter()
                _boxes, stages = runner.infer(frame, 0.18, 20)
                total_ms = (time.perf_counter() - started) * 1000.0
                self.samples.append(total_ms)
                if len(self.samples) > 2048:
                    del self.samples[: len(self.samples) - 2048]
                self.count += 1
                next_slot = time.monotonic() + period
                if time.monotonic() >= report_at:
                    print(
                        "CAMERA_V11_STEP2_CTXDIAG_TRT "
                        f"count={self.count} p50={self._p(self.samples, 0.50):.2f}ms "
                        f"p95={self._p(self.samples, 0.95):.2f}ms "
                        f"infer={float(stages.get('inference_ms', 0.0)):.2f}ms",
                        flush=True,
                    )
                    report_at = time.monotonic() + 5.0
        except Exception as exc:
            print(f"CAMERA_V11_STEP2_CTXDIAG_ERROR {type(exc).__name__}: {exc}", flush=True)
            try:
                self.GLib.idle_add(self.stop)
            except Exception:
                pass
        finally:
            if runner is not None:
                runner.close()
            print(
                f"CAMERA_V11_STEP2_CTXDIAG_STOP count={self.count} "
                f"p50={self._p(self.samples, 0.50):.2f}ms p95={self._p(self.samples, 0.95):.2f}ms",
                flush=True,
            )

    def _start_trt(self) -> bool:
        if self._stopping or self.trt_started:
            return False
        self.trt_started = True
        self.trt_thread = threading.Thread(target=self._trt_loop, daemon=True)
        self.trt_thread.start()
        print("CAMERA_V11_STEP2_CTXDIAG_PHASE phase=trt-on", flush=True)
        return False

    def run(self) -> int:
        self.GLib.timeout_add(int(round(self.delay_sec * 1000.0)), self._start_trt)
        try:
            return super().run()
        finally:
            self.trt_stop.set()
            if self.trt_thread is not None:
                self.trt_thread.join(timeout=5.0)
                self.trt_thread = None


def main() -> int:
    return V11Step2SameProcessTRTDiagV19().run()


if __name__ == "__main__":
    raise SystemExit(main())
