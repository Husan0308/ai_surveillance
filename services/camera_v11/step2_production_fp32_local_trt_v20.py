from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import numpy as np

from scripts.yolo26_trt86_step2_worker import Runner
from . import step2_production_fp32_v12
from .step2_production_fp32_v18 import V11Step2ProductionFP32V18
from .step2_trt86 import CONTENT_H, INPUT_H, INPUT_W, ROOT, TRTResult


class LocalTRT86Client:
    """The production Runner in the detector process; display remains separate."""

    def __init__(self) -> None:
        raw = os.environ.get(
            "V11_STEP2_ENGINE",
            str(ROOT / "artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine"),
        )
        engine = Path(raw).expanduser()
        if not engine.is_absolute():
            engine = ROOT / engine
        self.engine = engine.resolve()
        self.runner = Runner(self.engine)
        self.frame = np.full((INPUT_H, INPUT_W, 3), 114, dtype=np.uint8)
        self.content = self.frame[3:381]
        print(
            "CAMERA_V11_STEP2_TRT_READY "
            f"engine={self.engine} precision=fp32 batch=1 isolated=0 "
            "transport=in-process-preallocated-pinned-async stream=nonblocking-low-priority "
            f"priority={self.runner.priority_least}/range="
            f"{self.runner.priority_greatest}..{self.runner.priority_least}",
            flush=True,
        )

    def infer_preloaded(self, conf: float, max_det: int) -> TRTResult:
        started = time.perf_counter()
        boxes, stages = self.runner.infer(self.frame, conf, max_det)
        return TRTResult(
            boxes=boxes,
            stages={str(key): float(value) for key, value in stages.items()},
            roundtrip_ms=(time.perf_counter() - started) * 1000.0,
        )

    def close(self) -> None:
        runner = getattr(self, "runner", None)
        if runner is not None:
            runner.close()
            self.runner = None
        self.content = None
        self.frame = None


def main() -> int:
    step2_production_fp32_v12.Step2TRT86Client = LocalTRT86Client
    service = V11Step2ProductionFP32V18()

    def stop(_signum, _frame) -> None:
        service.stop_requested = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        return service.run()
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
