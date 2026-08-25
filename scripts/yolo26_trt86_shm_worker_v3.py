#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.yolo26_trt86_shm_worker_v2 import Runner as DiagnosticRunner  # noqa: E402
from scripts.yolo26_trt86_shm_worker import emit, trt  # noqa: E402


class Runner(DiagnosticRunner):
    """TRT86 runner with exact fixed-shape YOLO letterbox padding semantics."""

    def infer(self, frame: np.ndarray, conf: float, max_det: int):
        # nvvideoconvert dest-crop preserves geometry in 672x378 centered inside
        # the 672x384 output. Ultralytics LetterBox uses padding_value=114, so set
        # the 3-pixel bars explicitly before BGR->RGB/CHW normalization. The frame
        # is the private SHM staging surface and is overwritten by the next capture.
        if frame.shape != (384, 672, 3):
            raise RuntimeError(f"unexpected letterbox frame shape={frame.shape}")
        frame[:3, :, :] = 114
        frame[381:, :, :] = 114
        return super().infer(frame, conf, max_det)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    args = ap.parse_args()
    runner = Runner(Path(args.engine))

    emit(
        {
            "type": "ready",
            "tensorrt": trt.__version__,
            "engine": str(Path(args.engine).resolve()),
            "input_shape": runner.input_shape,
            "output_shape": runner.output_shape,
            "transport": "shm-bgr-v3-letterbox114",
        }
    )

    n = 0
    try:
        for line in sys.stdin:
            req = {}
            try:
                req = json.loads(line)
                if req.get("cmd") == "stop":
                    break
                request_id = req.get("id")
                conf = float(req.get("conf", 0.05))
                max_det = int(req.get("max_det", 40))
                boxes, prep_ms, trt_ms, total_ms, health = runner.infer_from_shm(
                    str(req["shm_name"]), conf, max_det
                )
                n += 1
                if n <= 3 or n % 20 == 0:
                    print(
                        "CAM01_TRT86_CLASSES "
                        f"n={n} histogram={health.get('class_hist_above_conf')} "
                        f"top={health.get('top_classes')} "
                        f"person_max={health.get('raw_person_max_conf')} "
                        f"channel_spread={health.get('channel_spread_mean')} "
                        "letterbox=672x378+3+3 pad114",
                        file=sys.stderr,
                        flush=True,
                    )
                emit(
                    {
                        "id": request_id,
                        "ok": True,
                        "boxes": boxes,
                        "prep_ms": prep_ms,
                        "trt_ms": trt_ms,
                        "total_ms": total_ms,
                        "health": health,
                    }
                )
            except Exception as exc:
                emit(
                    {
                        "id": req.get("id"),
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        runner.close()


if __name__ == "__main__":
    main()
