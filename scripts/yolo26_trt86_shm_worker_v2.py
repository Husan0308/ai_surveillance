#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.yolo26_trt86_shm_worker import Runner as BaseRunner, emit, trt  # noqa: E402


class Runner(BaseRunner):
    """Base TensorRT runner plus diagnostics that do not alter predictions."""

    def _attach(self, name: str):
        if self._shm is not None and self._shm_name == name:
            return self._shm
        if self._shm is not None:
            self._shm.close()
        shm = shared_memory.SharedMemory(name=name, create=False)
        # This sidecar is not the owner of the SHM block. Python <=3.12 registers
        # create=False blocks with its resource_tracker anyway, which later causes
        # a false "leaked shared_memory" warning after the creator unlinks it.
        # Unregister only in this non-owner process; the bridge process remains the
        # real owner and performs close()+unlink().
        try:
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass
        self._shm = shm
        self._shm_name = name
        return self._shm

    def infer(self, frame: np.ndarray, conf: float, max_det: int):
        rows, prep_ms, trt_ms, total_ms, health = super().infer(
            frame, conf, max_det
        )

        pred = self.y[0]
        finite = np.isfinite(pred).all(axis=1)
        finite_pred = pred[finite]
        strong = finite_pred[finite_pred[:, 4] >= float(conf)] if finite_pred.size else finite_pred

        histogram: Counter[int] = Counter()
        top = []
        if strong.size:
            order = np.argsort(strong[:, 4])[::-1]
            for idx in order:
                cls = int(round(float(strong[idx, 5])))
                score = float(strong[idx, 4])
                histogram[cls] += 1
                if len(top) < 8:
                    top.append([cls, round(score, 4)])

        # Mean per-pixel channel spread is ~0 for a true grayscale/IR frame.
        # This is diagnostic only; RGB/BGR conversion remains unchanged.
        channel_spread = np.ptp(frame.astype(np.int16), axis=2)
        health["channel_spread_mean"] = round(float(channel_spread.mean()), 3)
        health["class_hist_above_conf"] = {
            str(k): int(v) for k, v in sorted(histogram.items())
        }
        health["top_classes"] = top
        return rows, prep_ms, trt_ms, total_ms, health


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
            "transport": "shm-bgr-v2",
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
                        f"channel_spread={health.get('channel_spread_mean')}",
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
