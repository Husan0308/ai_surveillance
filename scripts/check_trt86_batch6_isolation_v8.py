#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.yolo26_trt86_batch6_worker_v8 import BATCH, INPUT_H, INPUT_W, Batch6Runner


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    i = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[i]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        default=str(ROOT / "artifacts/yolo26s_trt86/yolo26s-672x384-b6-fp32-trt86.engine"),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    engine = Path(args.engine).resolve()
    runner = Batch6Runner(engine)
    shm = shared_memory.SharedMemory(create=True, size=BATCH * INPUT_H * INPUT_W * 3)
    batch = np.ndarray((BATCH, INPUT_H, INPUT_W, 3), dtype=np.uint8, buffer=shm.buf)
    # Deterministic non-zero input prevents a degenerate all-zero memory path.
    for i in range(BATCH):
        batch[i].fill(32 + i * 17)
    try:
        for _ in range(max(1, args.warmup)):
            runner.infer_from_shm(shm.name, 0.18, 20)
        gpu: list[float] = []
        total: list[float] = []
        prep: list[float] = []
        for _ in range(max(1, args.runs)):
            _boxes, prep_ms, gpu_ms, total_ms = runner.infer_from_shm(shm.name, 0.18, 20)
            prep.append(prep_ms)
            gpu.append(gpu_ms)
            total.append(total_ms)
        print(
            "V8_TRT_BATCH6_ISO "
            f"runs={len(gpu)} gpu_med={statistics.median(gpu):.1f}ms gpu_p95={pct(gpu,0.95):.1f}ms "
            f"prep_med={statistics.median(prep):.1f}ms total_med={statistics.median(total):.1f}ms "
            f"effective_per_camera_gpu={statistics.median(gpu)/BATCH:.1f}ms",
            flush=True,
        )
        return 0
    finally:
        runner.close()
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
