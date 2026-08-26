#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from scripts.yolo26_trt86_shm_worker_v3 import Runner  # noqa: E402


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--dir", default=str(ROOT / ".runtime" / "yolo26_parity"))
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.08)
    args = ap.parse_args()

    engine = Path(args.engine)
    runner = Runner(engine)
    directory = Path(args.dir)
    samples = sorted(directory.glob("CAM-05_sample*.npy"))
    if samples:
        frame = np.load(samples[-1], allow_pickle=False)
    else:
        frame = np.full((384, 672, 3), 114, dtype=np.uint8)

    if frame.shape != (384, 672, 3) or frame.dtype != np.uint8:
        raise SystemExit(f"YOLO26M_RESCUE_BENCH_FAIL frame={frame.shape}/{frame.dtype}")

    try:
        for _ in range(max(0, args.warmup)):
            runner.infer(frame.copy(), args.conf, 40)
        values: list[float] = []
        boxes = []
        max_person = None
        for _ in range(max(1, args.runs)):
            rows, _prep, trt_ms, _total, health = runner.infer(frame.copy(), args.conf, 40)
            values.append(float(trt_ms))
            boxes = rows
            max_person = health.get("raw_person_max_conf")
    finally:
        runner.close()

    print(
        "YOLO26M_RESCUE_BENCH "
        f"engine={engine.name} sample={samples[-1].name if samples else 'synthetic'} "
        f"runs={len(values)} trt_p50={statistics.median(values):.1f}ms "
        f"trt_p95={p95(values):.1f}ms boxes={len(boxes)} person_max={max_person}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
