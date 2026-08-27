#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from services.camera_v11.batch6_trt86 import Batch6TRT86Client


def pct(values: list[float], q: float) -> float:
    rows = sorted(values)
    if not rows:
        return 0.0
    idx = min(len(rows) - 1, int(round((len(rows) - 1) * q)))
    return float(rows[idx])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=400)
    args = parser.parse_args()

    warmup = max(3, args.warmup)
    iterations = max(100, args.iterations)
    ids = [f"CAM-{i:02d}" for i in range(1, 7)]
    frame = np.full((384, 672, 3), 114, dtype=np.uint8)
    frames = [frame] * 6

    trt: list[float] = []
    roundtrip: list[float] = []
    prep: list[float] = []
    shm: list[float] = []

    client = Batch6TRT86Client()
    try:
        print(
            f"V11_TRT86_STANDALONE_READY engine={client.engine} batch=6 "
            f"warmup={warmup} iterations={iterations}",
            flush=True,
        )
        for _ in range(warmup):
            client.infer(ids, frames, conf=0.18, max_det=20)

        for i in range(1, iterations + 1):
            row = client.infer(ids, frames, conf=0.18, max_det=20)
            trt.append(row.trt_ms)
            roundtrip.append(row.roundtrip_ms)
            prep.append(row.prep_ms)
            shm.append(row.shm_copy_ms)
            if i <= 5 or i % 50 == 0 or row.trt_ms >= 200.0:
                print(
                    "V11_TRT86_STANDALONE_SAMPLE "
                    f"n={i} shm={row.shm_copy_ms:.1f}ms prep={row.prep_ms:.1f}ms "
                    f"trt={row.trt_ms:.1f}ms roundtrip={row.roundtrip_ms:.1f}ms",
                    flush=True,
                )
    finally:
        client.close()

    stalls_200 = sum(v >= 200.0 for v in trt)
    stalls_500 = sum(v >= 500.0 for v in trt)
    stalls_900 = sum(v >= 900.0 for v in trt)
    mean = statistics.fmean(trt)
    print(
        "V11_TRT86_STANDALONE_RESULT "
        f"trt_mean={mean:.1f}ms trt_p50={pct(trt, 0.50):.1f}ms "
        f"trt_p95={pct(trt, 0.95):.1f}ms trt_p99={pct(trt, 0.99):.1f}ms "
        f"trt_max={max(trt):.1f}ms roundtrip_p95={pct(roundtrip, 0.95):.1f}ms "
        f"prep_p95={pct(prep, 0.95):.1f}ms shm_p95={pct(shm, 0.95):.1f}ms "
        f"stalls_ge200={stalls_200} stalls_ge500={stalls_500} stalls_ge900={stalls_900} "
        f"iterations={len(trt)}",
        flush=True,
    )

    if pct(trt, 0.95) <= 50.0 and stalls_200 == 0:
        print("V11_TRT86_STANDALONE diagnosis=ENGINE_FAST camera_contention_suspected", flush=True)
    elif pct(trt, 0.50) > 50.0:
        print("V11_TRT86_STANDALONE diagnosis=ENGINE_TOO_SLOW_FOR_20FPS_BATCH6", flush=True)
    else:
        print("V11_TRT86_STANDALONE diagnosis=ENGINE_HAS_LONG_STALLS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
