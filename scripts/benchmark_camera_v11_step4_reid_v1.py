#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

# This benchmark is intentionally executable as `python scripts/...py` from the
# repository wrapper. In that launch mode Python prepends scripts/, not the repo
# root, to sys.path. Bootstrap the repository root before importing `services` so
# the benchmark is location-stable and does not depend on ambient PYTHONPATH.
ROOT = Path(__file__).resolve().parents[1]
root_text = str(ROOT)
if root_text not in sys.path:
    sys.path.insert(0, root_text)

import numpy as np

from services.camera_v11.step4_reid_trt86 import V11ReIDTRT86Client


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="artifacts/reid/resnet50_market1501_aicity156_b1-8_fp32_trt86.engine")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=40)
    args = parser.parse_args()
    engine = Path(args.engine)
    if not engine.is_absolute():
        engine = ROOT / engine
    engine = engine.resolve()
    if not engine.is_file():
        print(f"V11_STEP4_REID_BENCH RESULT=FAIL reason=engine_missing path={engine}")
        return 2
    rng = np.random.default_rng(11)
    samples = rng.normal(0.0, 1.0, size=(8, 3, 256, 128)).astype(np.float32)
    with V11ReIDTRT86Client(engine=engine) as client:
        print("V11_STEP4_REID_READY " + " ".join(f"{k}={v}" for k, v in sorted(client.info.items()) if not isinstance(v, list)), flush=True)
        for batch_size in (1, 2, 4, 8):
            totals: list[float] = []
            infer: list[float] = []
            norms: list[float] = []
            for index in range(max(1, args.warmup) + max(1, args.iterations)):
                started = time.perf_counter()
                output, stages = client.embed_preprocessed(samples[:batch_size])
                wall = (time.perf_counter() - started) * 1000.0
                if index < max(1, args.warmup):
                    continue
                totals.append(wall)
                infer.append(float(stages.get("inference_ms", 0.0)))
                norms.extend(np.linalg.norm(output, axis=1).tolist())
            print(
                "V11_STEP4_REID_BATCH "
                f"batch={batch_size} iterations={len(totals)} "
                f"wall_p50={pct(totals, .50):.3f}ms wall_p95={pct(totals, .95):.3f}ms "
                f"infer_p50={pct(infer, .50):.3f}ms infer_p95={pct(infer, .95):.3f}ms "
                f"norm_mean={statistics.fmean(norms):.6f} norm_min={min(norms):.6f} norm_max={max(norms):.6f}",
                flush=True,
            )
        first, _ = client.embed_preprocessed(samples[:1])
        second, _ = client.embed_preprocessed(samples[:1])
        repeat_cos = float(first[0] @ second[0])
        if repeat_cos < 0.999 or abs(float(np.linalg.norm(first[0])) - 1.0) > 1e-3:
            print(f"V11_STEP4_REID_BENCH RESULT=FAIL reason=embedding_sanity repeat_cos={repeat_cos:.6f}")
            return 1
        print(f"V11_STEP4_REID_BENCH RESULT=PASS repeat_cos={repeat_cos:.6f} transport=shm-f32 jpeg=0 base64=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
