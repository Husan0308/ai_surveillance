#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAME_SHAPE = (384, 672, 3)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def summarize(name: str, values: list[float]) -> str:
    return (
        f"{name}_mean={statistics.fmean(values):.3f}ms "
        f"{name}_p50={percentile(values, 0.50):.3f}ms "
        f"{name}_p95={percentile(values, 0.95):.3f}ms"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the exact production TRT8.6 async worker")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    engine = Path(args.engine).expanduser()
    if not engine.is_absolute():
        engine = ROOT / engine
    engine = engine.resolve()
    if not engine.is_file():
        raise SystemExit(f"V11_TRT86_ASYNC_WORKER FAIL missing={engine}")

    frame_bytes = int(np.prod(FRAME_SHAPE))
    shm = shared_memory.SharedMemory(create=True, size=frame_bytes)
    frame = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)
    frame.fill(115)

    worker = subprocess.Popen(
        [
            sys.executable,
            "-I",
            str(ROOT / "scripts/yolo26_trt86_step2_worker.py"),
            "--engine",
            str(engine),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert worker.stdin is not None
    assert worker.stdout is not None

    try:
        line = worker.stdout.readline()
        if not line:
            stderr = worker.stderr.read() if worker.stderr is not None else ""
            raise RuntimeError(f"worker exited before ready: {stderr}")
        ready = json.loads(line)
        if ready.get("type") != "ready":
            raise RuntimeError(f"worker did not become ready: {ready}")

        metrics = {
            name: []
            for name in (
                "preprocess",
                "enqueue",
                "sync_wait",
                "h2d",
                "inference",
                "d2h",
                "postprocess",
                "worker_total",
                "host_roundtrip",
            )
        }
        warmup = max(1, int(args.warmup))
        iterations = max(1, int(args.iterations))
        total = warmup + iterations
        for request_id in range(total):
            request = {
                "id": request_id,
                "shm_name": shm.name,
                "conf": 0.18,
                "max_det": 20,
            }
            started = time.perf_counter()
            worker.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            worker.stdin.flush()
            response_line = worker.stdout.readline()
            if not response_line:
                raise RuntimeError("worker stdout closed during benchmark")
            response = json.loads(response_line)
            host_roundtrip_ms = (time.perf_counter() - started) * 1000.0
            if not response.get("ok"):
                raise RuntimeError(f"worker request failed: {response}")
            if request_id < warmup:
                continue
            stages = response["stages"]
            for source, target in (
                ("preprocess_ms", "preprocess"),
                ("enqueue_ms", "enqueue"),
                ("sync_wait_ms", "sync_wait"),
                ("h2d_ms", "h2d"),
                ("inference_ms", "inference"),
                ("d2h_ms", "d2h"),
                ("postprocess_ms", "postprocess"),
                ("total_ms", "worker_total"),
            ):
                metrics[target].append(float(stages.get(source, 0.0)))
            metrics["host_roundtrip"].append(host_roundtrip_ms)

        print(
            "V11_TRT86_ASYNC_WORKER_RESULT "
            f"engine={engine.name} iterations={iterations} "
            f"priority_least={ready.get('priority_least')} priority_greatest={ready.get('priority_greatest')} "
            + " ".join(summarize(name, values) for name, values in metrics.items()),
            flush=True,
        )
    finally:
        if worker.poll() is None:
            try:
                worker.stdin.write('{"cmd":"stop"}\n')
                worker.stdin.flush()
                worker.wait(timeout=10)
            except Exception:
                worker.terminate()
                worker.wait(timeout=10)
        stderr = worker.stderr.read() if worker.stderr is not None else ""
        if stderr:
            print(stderr, file=sys.stderr, end="")
        shm.close()
        shm.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
