#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import statistics
import subprocess
import sys
import threading
import time
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
    return float(ordered[i])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_gpu_processes() -> list[str]:
    # Fail closed if a Camera V2/DeepStream/TRT worker from this repository is still
    # alive. A clean-room benchmark is meaningless if the production pipeline is up.
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid=,args="], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return []
    needles = (
        "services.camera_v2",
        "run_camera_v2_",
        "camera-v8-trt86",
        "yolo26_trt86_batch6_worker_v8.py",
        "yolo26_trt86_batch6_worker_v83.py",
    )
    me = os.getpid()
    rows: list[str] = []
    for line in out.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            pid = int(stripped.split(None, 1)[0])
        except Exception:
            continue
        if pid == me:
            continue
        if any(token in stripped for token in needles):
            rows.append(stripped)
    return rows


def gpu_sample() -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=pstate,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        default=str(ROOT / "artifacts/yolo26s_trt86/yolo26s-672x384-b6-fp32-trt86.engine"),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=40)
    args = parser.parse_args()

    leftovers = repo_gpu_processes()
    if leftovers:
        print("V83_CLEANROOM FAIL reason=repo_gpu_processes_alive", flush=True)
        for row in leftovers[:20]:
            print(f"V83_CLEANROOM_PROCESS {row}", flush=True)
        print("V83_CLEANROOM next=stop Camera V2 runtime and retry", flush=True)
        return 2

    engine = Path(args.engine).resolve()
    if not engine.is_file():
        raise SystemExit(f"V83_CLEANROOM FAIL engine_missing={engine}")

    print(
        "V83_CLEANROOM_READY "
        f"engine={engine.name} bytes={engine.stat().st_size} sha256={sha256(engine)[:16]} "
        f"gpu_before=[{gpu_sample()}]",
        flush=True,
    )

    runner = Batch6Runner(engine)
    shm = shared_memory.SharedMemory(create=True, size=BATCH * INPUT_H * INPUT_W * 3)
    batch = np.ndarray((BATCH, INPUT_H, INPUT_W, 3), dtype=np.uint8, buffer=shm.buf)
    rng = np.random.default_rng(83)
    batch[:] = rng.integers(0, 256, size=batch.shape, dtype=np.uint8)

    samples: list[str] = []
    stop_monitor = threading.Event()

    def monitor() -> None:
        while not stop_monitor.wait(0.25):
            samples.append(gpu_sample())

    monitor_thread = threading.Thread(target=monitor, name="v83-gpu-monitor", daemon=True)
    monitor_thread.start()
    try:
        for _ in range(max(1, args.warmup)):
            runner.infer_from_shm(shm.name, 0.18, 20)

        gpu: list[float] = []
        total: list[float] = []
        prep: list[float] = []
        for _ in range(max(1, args.runs)):
            _boxes, prep_ms, gpu_ms, total_ms = runner.infer_from_shm(
                shm.name, 0.18, 20
            )
            prep.append(float(prep_ms))
            gpu.append(float(gpu_ms))
            total.append(float(total_ms))
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=1.0)
        runner.close()
        shm.close()
        shm.unlink()

    gpu_med = statistics.median(gpu)
    gpu_p95 = pct(gpu, 0.95)
    total_med = statistics.median(total)
    prep_med = statistics.median(prep)
    unique_samples = []
    for row in samples:
        if row not in unique_samples:
            unique_samples.append(row)

    print(
        "V83_CLEANROOM_RESULT "
        f"runs={len(gpu)} gpu_med={gpu_med:.1f}ms gpu_p95={gpu_p95:.1f}ms "
        f"prep_med={prep_med:.1f}ms total_med={total_med:.1f}ms "
        f"per_camera_equiv={gpu_med/BATCH:.1f}ms gpu_after=[{gpu_sample()}]",
        flush=True,
    )
    for row in unique_samples[:12]:
        print(f"V83_CLEANROOM_GPU sample=[{row}]", flush=True)

    if gpu_med <= 220.0:
        print(
            "V83_CLEANROOM PASS diagnosis=engine-fast-without-deepstream "
            "next=step2-context-contention-test",
            flush=True,
        )
        return 0

    print(
        "V83_CLEANROOM FAIL diagnosis=engine-or-gpu-slow-even-without-deepstream "
        "next=inspect-engine-clocks-tactics-before-tracker",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
