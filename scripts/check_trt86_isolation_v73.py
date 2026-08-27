#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W, H = 672, 384


def read_json(proc, timeout: float = 10.0):
    import select

    if proc.stdout is None:
        raise RuntimeError("worker stdout missing")
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        raise TimeoutError("worker response timeout")
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError(f"worker exited rc={proc.poll()}")
    return json.loads(line)


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    i = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[i])


def main() -> int:
    ap = argparse.ArgumentParser(description="Isolated TRT8.6 latency probe; no cameras/NvDCF")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    py = Path(os.environ.get("CAMERA_V2_TRT86_PYTHON", ROOT / ".venv-trt86/bin/python"))
    worker = Path(os.environ.get("CAMERA_V2_TRT86_SHM_WORKER", ROOT / "scripts/yolo26_trt86_shm_worker_v4.py"))
    engine = Path(os.environ.get("CAMERA_V2_TRT86_ENGINE", ROOT / "artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine"))
    for path, label in ((py, "python"), (worker, "worker"), (engine, "engine")):
        if not path.exists():
            raise SystemExit(f"TRT86_ISO FAIL missing_{label}={path}")

    shm = shared_memory.SharedMemory(create=True, size=H * W * 3)
    frame = np.ndarray((H, W, 3), dtype=np.uint8, buffer=shm.buf)
    # A deterministic non-uniform image prevents pathological all-zero preprocessing.
    yy = np.arange(H, dtype=np.uint16)[:, None]
    xx = np.arange(W, dtype=np.uint16)[None, :]
    frame[:, :, 0] = ((xx + yy) % 256).astype(np.uint8)
    frame[:, :, 1] = ((2 * xx + yy) % 256).astype(np.uint8)
    frame[:, :, 2] = ((xx + 2 * yy) % 256).astype(np.uint8)

    proc = subprocess.Popen(
        [str(py), str(worker), "--engine", str(engine)],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )
    try:
        ready = read_json(proc, 30.0)
        if ready.get("type") != "ready":
            raise RuntimeError(f"bad ready={ready}")
        print(
            "TRT86_ISO_READY "
            f"trt={ready.get('tensorrt')} transport={ready.get('transport')} engine={engine.name}",
            flush=True,
        )

        trt_ms: list[float] = []
        total_ms: list[float] = []
        n_total = max(1, args.warmup) + max(1, args.runs)
        for i in range(n_total):
            req = {"id": i + 1, "shm_name": shm.name, "conf": 0.18, "max_det": 20}
            if proc.stdin is None:
                raise RuntimeError("worker stdin missing")
            started = time.perf_counter()
            proc.stdin.write(json.dumps(req, separators=(",", ":")) + "\n")
            proc.stdin.flush()
            res = read_json(proc, 10.0)
            wall = (time.perf_counter() - started) * 1000.0
            if not res.get("ok"):
                raise RuntimeError(str(res))
            if i >= args.warmup:
                trt_ms.append(float(res.get("trt_ms", 0.0)))
                total_ms.append(wall)

        med = statistics.median(trt_ms)
        p95 = pct(trt_ms, 0.95)
        wall_med = statistics.median(total_ms)
        status = "PASS" if med <= 35.0 and p95 <= 60.0 else "FAIL"
        print(
            f"TRT86_ISO {status} runs={len(trt_ms)} trt_med={med:.1f}ms "
            f"trt_p95={p95:.1f}ms wall_med={wall_med:.1f}ms",
            flush=True,
        )
        if status == "FAIL":
            print(
                "TRT86_ISO reason=isolated TensorRT is already slow; do not tune NvDCF/bbox yet",
                flush=True,
            )
            return 1
        return 0
    finally:
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write('{"cmd":"stop"}\n')
                proc.stdin.flush()
                proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        shm.close()
        try:
            shm.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
