#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import select
import statistics
import subprocess
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W, H = 672, 384


def _read_json(proc: subprocess.Popen[str], timeout: float = 10.0):
    if proc.stdout is None:
        raise RuntimeError("worker stdout unavailable")
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        raise TimeoutError("worker response timeout")
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError(f"worker exited rc={proc.poll()}")
    return json.loads(line)


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return float(ordered[idx])


def _find_trtexec() -> str | None:
    from shutil import which

    direct = which("trtexec")
    if direct:
        return direct
    candidates = [
        Path("/usr/src/tensorrt/bin/trtexec"),
        Path("/opt/tensorrt/bin/trtexec"),
        Path("/usr/local/TensorRT-8.6.1.6/bin/trtexec"),
        Path("/usr/local/TensorRT-8.6.1/bin/trtexec"),
        ROOT / ".venv-trt86/bin/trtexec",
    ]
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def _budget_profile(median_ms: float, cameras: int) -> tuple[float, float]:
    # Reserve ~70% of one GPU-second for NvDCF, decode/convert/OSD, and variance.
    # The detector may consume at most ~30% in steady state.
    sec = max(0.001, median_ms / 1000.0)
    detect_hz = 0.30 / (max(1, cameras) * sec)
    detect_hz = max(0.60, min(1.25, detect_hz))

    # The previous 20 Hz target was unrealistic on GP107. Keep enough temporal
    # localization to make the box sticky while leaving deterministic GPU headroom.
    if median_ms <= 35.0:
        track_hz = 12.0
    elif median_ms <= 75.0:
        track_hz = 10.0
    else:
        track_hz = 8.0
    return detect_hz, track_hz


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure isolated TRT86 and write V7.4 Pascal GPU budget")
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--cameras", type=int, default=6)
    ap.add_argument("--output", default="/tmp/camera_v74_profile.env")
    args = ap.parse_args()

    py = Path(os.environ.get("CAMERA_V2_TRT86_PYTHON", ROOT / ".venv-trt86/bin/python"))
    worker = Path(os.environ.get("CAMERA_V2_TRT86_SHM_WORKER", ROOT / "scripts/yolo26_trt86_shm_worker_v4.py"))
    engine = Path(os.environ.get("CAMERA_V2_TRT86_ENGINE", ROOT / "artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine"))
    for path, label in ((py, "python"), (worker, "worker"), (engine, "engine")):
        if not path.exists():
            raise SystemExit(f"CAMERA_V74_PROFILE FAIL missing_{label}={path}")

    shm = shared_memory.SharedMemory(create=True, size=H * W * 3)
    frame = np.ndarray((H, W, 3), dtype=np.uint8, buffer=shm.buf)
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
        ready = _read_json(proc, 30.0)
        if ready.get("type") != "ready":
            raise RuntimeError(f"bad TRT ready={ready}")

        trt_ms: list[float] = []
        wall_ms: list[float] = []
        total = max(1, args.warmup) + max(1, args.runs)
        for i in range(total):
            req = {"id": i + 1, "shm_name": shm.name, "conf": 0.18, "max_det": 20}
            if proc.stdin is None:
                raise RuntimeError("worker stdin unavailable")
            started = time.perf_counter()
            proc.stdin.write(json.dumps(req, separators=(",", ":")) + "\n")
            proc.stdin.flush()
            res = _read_json(proc, 10.0)
            wall = (time.perf_counter() - started) * 1000.0
            if not res.get("ok"):
                raise RuntimeError(str(res))
            if i >= args.warmup:
                trt_ms.append(float(res.get("trt_ms", 0.0)))
                wall_ms.append(wall)

        med = statistics.median(trt_ms)
        p95 = _percentile(trt_ms, 0.95)
        wall = statistics.median(wall_ms)
        detect_hz, track_hz = _budget_profile(med, args.cameras)

        # 60 ms on this GP107 is treated as a measured baseline, not as an automatic
        # failure. We only refuse clearly broken states that are >2x that observed range.
        if med > 130.0 or p95 > 180.0:
            print(
                f"CAMERA_V74_PROFILE FAIL trt_med={med:.1f}ms trt_p95={p95:.1f}ms "
                "reason=isolated TRT grossly degraded",
                file=sys.stderr,
                flush=True,
            )
            return 2

        out = Path(args.output)
        out.write_text(
            "\n".join(
                [
                    f"export CAMERA_V2_TRT_BASELINE_MS={med:.3f}",
                    f"export CAMERA_V2_DETECT_HZ={detect_hz:.3f}",
                    f"export CAMERA_V2_TRACK_FPS={track_hz:.1f}",
                    "export CAMERA_V2_NVDCF_FEATURE_LEVEL=2",
                    "export CAMERA_V2_NVDCF_USE_HOG=0",
                    "export CAMERA_V2_NVDCF_USE_COLORNAMES=1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        trtexec = _find_trtexec() or "not-found"
        detector_fraction = args.cameras * detect_hz * (med / 1000.0)
        print(
            "CAMERA_V74_PROFILE PASS "
            f"trt_med={med:.1f}ms trt_p95={p95:.1f}ms wall_med={wall:.1f}ms "
            f"detector_hz={detect_hz:.2f}/cam tracker_target={track_hz:.1f}Hz "
            f"detector_gpu_budget={detector_fraction*100.0:.1f}% trtexec={trtexec} "
            f"env={out}",
            flush=True,
        )
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
