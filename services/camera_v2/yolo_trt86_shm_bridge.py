from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INFER_WIDTH = 672
INFER_HEIGHT = 384
FRAME_BYTES = INFER_WIDTH * INFER_HEIGHT * 3


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_json(proc, timeout: float):
    if proc.stdout is None:
        raise RuntimeError("TRT86 worker stdout unavailable")
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        raise TimeoutError(f"TRT86 detector timeout after {timeout:.1f}s")
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError(f"TRT86 detector closed rc={proc.poll()}")
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"TRT86 worker emitted non-JSON stdout: {line[:160]!r}") from exc


def yolo_trt86_shm_worker(job_q, result_q) -> None:
    """Own one TRT8.6 sidecar and one latest-frame SHM segment.

    This process deliberately ignores terminal SIGINT. The parent runtime owns
    Ctrl+C and sends the queue sentinel during orderly shutdown, so SHM ownership
    and sidecar cleanup cannot be interrupted halfway through.
    """
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    proc = None
    shm = None
    try:
        python_path = _resolve(
            os.environ.get("CAMERA_V2_TRT86_PYTHON", ".venv-trt86/bin/python")
        )
        worker_path = _resolve(
            os.environ.get(
                "CAMERA_V2_TRT86_SHM_WORKER",
                "scripts/yolo26_trt86_shm_worker_v4.py",
            )
        )
        engine_path = _resolve(
            os.environ.get(
                "CAMERA_V2_TRT86_ENGINE",
                "artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine",
            )
        )
        for path, label in (
            (python_path, "python"),
            (worker_path, "worker"),
            (engine_path, "engine"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"TRT86 {label} missing: {path}")

        shm = shared_memory.SharedMemory(create=True, size=FRAME_BYTES)
        shm_frame = np.ndarray(
            (INFER_HEIGHT, INFER_WIDTH, 3),
            dtype=np.uint8,
            buffer=shm.buf,
        )

        proc = subprocess.Popen(
            [str(python_path), str(worker_path), "--engine", str(engine_path)],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        ready = _read_json(proc, 30.0)
        if ready.get("type") != "ready":
            raise RuntimeError(f"bad TRT86 handshake: {ready}")
        if not str(ready.get("tensorrt", "")).startswith("8.6.1"):
            raise RuntimeError(f"wrong TensorRT runtime: {ready}")
        if tuple(ready.get("input_shape", ())) != (1, 3, 384, 672):
            raise RuntimeError(f"wrong TRT86 input shape: {ready}")
        if tuple(ready.get("output_shape", ())) != (1, 300, 6):
            raise RuntimeError(f"wrong YOLO26 output shape: {ready}")

        result_q.put(
            {
                "type": "ready",
                "device": "NVIDIA/TensorRT86",
                "cuda": "TRT8.6.1",
                "model": str(engine_path),
                "backend": "trt86-sidecar-shm-v4",
                "capture_policy": "jit-latest",
            }
        )

        request_id = 0
        log_n = 0
        while True:
            job = job_q.get()
            if job is None:
                return

            output: dict[str, list[tuple[list[float], float]]] = {}
            trt_sum_ms = 0.0
            total_sum_ms = 0.0
            for cid, frame in zip(job["cameras"], job["frames"]):
                if frame.shape != (INFER_HEIGHT, INFER_WIDTH, 3):
                    raise RuntimeError(
                        f"{cid}: expected BGR {INFER_WIDTH}x{INFER_HEIGHT}, "
                        f"got {frame.shape}"
                    )
                if frame.dtype != np.uint8:
                    raise RuntimeError(f"{cid}: expected uint8, got {frame.dtype}")

                copy_started = time.perf_counter()
                np.copyto(shm_frame, frame, casting="no")
                shm_copy_ms = (time.perf_counter() - copy_started) * 1000.0

                request_id += 1
                request = {
                    "id": request_id,
                    "shm_name": shm.name,
                    "conf": float(os.environ.get("CAMERA_V2_DETECT_CONF", "0.05")),
                    "max_det": max(
                        1,
                        min(300, int(os.environ.get("CAMERA_V2_MAX_DET", "40"))),
                    ),
                }
                if proc.stdin is None:
                    raise RuntimeError("TRT86 worker stdin unavailable")
                roundtrip_started = time.perf_counter()
                proc.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
                proc.stdin.flush()
                response = _read_json(proc, 5.0)
                roundtrip_ms = (time.perf_counter() - roundtrip_started) * 1000.0

                if response.get("id") != request_id:
                    raise RuntimeError("TRT86 response ID mismatch")
                if not response.get("ok"):
                    raise RuntimeError(response.get("error", "TRT86 inference failed"))

                rows: list[tuple[list[float], float]] = []
                for row in response.get("boxes", []):
                    if not isinstance(row, (list, tuple)) or len(row) != 5:
                        raise RuntimeError(f"TRT86 invalid detection row: {row!r}")
                    x1, y1, x2, y2, score = row
                    rows.append(
                        (
                            [float(x1), float(y1), float(x2), float(y2)],
                            float(score),
                        )
                    )
                output[str(cid)] = rows

                prep_ms = float(response.get("prep_ms", 0.0))
                trt_ms = float(response.get("trt_ms", 0.0))
                sidecar_ms = float(response.get("total_ms", 0.0))
                trt_sum_ms += trt_ms
                total_sum_ms += roundtrip_ms

                log_n += 1
                if log_n <= 3 or log_n % 20 == 0:
                    health = response.get("health") or {}
                    print(
                        "CAMERA_CLEAN_TRT "
                        f"n={log_n} roundtrip={roundtrip_ms:.1f}ms "
                        f"shm={shm_copy_ms:.1f}ms prep={prep_ms:.1f}ms "
                        f"trt={trt_ms:.1f}ms sidecar={sidecar_ms:.1f}ms "
                        f"boxes={len(rows)} person_max={health.get('raw_person_max_conf')}",
                        flush=True,
                    )

            result_q.put(
                {
                    "type": "result",
                    "cameras": job["cameras"],
                    "captured": job["captured"],
                    "boxes": output,
                    "batch_ms": trt_sum_ms or total_sum_ms,
                    "total_ms": total_sum_ms,
                }
            )
    except BaseException as exc:
        try:
            result_q.put(
                {"type": "fatal", "error": f"{type(exc).__name__}: {exc}"},
                timeout=1.0,
            )
        except Exception:
            pass
    finally:
        if proc is not None:
            try:
                if proc.poll() is None and proc.stdin is not None:
                    proc.stdin.write('{"cmd":"stop"}\n')
                    proc.stdin.flush()
                    proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.terminate()
                    proc.wait(timeout=1.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        if shm is not None:
            try:
                shm.close()
            finally:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
