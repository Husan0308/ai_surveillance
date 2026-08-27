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
BATCH_SIZE = 6
FRAME_BYTES = INFER_WIDTH * INFER_HEIGHT * 3
BATCH_BYTES = BATCH_SIZE * FRAME_BYTES


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_json(proc, timeout: float):
    if proc.stdout is None:
        raise RuntimeError("TRT86 batch worker stdout unavailable")
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        raise TimeoutError(f"TRT86 batch detector timeout after {timeout:.1f}s")
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError(f"TRT86 batch detector closed rc={proc.poll()}")
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"TRT86 batch worker emitted non-JSON stdout: {line[:160]!r}") from exc


def yolo_trt86_batch6_worker(job_q, result_q) -> None:
    """One TRT8.6 batch-6 sidecar for all six cameras.

    The old bridge performed six separate batch-1 TensorRT round trips. V8 copies the
    six latest camera frames into one shared-memory batch and performs exactly one
    TensorRT enqueue. The bridge deliberately has no knowledge of NvDCF and owns no
    application GPU lock; DeepStream tracking is never dropped because detection runs.
    """
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    proc = None
    shm = None
    try:
        python_path = _resolve(
            os.environ.get("CAMERA_V8_TRT86_PYTHON", ".venv-trt86/bin/python")
        )
        worker_path = _resolve(
            os.environ.get(
                "CAMERA_V8_TRT86_BATCH_WORKER",
                "scripts/yolo26_trt86_batch6_worker_v8.py",
            )
        )
        engine_path = _resolve(
            os.environ.get(
                "CAMERA_V8_TRT86_ENGINE",
                "artifacts/yolo26s_trt86/yolo26s-672x384-b6-fp32-trt86.engine",
            )
        )
        for path, label in (
            (python_path, "python"),
            (worker_path, "worker"),
            (engine_path, "batch-6 engine"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"V8 TRT86 {label} missing: {path}")

        shm = shared_memory.SharedMemory(create=True, size=BATCH_BYTES)
        shm_batch = np.ndarray(
            (BATCH_SIZE, INFER_HEIGHT, INFER_WIDTH, 3),
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
        ready = _read_json(proc, 60.0)
        if ready.get("type") != "ready":
            raise RuntimeError(f"bad V8 TRT86 handshake: {ready}")
        if not str(ready.get("tensorrt", "")).startswith("8.6.1"):
            raise RuntimeError(f"wrong TensorRT runtime: {ready}")
        if tuple(ready.get("input_shape", ())) != (6, 3, 384, 672):
            raise RuntimeError(f"wrong V8 input shape: {ready}")
        if tuple(ready.get("output_shape", ())) != (6, 300, 6):
            raise RuntimeError(f"wrong V8 output shape: {ready}")

        result_q.put(
            {
                "type": "ready",
                "device": "NVIDIA/TensorRT86",
                "cuda": "TRT8.6.1",
                "model": str(engine_path),
                "backend": "trt86-sidecar-shm-batch6-v8",
                "batch_size": BATCH_SIZE,
                "capture_policy": "coalesced-latest-six",
            }
        )

        request_id = 0
        log_n = 0
        while True:
            job = job_q.get()
            if job is None:
                return

            cameras = list(job.get("cameras") or [])
            frames = list(job.get("frames") or [])
            captured = list(job.get("captured") or [])
            if len(cameras) != BATCH_SIZE or len(frames) != BATCH_SIZE:
                raise RuntimeError(
                    f"V8 requires exactly {BATCH_SIZE} cameras per detector batch, "
                    f"got cameras={len(cameras)} frames={len(frames)}"
                )

            copy_started = time.perf_counter()
            for index, (cid, frame) in enumerate(zip(cameras, frames)):
                if frame.shape != (INFER_HEIGHT, INFER_WIDTH, 3):
                    raise RuntimeError(
                        f"{cid}: expected BGR {INFER_WIDTH}x{INFER_HEIGHT}, got {frame.shape}"
                    )
                if frame.dtype != np.uint8:
                    raise RuntimeError(f"{cid}: expected uint8, got {frame.dtype}")
                np.copyto(shm_batch[index], frame, casting="no")
            shm_copy_ms = (time.perf_counter() - copy_started) * 1000.0

            request_id += 1
            request = {
                "id": request_id,
                "shm_name": shm.name,
                "conf": float(os.environ.get("CAMERA_V2_DETECT_CONF", "0.18")),
                "max_det": max(
                    1,
                    min(300, int(os.environ.get("CAMERA_V2_MAX_DET", "20"))),
                ),
            }
            if proc.stdin is None:
                raise RuntimeError("TRT86 batch worker stdin unavailable")
            roundtrip_started = time.perf_counter()
            proc.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            proc.stdin.flush()
            response = _read_json(proc, 10.0)
            roundtrip_ms = (time.perf_counter() - roundtrip_started) * 1000.0

            if response.get("id") != request_id:
                raise RuntimeError("V8 TRT86 response ID mismatch")
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "V8 TRT86 inference failed"))

            batch_boxes = response.get("boxes") or []
            if len(batch_boxes) != BATCH_SIZE:
                raise RuntimeError(
                    f"V8 TRT86 returned {len(batch_boxes)} batch outputs, expected {BATCH_SIZE}"
                )

            output: dict[str, list[tuple[list[float], float]]] = {}
            for cid, detection_rows in zip(cameras, batch_boxes):
                rows: list[tuple[list[float], float]] = []
                for row in detection_rows:
                    if not isinstance(row, (list, tuple)) or len(row) != 5:
                        raise RuntimeError(f"V8 TRT86 invalid detection row: {row!r}")
                    x1, y1, x2, y2, score = row
                    rows.append(
                        (
                            [float(x1), float(y1), float(x2), float(y2)],
                            float(score),
                        )
                    )
                output[str(cid)] = rows

            trt_ms = float(response.get("trt_ms", 0.0))
            prep_ms = float(response.get("prep_ms", 0.0))
            total_ms = float(response.get("total_ms", roundtrip_ms))
            log_n += 1
            if log_n <= 5 or log_n % 10 == 0:
                counts = ",".join(f"{cid}:{len(output.get(cid, []))}" for cid in cameras)
                print(
                    "CAMERA_V8_TRT_BATCH "
                    f"n={log_n} batch=6 shm={shm_copy_ms:.1f}ms prep={prep_ms:.1f}ms "
                    f"gpu={trt_ms:.1f}ms sidecar={total_ms:.1f}ms roundtrip={roundtrip_ms:.1f}ms "
                    f"counts=[{counts}]",
                    flush=True,
                )

            result_q.put(
                {
                    "type": "result",
                    "cameras": cameras,
                    "captured": captured,
                    "boxes": output,
                    "batch_ms": trt_ms,
                    "total_ms": roundtrip_ms,
                    "prep_ms": prep_ms,
                    "shm_copy_ms": shm_copy_ms,
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
