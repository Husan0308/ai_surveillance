from __future__ import annotations

import json
import os
import select
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
        raise RuntimeError("TRT86 SHM worker stdout unavailable")
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        raise TimeoutError(f"TRT86 SHM detector timeout after {timeout:.1f}s")
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError(f"TRT86 SHM detector closed rc={proc.poll()}")
    return json.loads(line)


def yolo_trt86_shm_worker(job_q, result_q) -> None:
    proc = None
    shm = None
    try:
        python_path = _resolve(
            os.environ.get("CAMERA_V2_TRT86_PYTHON", ".venv-trt86/bin/python")
        )
        worker_path = _resolve(
            os.environ.get(
                "CAMERA_V2_TRT86_SHM_WORKER",
                "scripts/yolo26_trt86_shm_worker.py",
            )
        )
        engine_path = _resolve(
            os.environ.get(
                "CAMERA_V2_TRT86_ENGINE",
                "artifacts/yolo26s_trt86/"
                "yolo26s-672x384-b1-fp32-trt86.engine",
            )
        )
        for path, name in (
            (python_path, "python"),
            (worker_path, "worker"),
            (engine_path, "engine"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"TRT86 {name} missing: {path}")

        shm = shared_memory.SharedMemory(create=True, size=FRAME_BYTES)
        shm_frame = np.ndarray(
            (INFER_HEIGHT, INFER_WIDTH, 3),
            dtype=np.uint8,
            buffer=shm.buf,
        )

        proc = subprocess.Popen(
            [
                str(python_path),
                str(worker_path),
                "--engine",
                str(engine_path),
            ],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        ready = _read_json(proc, 30.0)
        if ready.get("type") != "ready":
            raise RuntimeError(f"bad TRT86 SHM handshake: {ready}")
        if not str(ready.get("tensorrt", "")).startswith("8.6.1"):
            raise RuntimeError(f"wrong TensorRT: {ready}")
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
                "backend": "trt86-sidecar-shm-bgr",
                "capture_policy": "latest",
            }
        )

        request_id = 0
        breakdown_n = 0

        while True:
            job = job_q.get()
            if job is None:
                return

            output = {}
            trt_sum_ms = 0.0
            total_sum_ms = 0.0

            for cid, frame in zip(job["cameras"], job["frames"]):
                if frame.shape != (INFER_HEIGHT, INFER_WIDTH, 3):
                    raise RuntimeError(
                        f"{cid}: expected BGR {INFER_WIDTH}x{INFER_HEIGHT}, "
                        f"got shape={frame.shape}"
                    )
                if frame.dtype != np.uint8:
                    raise RuntimeError(f"{cid}: expected uint8, got {frame.dtype}")

                started = time.perf_counter()
                np.copyto(shm_frame, frame, casting="no")
                shm_copy_ms = (time.perf_counter() - started) * 1000.0

                request_id += 1
                req = {
                    "id": request_id,
                    "shm_name": shm.name,
                    "conf": float(
                        os.environ.get("CAMERA_V2_DETECT_CONF", "0.05")
                    ),
                }
                if proc.stdin is None:
                    raise RuntimeError("TRT86 SHM stdin unavailable")
                roundtrip_started = time.perf_counter()
                proc.stdin.write(json.dumps(req, separators=(",", ":")) + "\n")
                proc.stdin.flush()
                response = _read_json(proc, 5.0)
                roundtrip_ms = (time.perf_counter() - roundtrip_started) * 1000.0

                if response.get("id") != request_id:
                    raise RuntimeError("TRT86 SHM response ID mismatch")
                if not response.get("ok"):
                    raise RuntimeError(
                        response.get("error", "TRT86 SHM inference failed")
                    )

                rows = []
                for row in response.get("boxes", []):
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

                breakdown_n += 1
                if breakdown_n <= 3 or breakdown_n % 20 == 0:
                    health = response.get("health") or {}
                    print(
                        "CAM01_TRT86_BREAKDOWN "
                        f"n={breakdown_n} roundtrip={roundtrip_ms:.1f}ms "
                        f"shm_copy={shm_copy_ms:.1f}ms "
                        f"prep={prep_ms:.1f}ms trt={trt_ms:.1f}ms "
                        f"sidecar={sidecar_ms:.1f}ms boxes={len(rows)} "
                        f"raw_max_conf={health.get('raw_max_conf')} "
                        f"raw_person_rows={health.get('raw_person_rows')} "
                        f"raw_above_conf={health.get('raw_above_conf')} "
                        f"nonfinite={health.get('nonfinite_rows')} "
                        f"bgr_mean={health.get('bgr_mean')}",
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
        result_q.put(
            {
                "type": "fatal",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
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
