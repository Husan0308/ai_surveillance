from __future__ import annotations

import base64
import json
import os
import select
import subprocess
import time
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def _read_json(proc, timeout: float):
    if proc.stdout is None:
        raise RuntimeError("TRT86 worker stdout unavailable")

    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        raise TimeoutError(
            f"TRT86 detector timeout after {timeout:.1f}s"
        )

    line = proc.stdout.readline()

    if not line:
        raise RuntimeError(
            f"TRT86 detector closed rc={proc.poll()}"
        )

    return json.loads(line)


def yolo_trt86_worker(job_q, result_q):
    proc = None

    try:
        python_path = _resolve(
            os.environ.get(
                "CAMERA_V2_TRT86_PYTHON",
                ".venv-trt86/bin/python",
            )
        )

        worker_path = _resolve(
            os.environ.get(
                "CAMERA_V2_TRT86_WORKER",
                "scripts/yolo26_trt86_worker.py",
            )
        )

        engine_path = _resolve(
            os.environ.get(
                "CAMERA_V2_TRT86_ENGINE",
                "artifacts/yolo26s_trt86/"
                "yolo26s-672x384-b1-fp32-trt86.engine",
            )
        )

        quality = max(
            80,
            min(
                100,
                int(
                    os.environ.get(
                        "CAMERA_V2_TRT86_JPEG_QUALITY",
                        "90",
                    )
                ),
            ),
        )

        for path, name in (
            (python_path, "python"),
            (worker_path, "worker"),
            (engine_path, "engine"),
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"TRT86 {name} missing: {path}"
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
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        ready = _read_json(proc, 30.0)

        if ready.get("type") != "ready":
            raise RuntimeError(
                f"bad TRT86 handshake: {ready}"
            )

        if not str(
            ready.get("tensorrt", "")
        ).startswith("8.6.1"):
            raise RuntimeError(
                f"wrong TensorRT: {ready}"
            )

        result_q.put({
            "type": "ready",
            "device": "NVIDIA GeForce GTX 1050 Ti",
            "cuda": "TRT8.6.1",
            "model": str(engine_path),
            "backend": "yolo26-trt86",
        })

        request_id = 0

        while True:
            job = job_q.get()

            if job is None:
                return

            started = time.monotonic()

            output = {}
            gpu_ms = 0.0

            for cid, frame in zip(
                job["cameras"],
                job["frames"],
            ):
                ok, jpg = cv2.imencode(
                    ".jpg",
                    frame,
                    [
                        int(cv2.IMWRITE_JPEG_QUALITY),
                        quality,
                    ],
                )

                if not ok:
                    raise RuntimeError(
                        f"{cid}: JPEG encode failed"
                    )

                request_id += 1

                req = {
                    "id": request_id,
                    "jpeg_b64": base64.b64encode(
                        jpg.tobytes()
                    ).decode("ascii"),
                    "conf": float(
                        os.environ.get(
                            "CAMERA_V2_DETECT_CONF",
                            "0.05",
                        )
                    ),
                }

                if proc.stdin is None:
                    raise RuntimeError(
                        "TRT86 stdin unavailable"
                    )

                proc.stdin.write(
                    json.dumps(
                        req,
                        separators=(",", ":"),
                    ) + "\n"
                )
                proc.stdin.flush()

                response = _read_json(proc, 5.0)

                if response.get("id") != request_id:
                    raise RuntimeError(
                        "TRT86 response ID mismatch"
                    )

                if not response.get("ok"):
                    raise RuntimeError(
                        response.get(
                            "error",
                            "TRT86 inference failed",
                        )
                    )

                rows = []

                for row in response.get(
                    "boxes",
                    [],
                ):
                    x1, y1, x2, y2, score = row

                    rows.append((
                        [
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2),
                        ],
                        float(score),
                    ))

                output[cid] = rows
                gpu_ms += float(
                    response.get("trt_ms", 0.0)
                )

            total_ms = (
                time.monotonic() - started
            ) * 1000.0

            result_q.put({
                "type": "result",
                "cameras": job["cameras"],
                "captured": job["captured"],
                "boxes": output,

                # Existing scheduler uses this for GPU duty.
                "batch_ms": gpu_ms or total_ms,

                "total_ms": total_ms,
            })

    except BaseException as exc:
        result_q.put({
            "type": "fatal",
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        })

    finally:
        if proc is not None:
            try:
                if proc.poll() is None:
                    if proc.stdin is not None:
                        proc.stdin.write(
                            '{"cmd":"stop"}\n'
                        )
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
