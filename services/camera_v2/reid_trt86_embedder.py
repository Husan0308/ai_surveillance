from __future__ import annotations

import base64
import json
import select
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


class Trt86SubprocessEmbedder:
    """Isolated NVIDIA ReIdentificationNet TensorRT 8.6 worker.

    DeepStream remains in the system Python/TRT10 process.
    TensorRT 8.6 is loaded only by .venv-trt86/bin/python.
    """

    def __init__(self, config: dict | None = None, root: Path | None = None) -> None:
        cfg = dict(config or {})
        self.root = Path(root or ROOT)

        self.python_path = self._resolve(
            cfg.get("trt86_python", ".venv-trt86/bin/python")
        )
        self.worker_path = self._resolve(
            cfg.get("trt86_worker", "scripts/reid_trt86_worker.py")
        )
        self.engine_path = self._resolve(
            cfg.get(
                "trt86_engine",
                "artifacts/reid/"
                "resnet50_market1501_aicity156_b1-8_fp32_trt86.engine",
            )
        )

        self.timeout_sec = max(
            1.0,
            float(cfg.get("trt86_timeout_sec", 5.0)),
        )
        self.jpeg_quality = max(
            90,
            min(100, int(cfg.get("trt86_jpeg_quality", 95))),
        )

        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._request_id = 0

        self._batches = 0
        self._images = 0
        self._starts = 0
        self._errors = 0
        self._last_batch_ms = 0.0
        self._last_worker_ms = 0.0
        self._last_error = ""
        self._worker_info: dict = {}

    def _resolve(self, value) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else self.root / path

    def _stop_locked(self) -> None:
        proc = self._proc
        self._proc = None

        if proc is None:
            return

        if proc.poll() is None:
            try:
                if proc.stdin is not None:
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

        for stream in (proc.stdin, proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass

    def _readline_locked(self, timeout: float) -> str:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise RuntimeError("TRT86 worker stdout unavailable")

        ready, _, _ = select.select(
            [proc.stdout],
            [],
            [],
            timeout,
        )

        if not ready:
            raise TimeoutError(
                f"TRT86 worker timeout after {timeout:.1f}s"
            )

        line = proc.stdout.readline()

        if not line:
            rc = proc.poll()
            raise RuntimeError(
                f"TRT86 worker closed stdout rc={rc}"
            )

        return line

    def _start_locked(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return

        self._stop_locked()

        for path, label in (
            (self.python_path, "python"),
            (self.worker_path, "worker"),
            (self.engine_path, "engine"),
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"TRT86 {label} missing: {path}"
                )

        self._proc = subprocess.Popen(
            [
                str(self.python_path),
                str(self.worker_path),
                "--engine",
                str(self.engine_path),
            ],
            cwd=self.root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        try:
            row = json.loads(
                self._readline_locked(self.timeout_sec)
            )

            if row.get("type") != "ready":
                raise RuntimeError(
                    f"unexpected worker handshake: {row}"
                )

            if not str(row.get("tensorrt", "")).startswith("8.6.1"):
                raise RuntimeError(
                    f"wrong worker TensorRT: {row}"
                )

            if int(row.get("embedding_size", 0)) != 256:
                raise RuntimeError(
                    f"wrong embedding size: {row}"
                )

            self._worker_info = dict(row)
            self._starts += 1
            self._last_error = ""

        except Exception:
            self._stop_locked()
            raise

    def embed_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, 256), dtype=np.float32)

        if len(crops) > 8:
            raise ValueError(
                f"TRT86 ReID max batch is 8, got {len(crops)}"
            )

        encoded = []

        for crop in crops:
            if crop is None or crop.size == 0:
                raise ValueError("empty ReID crop")

            # Production crop is BGR. cv2 JPEG encoding preserves the
            # correct colors; PIL in the worker decodes it as RGB.
            ok, jpg = cv2.imencode(
                ".jpg",
                crop,
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    self.jpeg_quality,
                ],
            )

            if not ok:
                raise RuntimeError(
                    "failed to JPEG-encode ReID crop"
                )

            encoded.append(
                base64.b64encode(
                    jpg.tobytes()
                ).decode("ascii")
            )

        started = time.perf_counter()

        with self._lock:
            try:
                self._start_locked()

                proc = self._proc
                if (
                    proc is None
                    or proc.stdin is None
                ):
                    raise RuntimeError(
                        "TRT86 worker stdin unavailable"
                    )

                self._request_id += 1
                request_id = self._request_id

                request = {
                    "id": request_id,
                    "jpeg_b64": encoded,
                }

                proc.stdin.write(
                    json.dumps(
                        request,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                proc.stdin.flush()

                response = json.loads(
                    self._readline_locked(
                        self.timeout_sec
                    )
                )

                if response.get("id") != request_id:
                    raise RuntimeError(
                        "TRT86 worker response ID mismatch"
                    )

                if not response.get("ok"):
                    raise RuntimeError(
                        response.get(
                            "error",
                            "TRT86 worker inference failed",
                        )
                    )

                shape = tuple(
                    int(x)
                    for x in response["shape"]
                )

                if shape != (len(crops), 256):
                    raise RuntimeError(
                        f"wrong ReID output shape={shape}"
                    )

                output = np.frombuffer(
                    base64.b64decode(
                        response["embedding_b64"]
                    ),
                    dtype=np.float32,
                ).reshape(shape).copy()

                if not np.isfinite(output).all():
                    raise RuntimeError(
                        "non-finite ReID embedding"
                    )

                # Worker already normalizes, but normalize once more
                # defensively at the process boundary.
                norms = np.linalg.norm(
                    output,
                    axis=1,
                    keepdims=True,
                )

                output /= np.maximum(
                    norms,
                    1e-12,
                )

                self._batches += 1
                self._images += len(crops)
                self._last_worker_ms = float(
                    response.get("infer_ms", 0.0)
                )
                self._last_batch_ms = (
                    time.perf_counter() - started
                ) * 1000.0
                self._last_error = ""

                return output

            except Exception as exc:
                self._errors += 1
                self._last_error = (
                    f"{type(exc).__name__}:{exc}"
                )
                self._stop_locked()
                raise

    def close(self) -> None:
        with self._lock:
            self._stop_locked()

    def metrics(self) -> dict:
        with self._lock:
            proc = self._proc
            ready = (
                proc is not None
                and proc.poll() is None
            )

            return {
                "backend": "nvidia-reid-trt86-worker",
                "ready": ready,
                "python": str(self.python_path),
                "worker": str(self.worker_path),
                "engine": str(self.engine_path),
                "worker_info": dict(self._worker_info),
                "starts": self._starts,
                "batches": self._batches,
                "images": self._images,
                "errors": self._errors,
                "last_batch_ms": self._last_batch_ms,
                "worker_infer_ms": self._last_worker_ms,
                "last_error": self._last_error,
            }
