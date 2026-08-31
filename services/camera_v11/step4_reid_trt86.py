from __future__ import annotations

import json
import math
import os
import select
import subprocess
import threading
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRT86_RUNTIME_SITE = ROOT / "artifacts/reid/python_trt86_site"
TRT86_RUNTIME_MANIFEST = TRT86_RUNTIME_SITE / ".v11_runtime_paths.json"
MAX_BATCH = 8
EMBEDDING_DIMENSION = 256
INPUT_SHAPE = (MAX_BATCH, 3, 256, 128)
OUTPUT_SHAPE = (MAX_BATCH, EMBEDDING_DIMENSION)
OFFSETS = np.asarray([123.675, 116.280, 103.530], dtype=np.float32).reshape(1, 1, 3)
SCALE = np.float32(0.01735207)


def preprocess_bgr(crop: np.ndarray) -> np.ndarray:
    if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
        raise ValueError("empty ReID crop")
    if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3:
        raise ValueError(f"expected uint8 BGR HxWx3 crop, got {crop.shape}")
    target_w, target_h = 128, 256
    src_h, src_w = crop.shape[:2]
    scale = min(target_w / max(1, src_w), target_h / max(1, src_h))
    new_w = max(1, min(target_w, round(src_w * scale)))
    new_h = max(1, min(target_h, round(src_h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    rgb = resized[:, :, ::-1]
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    # Preserve the exact validated keepAspc top-left padding convention.
    canvas[:new_h, :new_w] = rgb
    values = (canvas.astype(np.float32) - OFFSETS) * SCALE
    return np.ascontiguousarray(values.transpose(2, 0, 1), dtype=np.float32)


class V11ReIDTRT86Client:
    """Isolated TensorRT 8.6.1 FP32 client using fixed shared-memory buffers."""

    def __init__(
        self,
        *,
        engine: str | Path | None = None,
        python: str | Path | None = None,
        worker: str | Path | None = None,
        timeout_sec: float = 8.0,
    ) -> None:
        self.engine = self._resolve(
            engine
            or os.environ.get(
                "V11_STEP4_REID_ENGINE",
                "artifacts/reid/resnet50_market1501_aicity156_b1-8_fp32_trt86.engine",
            )
        )
        self.python = self._resolve(
            python
            or os.environ.get("V11_STEP4_REID_PYTHON", ".venv-trt86/bin/python")
        )
        self.worker = self._resolve(
            worker
            or os.environ.get(
                "V11_STEP4_REID_WORKER", "scripts/reid_trt86_worker_v11.py"
            )
        )
        self.runtime_site = TRT86_RUNTIME_SITE.resolve()
        self.runtime_manifest = TRT86_RUNTIME_MANIFEST.resolve()
        self.timeout_sec = max(1.0, float(timeout_sec))
        for path, label in (
            (self.engine, "FP32 engine"),
            (self.python, "TRT86 Python"),
            (self.worker, "worker"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"ReID {label} missing: {path}")
        if not self.runtime_site.is_dir() or not self.runtime_manifest.is_file():
            raise FileNotFoundError(
                f"ReID TRT86 runtime overlay missing: {self.runtime_site}"
            )
        try:
            runtime = json.loads(self.runtime_manifest.read_text(encoding="utf-8"))
            self.tensorrt_root = Path(runtime["tensorrt_root"]).expanduser().resolve()
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"invalid ReID TRT86 runtime manifest: {self.runtime_manifest}: {exc}"
            ) from exc
        if not self.tensorrt_root.is_dir():
            raise FileNotFoundError(
                f"ReID TensorRT module root missing: {self.tensorrt_root}"
            )

        self.input_shm = shared_memory.SharedMemory(
            create=True, size=int(np.prod(INPUT_SHAPE) * 4)
        )
        self.output_shm = shared_memory.SharedMemory(
            create=True, size=int(np.prod(OUTPUT_SHAPE) * 4)
        )
        self.input = np.ndarray(INPUT_SHAPE, dtype=np.float32, buffer=self.input_shm.buf)
        self.output = np.ndarray(OUTPUT_SHAPE, dtype=np.float32, buffer=self.output_shm.buf)
        self.input.fill(0.0)
        self.output.fill(0.0)
        self._lock = threading.RLock()
        self._request_id = 0
        self.proc: subprocess.Popen[str] | None = None

        bootstrap = (
            "import runpy,sys;"
            f"sys.path[:0]=[{str(self.runtime_site)!r},{str(self.tensorrt_root)!r}];"
            f"sys.argv[0]={str(self.worker)!r};"
            f"runpy.run_path({str(self.worker)!r},run_name='__main__')"
        )
        self.proc = subprocess.Popen(
            [
                str(self.python),
                "-I",
                "-c",
                bootstrap,
                "--engine",
                str(self.engine),
                "--input-shm",
                self.input_shm.name,
                "--output-shm",
                self.output_shm.name,
            ],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            row = json.loads(self._readline())
            if (
                row.get("type") != "ready"
                or int(row.get("embedding_size", 0)) != EMBEDDING_DIMENSION
                or int(row.get("max_batch", 0)) != MAX_BATCH
            ):
                raise RuntimeError(f"ReID worker handshake failed: {row}")
            if not str(row.get("tensorrt", "")).startswith("8.6.1"):
                raise RuntimeError(f"wrong TensorRT worker: {row}")
            if row.get("precision") != "fp32":
                raise RuntimeError(f"wrong ReID precision: {row}")
            self.info = row
        except Exception:
            self.close()
            raise

    @staticmethod
    def _resolve(value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

    def _readline(self) -> str:
        proc = self.proc
        if proc is None or proc.stdout is None:
            raise RuntimeError("ReID worker stdout unavailable")
        ready, _, _ = select.select([proc.stdout], [], [], self.timeout_sec)
        if not ready:
            raise TimeoutError(f"ReID worker timeout after {self.timeout_sec:.1f}s")
        line = proc.stdout.readline()
        if not line:
            stderr = proc.stderr.read() if proc.stderr is not None else ""
            raise RuntimeError(
                f"ReID worker exited rc={proc.poll()} stderr={stderr[-1200:]}"
            )
        return line

    @staticmethod
    def _validate_embeddings(embeddings: np.ndarray, count: int) -> np.ndarray:
        values = np.asarray(embeddings, dtype=np.float32)
        if values.shape != (count, EMBEDDING_DIMENSION):
            raise RuntimeError(f"unexpected ReID embeddings shape={values.shape}")
        if not np.isfinite(values).all():
            raise RuntimeError("non-finite ReID embedding")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
            raise RuntimeError("invalid ReID embedding norm")
        values = np.ascontiguousarray(values / norms, dtype=np.float32)
        normalized = np.linalg.norm(values, axis=1)
        if any(
            not math.isfinite(float(value)) or abs(float(value) - 1.0) > 1e-4
            for value in normalized
        ):
            raise RuntimeError("ReID embedding not L2-normalized")
        return values

    def embed_preprocessed(self, batch: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        data = np.asarray(batch, dtype=np.float32)
        if data.ndim == 3:
            data = data[None, ...]
        if data.ndim != 4 or tuple(data.shape[1:]) != (3, 256, 128):
            raise ValueError(f"expected Nx3x256x128, got {data.shape}")
        count = int(data.shape[0])
        if not 1 <= count <= MAX_BATCH:
            raise ValueError(f"batch must be 1..{MAX_BATCH}, got {count}")
        with self._lock:
            np.copyto(self.input[:count], np.ascontiguousarray(data), casting="no")
            self._request_id += 1
            request_id = self._request_id
            proc = self.proc
            if proc is None or proc.stdin is None:
                raise RuntimeError("ReID worker stdin unavailable")
            proc.stdin.write(
                json.dumps({"id": request_id, "n": count}, separators=(",", ":"))
                + "\n"
            )
            proc.stdin.flush()
            response = json.loads(self._readline())
            if response.get("id") != request_id or not response.get("ok"):
                raise RuntimeError(f"ReID worker request failed: {response}")
            embeddings = self.output[:count].copy()
        return self._validate_embeddings(embeddings, count), {
            str(key): float(value)
            for key, value in dict(response.get("stages") or {}).items()
        }

    def embed_crops(self, crops: list[np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
        if not crops:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32), {}
        batch = np.stack([preprocess_bgr(crop) for crop in crops], axis=0)
        return self.embed_preprocessed(batch)

    def close(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write('{"cmd":"stop"}\n')
                    proc.stdin.flush()
                proc.wait(timeout=3.0)
            except Exception:
                proc.terminate()
                proc.wait(timeout=3.0)
        self.proc = None
        for shm in (
            getattr(self, "input_shm", None),
            getattr(self, "output_shm", None),
        ):
            if shm is not None:
                try:
                    shm.close()
                finally:
                    try:
                        shm.unlink()
                    except FileNotFoundError:
                        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
