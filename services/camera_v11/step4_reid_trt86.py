from __future__ import annotations

import json
import select
import subprocess
import threading
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRT86_RUNTIME_SITE = ROOT / "artifacts/reid/python_trt86_site"
MAX_BATCH = 8
INPUT_SHAPE = (MAX_BATCH, 3, 256, 128)
OUTPUT_SHAPE = (MAX_BATCH, 256)
OFFSETS = np.asarray([123.675, 116.280, 103.530], dtype=np.float32).reshape(1, 1, 3)
SCALE = np.float32(0.01735207)


def preprocess_bgr(crop: np.ndarray) -> np.ndarray:
    if crop is None or crop.size == 0:
        raise ValueError("empty ReID crop")
    if crop.ndim != 3 or crop.shape[2] != 3:
        raise ValueError(f"expected BGR HxWx3 crop, got {crop.shape}")
    target_w, target_h = 128, 256
    src_h, src_w = crop.shape[:2]
    scale = min(target_w / max(1, src_w), target_h / max(1, src_h))
    new_w = max(1, min(target_w, round(src_w * scale)))
    new_h = max(1, min(target_h, round(src_h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    rgb = resized[:, :, ::-1]
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = rgb
    x = (canvas.astype(np.float32) - OFFSETS) * SCALE
    return np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32)


class V11ReIDTRT86Client:
    def __init__(self, *, engine: str | Path | None = None, python: str | Path | None = None,
                 worker: str | Path | None = None, timeout_sec: float = 5.0) -> None:
        self.engine = self._resolve(engine or "artifacts/reid/resnet50_market1501_aicity156_b1-8_fp32_trt86.engine")
        self.python = self._resolve(python or ".venv-trt86/bin/python")
        self.worker = self._resolve(worker or "scripts/reid_trt86_worker_v11.py")
        self.runtime_site = TRT86_RUNTIME_SITE.resolve()
        self.timeout_sec = max(1.0, float(timeout_sec))
        for path, label in ((self.engine, "engine"), (self.python, "python"), (self.worker, "worker")):
            if not path.is_file():
                raise FileNotFoundError(f"ReID {label} missing: {path}")
        if not self.runtime_site.is_dir():
            raise FileNotFoundError(
                f"ReID TRT86 runtime site missing: {self.runtime_site}; "
                "run scripts/ensure_camera_v11_trt86_runtime_v1.sh"
            )
        self.input_shm = shared_memory.SharedMemory(create=True, size=int(np.prod(INPUT_SHAPE) * 4))
        self.output_shm = shared_memory.SharedMemory(create=True, size=int(np.prod(OUTPUT_SHAPE) * 4))
        self.input = np.ndarray(INPUT_SHAPE, dtype=np.float32, buffer=self.input_shm.buf)
        self.output = np.ndarray(OUTPUT_SHAPE, dtype=np.float32, buffer=self.output_shm.buf)
        self.input.fill(0.0)
        self.output.fill(0.0)
        self._lock = threading.RLock()
        self._request_id = 0

        # Keep Python isolated (-I), but prepend one project-local dependency
        # overlay before executing the unchanged TensorRT 8.6 worker. This avoids
        # relying on a broken /usr/lib NumPy while never modifying system Python.
        bootstrap = (
            "import runpy,sys;"
            f"sys.path.insert(0,{str(self.runtime_site)!r});"
            f"sys.argv[0]={str(self.worker)!r};"
            f"runpy.run_path({str(self.worker)!r},run_name='__main__')"
        )
        self.proc = subprocess.Popen(
            [str(self.python), "-I", "-c", bootstrap, "--engine", str(self.engine),
             "--input-shm", self.input_shm.name, "--output-shm", self.output_shm.name],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        try:
            row = json.loads(self._readline())
            if row.get("type") != "ready" or int(row.get("embedding_size", 0)) != 256:
                raise RuntimeError(f"ReID worker handshake failed: {row}")
            if not str(row.get("tensorrt", "")).startswith("8.6.1"):
                raise RuntimeError(f"wrong TensorRT worker: {row}")
            self.info = row
        except Exception:
            self.close()
            raise

    @staticmethod
    def _resolve(value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

    def _readline(self) -> str:
        if self.proc.stdout is None:
            raise RuntimeError("worker stdout unavailable")
        ready, _, _ = select.select([self.proc.stdout], [], [], self.timeout_sec)
        if not ready:
            raise TimeoutError(f"ReID worker timeout after {self.timeout_sec:.1f}s")
        line = self.proc.stdout.readline()
        if not line:
            stderr = self.proc.stderr.read() if self.proc.stderr is not None else ""
            raise RuntimeError(f"ReID worker exited rc={self.proc.poll()} stderr={stderr[-1000:]}")
        return line

    def embed_preprocessed(self, batch: np.ndarray) -> tuple[np.ndarray, dict]:
        data = np.asarray(batch, dtype=np.float32)
        if data.ndim == 3:
            data = data[None, ...]
        if data.ndim != 4 or tuple(data.shape[1:]) != (3, 256, 128):
            raise ValueError(f"expected Nx3x256x128, got {data.shape}")
        n = int(data.shape[0])
        if not 1 <= n <= MAX_BATCH:
            raise ValueError(f"batch must be 1..{MAX_BATCH}, got {n}")
        with self._lock:
            np.copyto(self.input[:n], np.ascontiguousarray(data), casting="no")
            self._request_id += 1
            request_id = self._request_id
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps({"id": request_id, "n": n}, separators=(",", ":")) + "\n")
            self.proc.stdin.flush()
            response = json.loads(self._readline())
            if response.get("id") != request_id or not response.get("ok"):
                raise RuntimeError(f"ReID worker request failed: {response}")
            embeddings = self.output[:n].copy()
        if not np.isfinite(embeddings).all():
            raise RuntimeError("non-finite ReID embedding")
        return embeddings, dict(response.get("stages") or {})

    def embed_crops(self, crops: list[np.ndarray]) -> tuple[np.ndarray, dict]:
        if not crops:
            return np.empty((0, 256), dtype=np.float32), {}
        batch = np.stack([preprocess_bgr(crop) for crop in crops], axis=0)
        return self.embed_preprocessed(batch)

    def close(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            try:
                assert proc.stdin is not None
                proc.stdin.write('{"cmd":"stop"}\n')
                proc.stdin.flush()
                proc.wait(timeout=3)
            except Exception:
                proc.terminate()
                proc.wait(timeout=3)
        for shm in (getattr(self, "input_shm", None), getattr(self, "output_shm", None)):
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

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
