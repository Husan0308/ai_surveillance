from __future__ import annotations

import hashlib
from pathlib import Path
import time
import urllib.request

import cv2
import numpy as np


class OsnetEmbedder:
    """Lazy CPU-only OSNet feature extractor.

    The detector owns CUDA. ReID intentionally stays on CPU so model loading or
    inference can never contend with the six-camera YOLO hot path. Torchreid is
    imported lazily; any failure is reported through metrics instead of taking
    down the ML service.
    """

    def __init__(self, config: dict, root: Path):
        cfg = dict(config or {})
        self.model_name = str(cfg.get("model_name", "osnet_x0_25"))
        self.device = str(cfg.get("device", "cpu")).strip().lower()
        self.input_height = int(cfg.get("input_height", 256))
        self.input_width = int(cfg.get("input_width", 128))
        self.cpu_threads = max(1, int(cfg.get("cpu_threads", 2)))
        self.model_path = root / str(
            cfg.get(
                "model_path",
                "models/reid/osnet_x0_25_msmt17_combineall.pth",
            )
        )
        self.model_url = str(cfg.get("model_url", ""))
        self.model_sha256 = str(cfg.get("model_sha256", "")).lower().strip()
        self.download_if_missing = bool(cfg.get("download_if_missing", True))

        self._torch = None
        self._model = None
        self._ready = False
        self._loading = False
        self._last_error = ""
        self._loaded_at = 0.0
        self._batches = 0
        self._images = 0
        self._last_batch_ms = 0.0
        self._batch_sizes: list[int] = []
        self._batch_latencies_ms: list[float] = []
        self._last_download_ms = 0.0

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _verify(self, path: Path) -> None:
        if not self.model_sha256:
            return
        actual = self._sha256(path)
        if actual != self.model_sha256:
            raise RuntimeError(
                f"ReID checkpoint SHA256 mismatch: expected {self.model_sha256}, got {actual}"
            )

    def _ensure_weights(self) -> None:
        if self.model_path.exists():
            self._verify(self.model_path)
            return
        if not self.download_if_missing or not self.model_url:
            raise FileNotFoundError(f"ReID checkpoint not found: {self.model_path}")

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.model_path.with_suffix(self.model_path.suffix + ".part")
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(self.model_url, timeout=45) as response:
                with temp_path.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            self._verify(temp_path)
            temp_path.replace(self.model_path)
        finally:
            self._last_download_ms = (time.perf_counter() - started) * 1000.0
            if temp_path.exists() and not self.model_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def _load(self) -> None:
        if self._ready:
            return
        if self._loading:
            raise RuntimeError("ReID embedder is already loading")
        self._loading = True
        try:
            self._ensure_weights()
            import torch
            from torchreid import models
            try:
                from torchreid.utils import load_pretrained_weights
            except ModuleNotFoundError:
                from torchreid.reid.utils import load_pretrained_weights

            if self.device == "cpu":
                torch.set_num_threads(self.cpu_threads)
                try:
                    torch.set_num_interop_threads(1)
                except RuntimeError:
                    pass
            elif not torch.cuda.is_available():
                raise RuntimeError("OSNet requested CUDA but torch.cuda.is_available() is false")

            model = models.build_model(
                name=self.model_name,
                num_classes=1000,
                loss="softmax",
                pretrained=False,
                use_gpu=self.device == "cuda",
            )
            load_pretrained_weights(model, str(self.model_path))
            model.eval()
            model.to(self.device)

            self._torch = torch
            self._model = model
            self._ready = True
            self._last_error = ""
            self._loaded_at = time.monotonic()
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._ready = False
            raise
        finally:
            self._loading = False

    def _tensor(self, crop: np.ndarray):
        if crop is None or crop.size == 0:
            raise ValueError("empty ReID crop")
        image = cv2.resize(
            crop,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        array = image.astype(np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        array = (array - mean) / std
        return self._torch.from_numpy(np.ascontiguousarray(array)).to(self.device)

    def embed_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, 512), dtype=np.float32)
        if not self._ready:
            self._load()
        started = time.perf_counter()
        torch = self._torch
        batch = torch.stack([self._tensor(crop) for crop in crops], dim=0)
        with torch.inference_mode():
            features = self._model(batch)
            features = torch.nn.functional.normalize(features.float(), p=2, dim=1)
        if self.device == "cuda":
            torch.cuda.synchronize()
        result = features.detach().cpu().numpy().astype(np.float32, copy=False)
        self._batches += 1
        self._images += len(crops)
        self._last_batch_ms = (time.perf_counter() - started) * 1000.0
        self._batch_sizes.append(len(crops))
        self._batch_latencies_ms.append(self._last_batch_ms)
        return result

    def metrics(self) -> dict:
        return {
            "ready": self._ready,
            "loading": self._loading,
            "device": self.device,
            "model_name": self.model_name,
            "model_path": str(self.model_path),
            "model_exists": self.model_path.exists(),
            "checkpoint_sha256": self.model_sha256,
            "last_error": self._last_error,
            "batches": self._batches,
            "images": self._images,
            "last_batch_ms": self._last_batch_ms,
            "batch_sizes": list(self._batch_sizes),
            "batch_latency_ms": list(self._batch_latencies_ms),
            "last_download_ms": self._last_download_ms,
            "loaded_for_sec": max(0.0, time.monotonic() - self._loaded_at)
            if self._loaded_at
            else 0.0,
        }


class OsnetCpuEmbedder(OsnetEmbedder):
    """Backward-compatible explicit CPU adapter for the offline sidecar."""

    def __init__(self, config: dict, root: Path):
        cfg = dict(config or {})
        cfg["device"] = "cpu"
        super().__init__(cfg, root)
