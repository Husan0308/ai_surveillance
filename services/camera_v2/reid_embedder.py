from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import threading
import time
import urllib.request

import cv2
import numpy as np

from .global_identity import normalize


ROOT = Path(__file__).resolve().parents[2]
OMZ_NAME = "person-reidentification-retail-0288"
OMZ_BASE = (
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/temp/"
    "person-reidentification-retail-0288/FP16"
)
OSNET_AIN_URL = (
    "https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
    "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_"
    "fb10_softmax_labsmth_flip_jitter.pth"
)
OSNET_AIN_SHA256 = "8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f"


def _download(url: str, target: Path, *, timeout: float = 60.0) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "sentinel-vms-reid/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response, temp.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        if temp.stat().st_size < 1024:
            raise RuntimeError(f"downloaded ReID model file is unexpectedly small: {url}")
        temp.replace(target)
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OpenVinoRetail0288Embedder:
    """Tiny official Open Model Zoo ReID fallback through OpenCV DNN, CPU-only."""

    def __init__(self, config: dict | None = None, root: Path | None = None) -> None:
        cfg = dict(config or {})
        root = Path(root or ROOT)
        model_dir = root / str(cfg.get("omz_model_dir", ".runtime/camera_v2/models/reid/omz0288"))
        self.xml_path = model_dir / f"{OMZ_NAME}.xml"
        self.bin_path = model_dir / f"{OMZ_NAME}.bin"
        self.download_if_missing = bool(cfg.get("download_if_missing", True))
        self.input_width = 128
        self.input_height = 256
        self._net = None
        self._lock = threading.RLock()
        self._last_error = ""
        self._batches = 0
        self._images = 0
        self._last_batch_ms = 0.0

    def _ensure_files(self) -> None:
        if self.xml_path.exists() and self.bin_path.exists():
            return
        if not self.download_if_missing:
            raise FileNotFoundError(
                f"{OMZ_NAME} missing under {self.xml_path.parent}; run setup_camera_v2_reid.py"
            )
        if not self.xml_path.exists():
            _download(f"{OMZ_BASE}/{OMZ_NAME}.xml", self.xml_path)
        if not self.bin_path.exists():
            _download(f"{OMZ_BASE}/{OMZ_NAME}.bin", self.bin_path)

    def _load(self) -> None:
        if self._net is not None:
            return
        with self._lock:
            if self._net is not None:
                return
            self._ensure_files()
            try:
                net = cv2.dnn.readNet(str(self.xml_path), str(self.bin_path))
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self._net = net
                self._last_error = ""
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                raise

    def embed_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, 256), dtype=np.float32)
        self._load()
        started = time.perf_counter()
        vectors: list[np.ndarray] = []
        with self._lock:
            for crop in crops:
                if crop is None or crop.size == 0:
                    raise ValueError("empty ReID crop")
                image = cv2.resize(crop, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
                blob = cv2.dnn.blobFromImage(
                    image, scalefactor=1.0, size=(self.input_width, self.input_height),
                    mean=(0.0, 0.0, 0.0), swapRB=False, crop=False,
                )
                self._net.setInput(blob)
                output = self._net.forward()
                vectors.append(normalize(np.asarray(output, dtype=np.float32).reshape(-1)))
        self._batches += 1
        self._images += len(crops)
        self._last_batch_ms = (time.perf_counter() - started) * 1000.0
        return np.stack(vectors, axis=0).astype(np.float32, copy=False)

    def metrics(self) -> dict:
        return {
            "backend": "opencv-omz0288-cpu",
            "ready": self._net is not None,
            "model": OMZ_NAME,
            "xml": str(self.xml_path),
            "bin": str(self.bin_path),
            "model_exists": self.xml_path.exists() and self.bin_path.exists(),
            "batches": self._batches,
            "images": self._images,
            "last_batch_ms": self._last_batch_ms,
            "last_error": self._last_error,
        }


class OsnetAinCpuEmbedder:
    """Stronger cross-domain OSNet-AIN x1.0 backend; CPU-only and lazy loaded."""

    def __init__(self, config: dict | None = None, root: Path | None = None) -> None:
        cfg = dict(config or {})
        root = Path(root or ROOT)
        self.model_path = root / str(
            cfg.get("osnet_model_path", ".runtime/camera_v2/models/reid/osnet_ain_x1_0_msmt17.pth")
        )
        self.download_if_missing = bool(cfg.get("download_if_missing", True))
        self.cpu_threads = max(1, int(cfg.get("cpu_threads", 2)))
        self._torch = None
        self._model = None
        self._lock = threading.RLock()
        self._last_error = ""
        self._batches = 0
        self._images = 0
        self._last_batch_ms = 0.0

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchreid") is not None

    def _ensure_weights(self) -> None:
        if self.model_path.exists():
            actual = _sha256(self.model_path)
            if actual != OSNET_AIN_SHA256:
                raise RuntimeError(
                    f"OSNet-AIN checkpoint SHA256 mismatch: expected {OSNET_AIN_SHA256}, got {actual}"
                )
            return
        if not self.download_if_missing:
            raise FileNotFoundError(f"OSNet-AIN checkpoint missing: {self.model_path}")
        _download(OSNET_AIN_URL, self.model_path)
        actual = _sha256(self.model_path)
        if actual != OSNET_AIN_SHA256:
            try:
                self.model_path.unlink()
            except OSError:
                pass
            raise RuntimeError(
                f"OSNet-AIN checkpoint SHA256 mismatch: expected {OSNET_AIN_SHA256}, got {actual}"
            )

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.available():
                raise RuntimeError("torchreid is not installed")
            self._ensure_weights()
            try:
                import torch
                from torchreid import models
                from torchreid.utils import load_pretrained_weights

                torch.set_num_threads(self.cpu_threads)
                try:
                    torch.set_num_interop_threads(1)
                except RuntimeError:
                    pass
                model = models.build_model(
                    name="osnet_ain_x1_0", num_classes=1000, loss="softmax",
                    pretrained=False, use_gpu=False,
                )
                load_pretrained_weights(model, str(self.model_path))
                model.eval().to("cpu")
                self._torch = torch
                self._model = model
                self._last_error = ""
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                raise

    def _tensor(self, crop: np.ndarray):
        image = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        array = image.astype(np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        return self._torch.from_numpy(np.ascontiguousarray((array - mean) / std))

    def embed_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, 512), dtype=np.float32)
        self._load()
        started = time.perf_counter()
        with self._lock:
            torch = self._torch
            batch = torch.stack([self._tensor(crop) for crop in crops], dim=0)
            with torch.inference_mode():
                output = self._model(batch)
                output = torch.nn.functional.normalize(output.float(), p=2, dim=1)
            result = output.cpu().numpy().astype(np.float32, copy=False)
        self._batches += 1
        self._images += len(crops)
        self._last_batch_ms = (time.perf_counter() - started) * 1000.0
        return result

    def metrics(self) -> dict:
        return {
            "backend": "osnet-ain-x1.0-cpu",
            "ready": self._model is not None,
            "model_path": str(self.model_path),
            "model_exists": self.model_path.exists(),
            "torchreid_available": self.available(),
            "batches": self._batches,
            "images": self._images,
            "last_batch_ms": self._last_batch_ms,
            "last_error": self._last_error,
        }


class AutoReIdEmbedder:
    """Prefer OSNet-AIN when available; keep official OMZ-0288 as CPU fallback."""

    def __init__(self, config: dict | None = None, root: Path | None = None) -> None:
        cfg = dict(config or {})
        self.requested = str(
            os.environ.get("CAMERA_V2_REID_BACKEND", cfg.get("backend", "auto"))
        ).strip().lower()
        self._active = None
        self._cfg = cfg
        self._root = Path(root or ROOT)
        self._fallback_reason = ""

    def _select(self):
        if self._active is not None:
            return self._active

        if self.requested in {
            "trt86",
            "trt86_worker",
            "trt86-worker",
            "nvidia_trt86",
        }:
            from .reid_trt86_embedder import Trt86SubprocessEmbedder

            self._active = Trt86SubprocessEmbedder(
                self._cfg,
                self._root,
            )
            return self._active
        if self.requested in {"auto", "osnet", "osnet_ain", "osnet-ain"}:
            if OsnetAinCpuEmbedder.available():
                self._active = OsnetAinCpuEmbedder(self._cfg, self._root)
                return self._active
            if self.requested != "auto":
                raise RuntimeError("OSNet-AIN requested but torchreid is unavailable")
            self._fallback_reason = "torchreid unavailable; using official OMZ-0288 CPU model"
        if self.requested not in {"auto", "omz", "omz0288", "opencv", "opencv_omz"}:
            raise ValueError(f"unsupported ReID backend: {self.requested}")
        self._active = OpenVinoRetail0288Embedder(self._cfg, self._root)
        return self._active

    def embed_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        backend = self._select()
        if isinstance(backend, OsnetAinCpuEmbedder) and self.requested == "auto":
            try:
                return backend.embed_batch(crops)
            except Exception as exc:
                self._fallback_reason = f"OSNet-AIN failed: {type(exc).__name__}: {exc}; switched to OMZ-0288"
                self._active = OpenVinoRetail0288Embedder(self._cfg, self._root)
        return self._active.embed_batch(crops)

    def metrics(self) -> dict:
        active = self._select()
        return {
            "requested": self.requested,
            "fallback_reason": self._fallback_reason,
            **active.metrics(),
        }
