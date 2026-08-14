from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import tempfile
import threading
import urllib.request
import zipfile

from .face_service_safe import SafeFaceRecognitionService


class CudaFaceRecognitionService(SafeFaceRecognitionService):
    """CUDA-first Face service with bounded VRAM use and CPU fallback.

    Face inference remains a low-rate side-path. The detector/tracker hot path is
    unchanged. CUDA is verified from the actual InsightFace ONNX sessions so a
    silent CPU fallback is visible in /health instead of being mistaken for GPU.
    All ambiguity and enrollment-consistency guards from SafeFaceRecognitionService
    remain active.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requested_provider = str(
            self.config.get("provider", "CUDAExecutionProvider")
        )
        self.device_id = max(0, int(self.config.get("device_id", 0)))
        self.gpu_mem_limit_mb = max(128, int(self.config.get("gpu_mem_limit_mb", 768)))
        self.allow_cpu_fallback = bool(self.config.get("allow_cpu_fallback", True))
        self.cuda_cudnn_search = str(
            self.config.get("cuda_cudnn_conv_algo_search", "HEURISTIC")
        ).upper()
        self.model_url = str(self.config.get("model_url", "")).strip()
        self.model_sha256 = str(self.config.get("model_sha256", "")).strip().lower()
        self.download_if_missing = bool(self.config.get("download_if_missing", True))
        self._actual_provider = "not_loaded"
        self._available_ort_providers: list[str] = []
        self._session_providers: dict[str, list[str]] = {}
        self._cuda_verified = False
        self._used_cpu_fallback = False
        self._model_pack_dir = ""
        self._model_pack_downloaded = False

    def start(self):
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="core-v1-face-cuda",
            daemon=True,
        )
        self._thread.start()

    def _pack_dir(self) -> Path:
        # InsightFace FaceAnalysis(root=R, name=N) resolves R/models/N/*.onnx.
        return Path(self.model_root) / "models" / self.model_pack

    @staticmethod
    def _pack_has_required_models(path: Path) -> bool:
        if not path.is_dir():
            return False
        names = [item.name.lower() for item in path.glob("*.onnx") if item.is_file()]
        if not names:
            return False
        has_detection = any(
            name.startswith("det_") or "scrfd" in name or "retina" in name
            for name in names
        )
        has_recognition = any(
            "w600k" in name or "arcface" in name or "recognition" in name
            for name in names
        )
        return has_detection and has_recognition

    def _normalize_extracted_pack(self, pack_dir: Path):
        if self._pack_has_required_models(pack_dir):
            return
        nested = [child for child in pack_dir.iterdir() if child.is_dir()] if pack_dir.exists() else []
        for child in nested:
            if not self._pack_has_required_models(child):
                continue
            for item in child.iterdir():
                destination = pack_dir / item.name
                if destination.exists():
                    if destination.is_dir():
                        shutil.rmtree(destination)
                    else:
                        destination.unlink()
                shutil.move(str(item), str(destination))
            try:
                child.rmdir()
            except OSError:
                pass
            return

    def _ensure_model_pack(self):
        pack_dir = self._pack_dir()
        self._model_pack_dir = str(pack_dir)
        if self._pack_has_required_models(pack_dir):
            return

        if not self.download_if_missing:
            raise RuntimeError(
                f"InsightFace model pack is missing or incomplete: {pack_dir}"
            )
        if not self.model_url:
            raise RuntimeError(
                f"InsightFace model pack {self.model_pack!r} is not auto-downloadable; model_url is required"
            )
        if not self.model_sha256 or len(self.model_sha256) != 64:
            raise RuntimeError("a pinned 64-character model_sha256 is required")

        pack_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="insightface-pack-") as temp_dir:
            archive = Path(temp_dir) / f"{self.model_pack}.zip"
            request = urllib.request.Request(
                self.model_url,
                headers={"User-Agent": "ai-surveillance/face-model-loader"},
            )
            digest = hashlib.sha256()
            try:
                with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        output.write(chunk)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to download InsightFace {self.model_pack} from official release: {exc}"
                ) from exc

            actual_sha256 = digest.hexdigest().lower()
            if actual_sha256 != self.model_sha256:
                raise RuntimeError(
                    f"InsightFace {self.model_pack} SHA256 mismatch: expected "
                    f"{self.model_sha256}, got {actual_sha256}"
                )

            extract_root = Path(temp_dir) / "extract"
            extract_root.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(archive) as bundle:
                    bundle.extractall(extract_root)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to extract InsightFace {self.model_pack}: {exc}"
                ) from exc

            candidates = [extract_root]
            candidates.extend(child for child in extract_root.iterdir() if child.is_dir())
            source = next(
                (candidate for candidate in candidates if self._pack_has_required_models(candidate)),
                None,
            )
            if source is None:
                onnx_files = sorted(str(path.relative_to(extract_root)) for path in extract_root.rglob("*.onnx"))
                raise RuntimeError(
                    f"official {self.model_pack} archive has no usable detection+recognition pack; "
                    f"onnx={onnx_files}"
                )

            staged = pack_dir.with_name(pack_dir.name + ".staging")
            shutil.rmtree(staged, ignore_errors=True)
            staged.mkdir(parents=True, exist_ok=True)
            for item in source.iterdir():
                destination = staged / item.name
                if item.is_dir():
                    shutil.copytree(item, destination)
                else:
                    shutil.copy2(item, destination)

            if not self._pack_has_required_models(staged):
                raise RuntimeError(
                    f"staged InsightFace model pack is incomplete: {staged}"
                )

            shutil.rmtree(pack_dir, ignore_errors=True)
            staged.replace(pack_dir)
            self._model_pack_downloaded = True

    def _load_engine(self):
        try:
            self._ensure_model_pack()

            # ONNX Runtime documents importing PyTorch before creating CUDA EP
            # sessions as a supported way to preload matching CUDA/cuDNN libs.
            import torch  # noqa: F401
            import onnxruntime as ort

            if hasattr(ort, "preload_dlls"):
                try:
                    ort.preload_dlls()
                except Exception:
                    # Linux/system CUDA can already be loaded by torch; failure
                    # here is not fatal as long as CUDAExecutionProvider appears.
                    pass

            available = [str(value) for value in ort.get_available_providers()]
            wants_cuda = self.requested_provider == "CUDAExecutionProvider"
            has_cuda = "CUDAExecutionProvider" in available
            if wants_cuda and not has_cuda and not self.allow_cpu_fallback:
                raise RuntimeError(
                    "CUDAExecutionProvider is unavailable; install a compatible onnxruntime-gpu build"
                )

            from insightface.app import FaceAnalysis

            cuda_options = {
                "device_id": str(self.device_id),
                "gpu_mem_limit": str(self.gpu_mem_limit_mb * 1024 * 1024),
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": self.cuda_cudnn_search,
                "do_copy_in_default_stream": "1",
            }
            if wants_cuda and has_cuda:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                provider_options = [cuda_options, {}]
                ctx_id = self.device_id
            else:
                providers = ["CPUExecutionProvider"]
                provider_options = [{}]
                ctx_id = -1

            app = FaceAnalysis(
                name=self.model_pack,
                root=self.model_root,
                allowed_modules=["detection", "recognition"],
                providers=providers,
                provider_options=provider_options,
            )
            if "detection" not in (getattr(app, "models", {}) or {}):
                raise RuntimeError(
                    f"InsightFace did not load a detection model from {self._pack_dir()}"
                )
            if "recognition" not in (getattr(app, "models", {}) or {}):
                raise RuntimeError(
                    f"InsightFace did not load a recognition model from {self._pack_dir()}"
                )
            app.prepare(
                ctx_id=ctx_id,
                det_thresh=self.det_thresh,
                det_size=(self.det_size, self.det_size),
            )

            session_providers: dict[str, list[str]] = {}
            for task_name, model in (getattr(app, "models", {}) or {}).items():
                session = getattr(model, "session", None)
                get_providers = getattr(session, "get_providers", None)
                if callable(get_providers):
                    session_providers[str(task_name)] = [
                        str(value) for value in get_providers()
                    ]

            sessions = list(session_providers.values())
            cuda_verified = bool(sessions) and all(
                "CUDAExecutionProvider" in providers_for_model
                for providers_for_model in sessions
            )
            if cuda_verified:
                actual_provider = "CUDAExecutionProvider"
            elif any(
                "CUDAExecutionProvider" in providers_for_model
                for providers_for_model in sessions
            ):
                actual_provider = "mixed-cuda-cpu"
            else:
                actual_provider = "CPUExecutionProvider"

            used_cpu_fallback = wants_cuda and not cuda_verified
            if used_cpu_fallback and not self.allow_cpu_fallback:
                raise RuntimeError(
                    f"InsightFace sessions did not bind fully to CUDA: {session_providers}"
                )

            with self._lock:
                self._app = app
                self._ready = True
                self._last_error = ""
                self._available_ort_providers = available
                self._session_providers = session_providers
                self._cuda_verified = cuda_verified
                self._actual_provider = actual_provider
                self._used_cpu_fallback = used_cpu_fallback
        except Exception as exc:
            with self._lock:
                self._app = None
                self._ready = False
                detail = str(exc).strip() or repr(exc)
                if isinstance(exc, AssertionError):
                    detail = (
                        f"InsightFace assertion while loading {self.model_pack}; "
                        f"verify detection+recognition ONNX files under {self._pack_dir()}"
                    )
                self._last_error = f"{type(exc).__name__}: {detail}"
                self._actual_provider = "load_failed"
                self._cuda_verified = False

    def metrics(self) -> dict:
        payload = super().metrics()
        with self._lock:
            payload.update(
                {
                    "provider": self._actual_provider,
                    "requested_provider": self.requested_provider,
                    "cuda_verified": self._cuda_verified,
                    "cpu_fallback": self._used_cpu_fallback,
                    "allow_cpu_fallback": self.allow_cpu_fallback,
                    "device_id": self.device_id,
                    "gpu_mem_limit_mb": self.gpu_mem_limit_mb,
                    "cuda_cudnn_conv_algo_search": self.cuda_cudnn_search,
                    "available_ort_providers": list(self._available_ort_providers),
                    "session_providers": dict(self._session_providers),
                    "sidepath": "low-rate-cuda-face",
                    "model_pack_dir": self._model_pack_dir or str(self._pack_dir()),
                    "model_pack_ready": self._pack_has_required_models(self._pack_dir()),
                    "model_pack_downloaded": self._model_pack_downloaded,
                    "model_url": self.model_url,
                }
            )
        return payload
