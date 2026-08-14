from __future__ import annotations

import threading

from .face_service import FaceRecognitionService


class CudaFaceRecognitionService(FaceRecognitionService):
    """CUDA-first Face service with bounded VRAM use and CPU fallback.

    Face inference remains a low-rate side-path. The detector/tracker hot path is
    unchanged. CUDA is verified from the actual InsightFace ONNX sessions so a
    silent CPU fallback is visible in /health instead of being mistaken for GPU.
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
        self._actual_provider = "not_loaded"
        self._available_ort_providers: list[str] = []
        self._session_providers: dict[str, list[str]] = {}
        self._cuda_verified = False
        self._used_cpu_fallback = False

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

    def _load_engine(self):
        try:
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
                self._last_error = f"{type(exc).__name__}: {exc}"
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
                }
            )
        return payload
