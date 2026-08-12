import os
import site
import ctypes
import cv2
import torch
from shared.logging import get_logger

log=get_logger(__name__)

_GPU_SETUP_DONE = False
_TORCH_CUDA_USABLE = None

def torch_cuda_usable():
    global _TORCH_CUDA_USABLE
    if _TORCH_CUDA_USABLE is not None:
        return _TORCH_CUDA_USABLE
    """Check that this PyTorch build contains kernels for the installed GPU."""
    try:
        if not torch.cuda.is_available():
            return False
        probe = torch.ones(1, device="cuda")
        _ = float(probe.sum().item())
        _TORCH_CUDA_USABLE = True
        return _TORCH_CUDA_USABLE
    except Exception as exc:
        log.warning("CUDA check failed: %s; AI uses CPU",exc)
        _TORCH_CUDA_USABLE = False
        return _TORCH_CUDA_USABLE


def resolve_torch_device(requested="auto"):
    if str(requested or "auto").lower() == "cpu":
        return "cpu"
    return "cuda" if torch_cuda_usable() else "cpu"


def setup_gpu_environment():
    """
    1. Targeted preloading of required CUDA/cuDNN libraries into RTLD_GLOBAL
       so ONNX Runtime (InsightFace) uses CUDAExecutionProvider on GPU cleanly.
    2. Limit CPU thread overload from OpenCV / PyTorch.
    """
    global _GPU_SETUP_DONE
    if _GPU_SETUP_DONE:
        return
    _GPU_SETUP_DONE = True

    # Limit OpenCV, PyTorch & BLAS CPU worker thread hogging
    try:
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["MKL_NUM_THREADS"] = "2"
        os.environ["OPENBLAS_NUM_THREADS"] = "2"
        cv2.setNumThreads(2)
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = True
    except (RuntimeError,AttributeError) as exc:
        log.warning("GPU runtime tuning unavailable: %s",exc)

    # Targeted preloading for ONNX Runtime & PyTorch
    try:
        sp_list = site.getsitepackages()
        for sp in sp_list:
            nvidia_dir = os.path.join(sp, 'nvidia')
            if not os.path.exists(nvidia_dir):
                continue

            libs_to_load = [
                os.path.join(nvidia_dir, 'cuda_runtime', 'lib', 'libcudart.so.12'),
                os.path.join(nvidia_dir, 'cublas', 'lib', 'libcublasLt.so.12'),
                os.path.join(nvidia_dir, 'cublas', 'lib', 'libcublas.so.12'),
                os.path.join(nvidia_dir, 'cudnn', 'lib', 'libcudnn.so.9'),
                os.path.join(nvidia_dir, 'cufft', 'lib', 'libcufft.so.11'),
                os.path.join(nvidia_dir, 'cu13', 'lib', 'libcurand.so.10'),
            ]

            for p in libs_to_load:
                if os.path.exists(p):
                    try:
                        ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                    except OSError as exc:
                        log.debug("CUDA preload skipped %s: %s",p,exc)

        log.info("Targeted CUDA and cuDNN libraries initialized")
    except Exception as e:
        log.warning("GPU setup notice: %s",e)
