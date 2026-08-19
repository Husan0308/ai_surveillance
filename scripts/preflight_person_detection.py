from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.ml_service.app.config import load_settings


def fail(message: str) -> int:
    print(f"PERSON_DETECT_PREFLIGHT=FAIL {message}", flush=True)
    return 1


def _device_index(device: str) -> int:
    text = str(device).strip().lower()
    if text == "cuda":
        return 0
    if text.startswith("cuda:"):
        return int(text.split(":", 1)[1])
    raise ValueError(f"person detector requires CUDA device, got {device!r}")


def main() -> int:
    settings = load_settings()
    cfg = settings.detection
    if not cfg.enabled:
        print("PERSON_DETECT_PREFLIGHT=SKIP detection disabled", flush=True)
        return 0

    try:
        import torch
    except Exception as exc:
        return fail(f"PyTorch import failed: {type(exc).__name__}: {exc}")

    try:
        import ultralytics
        from ultralytics import YOLO  # noqa: F401
    except Exception as exc:
        return fail(f"Ultralytics import failed: {type(exc).__name__}: {exc}")

    if not torch.cuda.is_available():
        return fail("torch.cuda.is_available() is false")

    try:
        device_index = _device_index(cfg.device)
        gpu_name = torch.cuda.get_device_name(device_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        capability = torch.cuda.get_device_capability(device_index)
        required_arch = f"sm_{capability[0]}{capability[1]}"
        compiled_arches = tuple(torch.cuda.get_arch_list())
    except Exception as exc:
        return fail(f"CUDA device query failed: {type(exc).__name__}: {exc}")

    model_path = Path(cfg.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    model_state = str(model_path) if model_path.is_file() else f"Ultralytics model spec: {cfg.model}"

    print(
        "PERSON_DETECT_CONFIG "
        f"model={cfg.model} device={cfg.device} batch={cfg.batch_size} "
        f"imgsz={cfg.width}x{cfg.height} target_fps_per_camera={cfg.target_fps_per_camera} "
        f"conf={cfg.confidence} half={cfg.half}",
        flush=True,
    )
    print(
        f"PERSON_DETECT_GPU name={gpu_name} capability={required_arch} "
        f"free_mb={free_bytes / 1024 / 1024:.0f} total_mb={total_bytes / 1024 / 1024:.0f}",
        flush=True,
    )
    print(
        f"PERSON_DETECT_LIB torch={torch.__version__} cuda={torch.version.cuda} "
        f"ultralytics={ultralytics.__version__} arches={','.join(compiled_arches)}",
        flush=True,
    )
    print(f"PERSON_DETECT_MODEL {model_state}", flush=True)

    if compiled_arches and required_arch not in compiled_arches:
        return fail(
            f"GPU requires {required_arch}, but installed PyTorch binary supports "
            f"{','.join(compiled_arches)}"
        )

    # Run a tiny real CUDA kernel. cuda.is_available() alone only proves that a
    # CUDA runtime/device can be discovered; it does not prove this wheel can
    # execute kernels on this GPU architecture.
    try:
        probe = torch.ones((32, 32), device=cfg.device)
        probe = probe @ probe
        value = float(probe.sum().item())
        torch.cuda.synchronize(device_index)
        del probe
    except BaseException as exc:
        return fail(f"CUDA kernel probe failed: {type(exc).__name__}: {exc}")

    print(f"PERSON_DETECT_CUDA_KERNEL=PASS value={value:.1f}", flush=True)
    print("PERSON_DETECT_PREFLIGHT=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
