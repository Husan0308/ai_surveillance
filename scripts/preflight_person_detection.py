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
        gpu_name = torch.cuda.get_device_name(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
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
        f"PERSON_DETECT_GPU name={gpu_name} "
        f"free_mb={free_bytes / 1024 / 1024:.0f} total_mb={total_bytes / 1024 / 1024:.0f}",
        flush=True,
    )
    print(
        f"PERSON_DETECT_LIB torch={torch.__version__} cuda={torch.version.cuda} "
        f"ultralytics={ultralytics.__version__}",
        flush=True,
    )
    print(f"PERSON_DETECT_MODEL {model_state}", flush=True)
    print("PERSON_DETECT_PREFLIGHT=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
