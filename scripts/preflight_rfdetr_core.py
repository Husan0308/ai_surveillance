from __future__ import annotations

import importlib.metadata
import importlib.util
import os


def fail(message: str) -> None:
    print(f"RFDETR_PREFLIGHT=FAIL {message}", flush=True)
    raise SystemExit(2)


if importlib.util.find_spec("rfdetr") is None:
    fail("missing package; install with: python -m pip install 'rfdetr>=1.9.3,<2'")

try:
    version = importlib.metadata.version("rfdetr")
except Exception:
    version = "unknown"

try:
    from rfdetr import RFDETRSmall  # noqa: F401
    from rfdetr.config import RFDETRSmallConfig
except Exception as exc:
    fail(f"cannot import RF-DETR-S: {type(exc).__name__}: {exc}")

try:
    import torch
except Exception as exc:
    fail(f"cannot import torch: {type(exc).__name__}: {exc}")

if not torch.cuda.is_available():
    fail("PyTorch CUDA unavailable")

width = int(os.environ.get("CAMERA_V2_DETECT_WIDTH", "736"))
height = int(os.environ.get("CAMERA_V2_DETECT_HEIGHT", "416"))
if width <= 0 or height <= 0:
    fail(f"invalid detector shape {width}x{height}")

try:
    config = RFDETRSmallConfig()
    patch_size = int(config.patch_size)
    num_windows = int(config.num_windows)
    block_size = patch_size * num_windows
except Exception as exc:
    fail(f"cannot resolve RF-DETR-S shape contract: {type(exc).__name__}: {exc}")

if block_size <= 0 or width % block_size or height % block_size:
    fail(
        f"RF-DETR-S detector shape must be divisible by {block_size} "
        f"(patch={patch_size} windows={num_windows}), got {width}x{height}"
    )

micro_batch = int(os.environ.get("CAMERA_V2_MICRO_BATCH", "1"))
if micro_batch != 1:
    fail(f"production bring-up requires micro-batch=1 on the target GPU, got {micro_batch}")

threshold = float(os.environ.get("CAMERA_V2_DETECT_CONF", "0.18"))
if not 0.05 <= threshold <= 0.60:
    fail(f"unexpected RF-DETR threshold {threshold}")

print(
    "RFDETR_PREFLIGHT=PASS "
    f"version={version} model=RF-DETR-S device={torch.cuda.get_device_name(0)} "
    f"cuda={torch.version.cuda} shape={width}x{height} block={block_size} "
    f"batch={micro_batch} threshold={threshold:.2f}",
    flush=True,
)
