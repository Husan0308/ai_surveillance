from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import sys


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
except Exception as exc:
    fail(f"cannot import RFDETRSmall: {type(exc).__name__}: {exc}")

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

# RF-DETR-S uses patch_size=16 and the published small model's window layout
# requires dimensions divisible by the effective 32-pixel grid. The selected
# 736x416 shape preserves the CCTV stream aspect and satisfies that contract.
if width % 32 or height % 32:
    fail(f"RF-DETR-S detector shape must be divisible by 32, got {width}x{height}")

micro_batch = int(os.environ.get("CAMERA_V2_MICRO_BATCH", "1"))
if micro_batch != 1:
    fail(f"production bring-up requires micro-batch=1 on the target GPU, got {micro_batch}")

threshold = float(os.environ.get("CAMERA_V2_DETECT_CONF", "0.18"))
if not 0.05 <= threshold <= 0.60:
    fail(f"unexpected RF-DETR threshold {threshold}")

print(
    "RFDETR_PREFLIGHT=PASS "
    f"version={version} model=RF-DETR-S device={torch.cuda.get_device_name(0)} "
    f"cuda={torch.version.cuda} shape={width}x{height} batch={micro_batch} threshold={threshold:.2f}",
    flush=True,
)
