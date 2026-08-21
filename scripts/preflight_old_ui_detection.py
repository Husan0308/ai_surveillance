from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fail(message: str) -> None:
    print(f"OLD_UI_DETECTION_PREFLIGHT=FAIL {message}", flush=True)
    raise SystemExit(2)


if importlib.util.find_spec("ultralytics") is None:
    fail(
        "ultralytics is not installed; run: "
        "python -m pip install -r requirements/old_ui_detection.txt"
    )

try:
    import torch
except Exception as exc:
    fail(f"cannot import torch: {type(exc).__name__}: {exc}")
if not torch.cuda.is_available():
    fail("PyTorch CUDA unavailable")

backend = os.environ.get("CAMERA_V2_DETECT_BACKEND", "").strip().lower()
if backend not in {"stable-yolo26m", "old-ui-yolo26m", "old-ui", "ui-yolo26m"}:
    fail(f"old UI detector backend not selected: {backend!r}")

expected = {
    "CAMERA_V2_DETECT_WIDTH": "704",
    "CAMERA_V2_DETECT_HEIGHT": "448",
    "CAMERA_V2_MICRO_BATCH": "2",
    "CAMERA_V2_DETECT_CONF": "0.06",
    "CAMERA_V2_DETECT_IOU": "0.50",
    "CAMERA_V2_MAX_DET": "50",
}
for key, value in expected.items():
    if os.environ.get(key) != value:
        fail(f"{key} must be {value}, got {os.environ.get(key)!r}")

backend_source = (ROOT / "services/camera_v2/old_ui_detection_backend.py").read_text(
    encoding="utf-8"
)
for token in (
    "ui-aspect-ratio-final",
    "FULL_CONF = 0.06",
    "FULL_IOU = 0.50",
    "FULL_MAX_DET = 50",
    "MODEL_WIDTH = 704",
    "MODEL_HEIGHT = 448",
    '"CAM-05"',
    '"mode": "verify"',
    '"CAM-06"',
    '"mode": "augment"',
    "_deduplicate_boxes",
    "OldUIBoxManager",
    "MAX_SUBMIT_AGE_SEC = 0.300",
    "MAX_RESULT_AGE_SEC = 0.900",
    "OLD_UI_OVERLAY_READY",
):
    if token not in backend_source:
        fail(f"backend contract missing: {token}")

tracker_source = (ROOT / "services/camera_v2/old_ui_visual_tracker.py").read_text(
    encoding="utf-8"
)
for token in (
    "class VisualTracker",
    "hold_ms=800",
    "memory_ms=3000",
    "prediction_ms=420",
    "match_iou=0.12",
    "reacquire_distance=0.85",
    "max_prediction_shift_boxes=0.55",
):
    if token not in tracker_source:
        fail(f"old UI tracker contract missing: {token}")

selector_source = (ROOT / "services/camera_v2/backend_selector.py").read_text(encoding="utf-8")
for token in (
    "legacy_unknown_overlay",
    "install_unknown_overlay",
):
    if token not in selector_source:
        fail(f"legacy Unknown selector contract missing: {token}")

overlay_source = (ROOT / "services/camera_v2/legacy_unknown_overlay.py").read_text(encoding="utf-8")
native_source = (ROOT / "services/camera_v2/native_unknown_overlay.c").read_text(encoding="utf-8")
for token in (
    "LEGACY_UNKNOWN_OVERLAY_READY",
    "Unknown_C{camera}_{track}",
    "_visible_tracks",
    "display-text",
    "post-tiler-pre-osd",
):
    if token not in overlay_source:
        fail(f"legacy Unknown overlay contract missing: {token}")
for token in (
    "camera_v2_add_unknown_tracks",
    "Unknown_C%u_%02",
    "0.965f",
    "0.725f",
    "0.294f",
    "Monospace",
):
    if token not in native_source:
        fail(f"legacy Unknown native style contract missing: {token}")

# Compile the small native presentation helper before opening six cameras so a
# toolchain/header issue fails early and cannot masquerade as a detector problem.
try:
    from services.camera_v2.legacy_unknown_overlay import _ensure_library

    overlay_library = _ensure_library()
except Exception as exc:
    fail(f"legacy Unknown native overlay unavailable: {type(exc).__name__}: {exc}")

print(
    "OLD_UI_DETECTION_PREFLIGHT=PASS "
    f"source=core-v1-clean/ui-aspect-ratio-final model=YOLO26m "
    f"device={torch.cuda.get_device_name(0)} cuda={torch.version.cuda} "
    "input=704x448 conf=0.06 iou=0.50 max_det=50 batch=2 "
    "roi=CAM05-verify+CAM06-augment tracker=exact-old-ui-kalman-byte "
    f"overlay=gpu-v2-yellow-Unknown native={overlay_library}",
    flush=True,
)
