from __future__ import annotations

import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    print(f"OLD_UI_DETECTION_PREFLIGHT=FAIL {message}", flush=True)
    raise SystemExit(2)


if importlib.util.find_spec("ultralytics") is None:
    fail("ultralytics is not installed")

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

print(
    "OLD_UI_DETECTION_PREFLIGHT=PASS "
    f"source=ui-aspect-ratio-final@865bfedf model=YOLO26m "
    f"device={torch.cuda.get_device_name(0)} cuda={torch.version.cuda} "
    "input=704x448 conf=0.06 iou=0.50 max_det=50 batch=2 "
    "roi=CAM05-verify+CAM06-augment tracker=exact-old-ui-kalman-byte",
    flush=True,
)
