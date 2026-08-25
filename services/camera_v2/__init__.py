"""Camera V2 audited production defaults.

The active hot path is six-camera DeepStream ingest/display plus CAM-01 YOLO26
TensorRT detector metadata feeding per-frame NvDCF. ReID/global identity remains a
later asynchronous layer and must not own or block the local camera/tracker path.
"""

from __future__ import annotations

import os

# Canonical TensorRT engine geometry. Launchers may override before import, but a
# direct `python -m services.camera_v2` must resolve the same shape.
os.environ.setdefault("CAMERA_V2_DETECT_WIDTH", "672")
os.environ.setdefault("CAMERA_V2_DETECT_HEIGHT", "384")
os.environ.setdefault("CAMERA_V2_MICRO_BATCH", "1")
