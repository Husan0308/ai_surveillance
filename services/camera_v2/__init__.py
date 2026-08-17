"""Camera V2 production defaults.

The active production path is intentionally limited to camera ingest, YOLO person
metadata, per-camera NvDCF tracking and camera-space heatmap. Cross-camera ReID,
Qwen/KPR verification and room calibration are not part of the live runtime.
"""

from __future__ import annotations

import os

# Stride-32 detector geometry; keeps configured and actual YOLO input identical.
os.environ.setdefault("CAMERA_V2_DETECT_WIDTH", "704")
os.environ.setdefault("CAMERA_V2_DETECT_HEIGHT", "384")
