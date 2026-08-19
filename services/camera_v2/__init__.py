"""Sentinel Camera V2 production runtime.

Active path: RTSP/NVDEC -> fresh YOLO person detection -> per-camera NvDCF ->
native metadata/heatmap -> native Sentinel wall. Identity/ReID is not part of the
current production pipeline.
"""

from __future__ import annotations

import os

# Canonical detector geometry enforced by the production preflight.
os.environ.setdefault("CAMERA_V2_DETECT_WIDTH", "736")
os.environ.setdefault("CAMERA_V2_DETECT_HEIGHT", "416")
