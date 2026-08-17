"""Camera V2 production defaults.

The hot path remains camera ingest -> YOLO person metadata -> per-camera NvDCF.
Cross-camera Global ReID is layered on top as a bounded asynchronous side path:
quality/diversity crop bank -> CPU ReID gallery -> room/time constraints -> optional
Qwen visual verification -> reversible Global ID state machine. ReID/Qwen never own
or block the local tracker/display path.
"""

from __future__ import annotations

import os

# Stride-32 detector geometry; keeps configured and actual YOLO input identical.
os.environ.setdefault("CAMERA_V2_DETECT_WIDTH", "704")
os.environ.setdefault("CAMERA_V2_DETECT_HEIGHT", "384")
