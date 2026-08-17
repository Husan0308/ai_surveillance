"""Camera V2 production defaults.

Keep the sidecar capture resolution close to the 16:9 CCTV source while making
both dimensions valid for YOLO's 32-pixel stride. 704x396 caused Ultralytics to
silently change every inference request to 704x416, producing repeated warnings
and making the configured geometry disagree with the actual model geometry.
704x384 removes that hidden resize step while retaining the horizontal detail
needed for distant people on the six-camera GTX 1050 Ti deployment.
"""

from __future__ import annotations

import os

os.environ.setdefault("CAMERA_V2_DETECT_WIDTH", "704")
os.environ.setdefault("CAMERA_V2_DETECT_HEIGHT", "384")

# Physical room topology for the current cameras.yaml order:
# CAM-01/CAM-02, CAM-03/CAM-06, CAM-05/CAM-04.
os.environ.setdefault(
    "CAMERA_V2_REID_ROOM_MAP",
    "0:0,1:0,2:1,5:1,4:2,3:2",
)
