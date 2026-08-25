from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

os.environ["CAMERA_CONFIG"] = str(
    ROOT / "artifacts/reid/cameras_room1.yaml"
)

os.environ["CAMERA_V2_TRACKER_CONFIG"] = str(
    ROOT / "config/reid/config_tracker_room.yml"
)

# Two-camera lightweight smoke profile.
os.environ["CAMERA_V2_FRAME_WIDTH"] = "1280"
os.environ["CAMERA_V2_FRAME_HEIGHT"] = "720"
os.environ["CAMERA_V2_TILER_COLUMNS"] = "2"
os.environ["CAMERA_V2_RTSP_TRANSPORT"] = "tcp"
os.environ["CAMERA_V2_RTSP_LATENCY_MS"] = "150"

from services.camera_v2.person_tracking import CameraPersonTrackingV2
from services.camera_v2.main import CameraWallV2

wall = CameraPersonTrackingV2()

print(
    "REID_SMOKE_START "
    f"cameras={[c.camera_id for c in wall.cameras]} "
    f"tracker_config={wall.tracker_config} "
    f"tracker_lib={wall.tracker_lib}",
    flush=True,
)

# Important:
# bypass CameraDetectionV2.run(), so YOLO/RF-DETR worker is NOT started.
# We only start RTSP -> nvstreammux -> NvDeepSORT -> display.
raise SystemExit(CameraWallV2.run(wall))
