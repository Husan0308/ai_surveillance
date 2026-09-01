#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
APP_PY="${V11_UI_PYTHON:-$HOME/ai_surveillance/.venv/bin/python}"
[[ -x "$APP_PY" ]] || APP_PY="$(command -v python3)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SENTINEL_LIVE_PREVIEW_CAMERAS="CAM-01,CAM-02"
export V11_UI_PREVIEW_PATH_CAM01="${V11_UI_PREVIEW_PATH_CAM01:-/dev/shm/v11_ui_preview_cam01_v1.bin}"
export V11_UI_PREVIEW_PATH_CAM02="${V11_UI_PREVIEW_PATH_CAM02:-/dev/shm/v11_ui_preview_cam02_v1.bin}"
CAMERA_STATE="$($APP_PY - <<'PY'
import PySide6
import services.frontend.sentinel_v1.ui
from services.frontend.sentinel_v1.data import CAMERAS
ids = [camera.id for camera in CAMERAS]
if ids != ["CAM-01", "CAM-02"]:
    raise SystemExit(f"unexpected_camera_cards={ids}")
print(",".join(ids))
PY
)" || { echo 'V11_SENTINEL_UI_CAM01_CAM02_PREFLIGHT RESULT=FAIL reason=PySide6_ui_or_camera_state' >&2; exit 1; }
echo "V11_SENTINEL_UI_CAM01_CAM02_PREFLIGHT RESULT=PASS cards=$CAMERA_STATE demo_cameras=0 cam01=$V11_UI_PREVIEW_PATH_CAM01 cam02=$V11_UI_PREVIEW_PATH_CAM02"
exec "$APP_PY" -m services.frontend.sentinel_v1.main
