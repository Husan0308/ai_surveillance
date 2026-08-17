#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/preflight_camera_v2_reid.py
python scripts/preflight_sentinel_ui.py

# Qt owns only the desktop shell/X11 target. The child process keeps the stable
# YOLO26m -> NvDCF hot path and runs crop selection, ReID and Qwen on bounded
# asynchronous side paths so identity work cannot stall the camera wall.
exec python -m services.camera_v2.monitor_ui
