#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Fail before opening the UI if the ReID model is missing, corrupt, or cannot be
# loaded. On the first run this may download the CPU ReID model; later runs only
# verify/warm the already cached model.
python scripts/setup_camera_v2_reid.py
python scripts/preflight_camera_v2_reid.py
python scripts/preflight_sentinel_ui.py

# Qt owns only the desktop shell/X11 target. The child process keeps the stable
# YOLO26m -> NvDCF hot path and runs crop selection, ReID and Qwen on bounded
# asynchronous side paths so identity work cannot stall the camera wall.
exec python -m services.camera_v2.monitor_ui
