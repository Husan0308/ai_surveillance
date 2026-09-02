#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export V11_UI_STAGE_CAMERAS="CAM-01,CAM-02,CAM-03,CAM-04,CAM-05,CAM-06"
export SENTINEL_MONITORING_REALTIME=1
export SENTINEL_MONITORING_WS_URL="${SENTINEL_MONITORING_WS_URL:-ws://127.0.0.1:8000/ws/v1/monitoring}"
exec bash scripts/run_sentinel_ui_cam01_cam06_v1.sh
