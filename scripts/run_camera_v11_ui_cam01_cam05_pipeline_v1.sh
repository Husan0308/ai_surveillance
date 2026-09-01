#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export V11_UI_STAGE_CAMERAS="CAM-01,CAM-02,CAM-03,CAM-04,CAM-05"
exec bash scripts/run_camera_v11_ui_pipeline_v1.sh
