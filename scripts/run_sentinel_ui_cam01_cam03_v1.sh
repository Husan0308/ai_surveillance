#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export V11_UI_STAGE_CAMERAS="CAM-01,CAM-02,CAM-03"
exec bash scripts/run_sentinel_ui_v1.sh
