#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/preflight_sentinel_ui.py

# Native metadata/heatmap bridge auto-rebuilds when its C sources are newer.
# Running the full UI keeps camera/decode/detection/tracking in the isolated
# DeepStream child process while Qt only owns controls and the native X11 surface.
exec python -m services.camera_v2.monitor_ui
