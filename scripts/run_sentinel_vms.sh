#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/preflight_sentinel_ui.py

# The supplied Sentinel Qt shell owns only the desktop UI. The existing stable
# Camera V2 detector/tracker stays isolated in its DeepStream child process and
# renders the live 2x3 wall directly into the Monitoring native X11 surface.
exec python -m services.camera_v2.monitor_ui
