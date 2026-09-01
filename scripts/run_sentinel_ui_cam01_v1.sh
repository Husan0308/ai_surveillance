#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
APP_PY="${V11_UI_PYTHON:-$HOME/ai_surveillance/.venv/bin/python}"
[[ -x "$APP_PY" ]] || APP_PY="$(command -v python3)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SENTINEL_LIVE_PREVIEW_CAMERA="CAM-01"
export V11_UI_PREVIEW_PATH="${V11_UI_PREVIEW_PATH:-/dev/shm/v11_ui_preview_cam01_v1.bin}"
"$APP_PY" -c 'import PySide6; import services.frontend.sentinel_v1.ui' || { echo 'V11_SENTINEL_UI_PREFLIGHT RESULT=FAIL reason=PySide6_or_ui_import_missing' >&2; exit 1; }
echo "V11_SENTINEL_UI_PREFLIGHT RESULT=PASS camera=CAM-01 path=$V11_UI_PREVIEW_PATH"
exec "$APP_PY" -m services.frontend.sentinel_v1.main
