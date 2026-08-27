#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
CAMERA="${1:-CAM-01}"
TRACK_LOG="${ML_TRACK_VISUAL_LOG:-/tmp/ML_STEP4_V3_VISUAL.log}"
PYTHON="${ML_TRACK_VISUAL_PYTHON:-$ROOT/.venv/bin/python}"
SHOW_SHADOW="${ML_TRACK_VISUAL_SHOW_SHADOW:-0}"

[[ -x "$PYTHON" ]] || { echo "STEP4_VISUAL_PREFLIGHT ERROR: python missing: $PYTHON" >&2; exit 1; }

"$PYTHON" - <<'PY'
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
from PySide6.QtWidgets import QApplication  # noqa: F401
Gst.init(None)
missing = [name for name in ("nvurisrcbin", "queue", "nvvideoconvert", "capsfilter", "appsink") if Gst.ElementFactory.find(name) is None]
if missing:
    raise SystemExit("STEP4_VISUAL_PREFLIGHT ERROR: missing plugins: " + ",".join(missing))
print("STEP4_VISUAL_PREFLIGHT status=OK pyside6=1 gstreamer=1 deepstream=1", flush=True)
PY

printf '%s\n' \
  "STEP4_VISUAL_PROFILE camera=${CAMERA} main_stream_debug_decode=1 tracker_log=${TRACK_LOG} show_shadow=${SHOW_SHADOW}" \
  "STEP4_VISUAL_BOUNDARY detector=unchanged tracker=unchanged camera_service=unchanged overlay=viewer-only"

if [[ "$SHOW_SHADOW" == "1" ]]; then
  exec "$PYTHON" scripts/step4_visual_debug_viewer.py \
    --camera "$CAMERA" \
    --track-log "$TRACK_LOG"
fi

exec "$PYTHON" scripts/step4_visual_active_viewer.py \
  --camera "$CAMERA" \
  --track-log "$TRACK_LOG"
