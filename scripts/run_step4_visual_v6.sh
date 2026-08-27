#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
CAMERA="${1:-CAM-01}"
TRACK_LOG="${ML_TRACK_VISUAL_LOG:-/tmp/ML_STEP4_V6_VISUAL.log}"
PYTHON="${ML_TRACK_VISUAL_PYTHON:-$ROOT/.venv/bin/python}"
SHOW_SHADOW="${ML_TRACK_VISUAL_SHOW_SHADOW:-0}"
WIDTH="${ML_TRACK_VISUAL_WIDTH:-1280}"
HEIGHT="${ML_TRACK_VISUAL_HEIGHT:-720}"
LATENCY_MS="${ML_TRACK_VISUAL_LATENCY_MS:-120}"

[[ -x "$PYTHON" ]] || { echo "STEP4_V6_VISUAL_PREFLIGHT ERROR: python missing: $PYTHON" >&2; exit 1; }

"$PYTHON" - <<'PY'
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
from PySide6.QtWidgets import QApplication  # noqa: F401
Gst.init(None)
missing = [name for name in ("nvurisrcbin", "queue", "nvvideoconvert", "capsfilter", "appsink") if Gst.ElementFactory.find(name) is None]
if missing:
    raise SystemExit("STEP4_V6_VISUAL_PREFLIGHT ERROR: missing plugins: " + ",".join(missing))
print("STEP4_V6_VISUAL_PREFLIGHT status=OK pyside6=1 gstreamer=1 deepstream=1", flush=True)
PY

printf '%s\n' \
  "STEP4_V6_VISUAL_PROFILE camera=${CAMERA} main=${WIDTH}x${HEIGHT} latency=${LATENCY_MS}ms tracker_log=${TRACK_LOG} show_shadow=${SHOW_SHADOW}" \
  "STEP4_V6_VISUAL_POLICY body_envelope=1 raw_detector_inside_render=1 metadata_lag_comp=1 predict_max=0.34s size_prediction=0 stale_box_max=1.10s" \
  "STEP4_V6_VISUAL_BOUNDARY detector=substream tracker=cpu-v6 camera_service=independent overlay=viewer-only"

COMMON=(
  --camera "$CAMERA"
  --track-log "$TRACK_LOG"
  --width "$WIDTH"
  --height "$HEIGHT"
  --latency-ms "$LATENCY_MS"
)

if [[ "$SHOW_SHADOW" == "1" ]]; then
  exec "$PYTHON" scripts/step4_visual_debug_viewer_v6.py "${COMMON[@]}"
fi

exec "$PYTHON" scripts/step4_visual_active_viewer_v6.py "${COMMON[@]}"
