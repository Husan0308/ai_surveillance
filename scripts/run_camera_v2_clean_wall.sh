#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

export CAMERA_V2_RTSP_TRANSPORT="${CAMERA_V2_RTSP_TRANSPORT:-tcp}"
export CAMERA_V2_RTSP_LATENCY_MS="${CAMERA_V2_RTSP_LATENCY_MS:-100}"
export CAMERA_V2_SOURCE_FPS="${CAMERA_V2_SOURCE_FPS:-20}"
export CAMERA_V2_EXTRA_SURFACES="${CAMERA_V2_EXTRA_SURFACES:-6}"
export CAMERA_V2_LOW_LATENCY_MODE="${CAMERA_V2_LOW_LATENCY_MODE:-0}"

# Camera-only quality baseline. 1280x720 keeps a sharp 16:9 working surface;
# 1920x720 gives a 3x2 wall with exact 640x360 tiles instead of the blurry
# 480x270 tiles from the temporary 1440x540 analytics profile.
export CAMERA_V2_FRAME_WIDTH="${CAMERA_V2_FRAME_WIDTH:-1280}"
export CAMERA_V2_FRAME_HEIGHT="${CAMERA_V2_FRAME_HEIGHT:-720}"
export CAMERA_V2_TILER_COLUMNS="${CAMERA_V2_TILER_COLUMNS:-3}"
export CAMERA_V2_WALL_WIDTH="${CAMERA_V2_WALL_WIDTH:-1920}"
export CAMERA_V2_WALL_HEIGHT="${CAMERA_V2_WALL_HEIGHT:-720}"
export CAMERA_V2_MUX_TIMEOUT_US="${CAMERA_V2_MUX_TIMEOUT_US:-50000}"
export CAMERA_V2_STARTUP_STAGGER_SEC="${CAMERA_V2_STARTUP_STAGGER_SEC:-0.50}"
export CAMERA_V2_CLEAN_STALL_SEC="${CAMERA_V2_CLEAN_STALL_SEC:-12}"

# Explicitly disable every analytics subsystem for this proof run.
export QWEN_REID_ENABLED=0
export CAMERA_V2_DETECT_ACTIVE_CAMERAS=""
unset QWEN_REID_URL QWEN_REID_MODEL QWEN_REID_TIMEOUT_SEC || true

fail() { printf 'CAMERA_CLEAN_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }

for plugin in nvurisrcbin nvstreammux nvmultistreamtiler nveglglessink queue; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing GStreamer/DeepStream plugin: $plugin"
done

MAIN_PYTHON=""
for candidate in "${CAMERA_V2_MAIN_PYTHON:-}" "$ROOT/.venv/bin/python" "$(command -v python3 2>/dev/null || true)"; do
  [[ -n "$candidate" && -x "$candidate" ]] || continue
  if "$candidate" - <<'PY' >/dev/null 2>&1
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: F401
import yaml, dotenv  # noqa: F401
import services.camera_v2.clean_wall  # noqa: F401
PY
  then MAIN_PYTHON="$candidate"; break; fi
done
[[ -n "$MAIN_PYTHON" ]] || fail "no Python can import clean Camera V2 runtime"

printf '%s\n' \
  "CAMERA_CLEAN_PROFILE source=6xRTSP@${CAMERA_V2_SOURCE_FPS} mux=${CAMERA_V2_FRAME_WIDTH}x${CAMERA_V2_FRAME_HEIGHT} wall=${CAMERA_V2_WALL_WIDTH}x${CAMERA_V2_WALL_HEIGHT}" \
  "CAMERA_CLEAN_PIPELINE RTSP->nvurisrcbin/NVDEC->queue1->nvstreammux->nvmultistreamtiler->queue1->EGL" \
  "CAMERA_CLEAN_ANALYTICS detector=off tracker=off appsink=off shm=off tensorrt=off osd=off global_id=off" \
  "CAMERA_CLEAN_DISPLAY tile=640x360 expected_fps=20 sync=0 qos=0 quality=lanczos" \
  "CAMERA_CLEAN_MAIN_PYTHON executable=$MAIN_PYTHON"

restart_count=0
while true; do
  set +e
  "$MAIN_PYTHON" -u -m services.camera_v2.clean_wall
  rc=$?
  set -e
  [[ $rc -eq 75 ]] || exit "$rc"
  restart_count=$((restart_count + 1))
  delay=$restart_count; (( delay > 10 )) && delay=10
  echo "CAMERA_CLEAN_SUPERVISOR restart=$restart_count delay=${delay}s" >&2
  sleep "$delay"
done
