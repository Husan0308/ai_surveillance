#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
LOCK_FILE="/tmp/ai_surveillance_camera_v2_gpu.lock"
fail() { printf 'CAMERA_V89_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }
command -v flock >/dev/null 2>&1 || fail "flock missing"
exec 9>"$LOCK_FILE"
flock -n 9 || fail "another Camera V2 owner holds $LOCK_FILE; stop it first"

# Exact V8.5 analytics baseline. Only visual compositor/rendering is removed.
export CAMERA_V2_RTSP_TRANSPORT="${CAMERA_V2_RTSP_TRANSPORT:-tcp}"
export CAMERA_V2_RTSP_LATENCY_MS="${CAMERA_V2_RTSP_LATENCY_MS:-60}"
export CAMERA_V2_SOURCE_FPS="${CAMERA_V2_SOURCE_FPS:-20}"
export CAMERA_V2_EXTRA_SURFACES="${CAMERA_V2_EXTRA_SURFACES:-6}"
export CAMERA_V2_DISPLAY_WIDTH="${CAMERA_V2_DISPLAY_WIDTH:-1280}"
export CAMERA_V2_DISPLAY_HEIGHT="${CAMERA_V2_DISPLAY_HEIGHT:-720}"
export CAMERA_V2_WALL_WIDTH="${CAMERA_V2_WALL_WIDTH:-1920}"
export CAMERA_V2_WALL_HEIGHT="${CAMERA_V2_WALL_HEIGHT:-720}"

export CAMERA_V2_TRACK_WIDTH="${CAMERA_V2_TRACK_WIDTH:-512}"
export CAMERA_V2_TRACK_HEIGHT="${CAMERA_V2_TRACK_HEIGHT:-288}"
export CAMERA_V2_TRACK_FPS="${CAMERA_V2_TRACK_FPS:-8}"
export CAMERA_V2_MIN_DISPLAY_TRACK_CONF="${CAMERA_V2_MIN_DISPLAY_TRACK_CONF:-0.28}"
export CAMERA_V2_DISPLAY_TRACK_MAX_AGE_MS="${CAMERA_V2_DISPLAY_TRACK_MAX_AGE_MS:-420}"
export CAMERA_V2_DISPLAY_EMPTY_HOLD_MS="${CAMERA_V2_DISPLAY_EMPTY_HOLD_MS:-350}"
export CAMERA_V2_DISPLAY_BOX_SIDE_MARGIN="${CAMERA_V2_DISPLAY_BOX_SIDE_MARGIN:-0.06}"
export CAMERA_V2_DISPLAY_BOX_TOP_MARGIN="${CAMERA_V2_DISPLAY_BOX_TOP_MARGIN:-0.04}"
export CAMERA_V2_DISPLAY_BOX_BOTTOM_MARGIN="${CAMERA_V2_DISPLAY_BOX_BOTTOM_MARGIN:-0.07}"
export CAMERA_V2_DISPLAY_SIZE_HOLD_SEC="${CAMERA_V2_DISPLAY_SIZE_HOLD_SEC:-0.22}"
export CAMERA_V2_DISPLAY_SHRINK_ALPHA="${CAMERA_V2_DISPLAY_SHRINK_ALPHA:-0.42}"
export CAMERA_V2_TRACK_JUMP_DIAG_LIMIT="${CAMERA_V2_TRACK_JUMP_DIAG_LIMIT:-1.00}"
export CAMERA_V85_NVDCF_FEATURE_LEVEL="${CAMERA_V85_NVDCF_FEATURE_LEVEL:-1}"

export CAMERA_V2_DETECT_CONF="${CAMERA_V2_DETECT_CONF:-0.18}"
export CAMERA_V2_MAX_DET="${CAMERA_V2_MAX_DET:-20}"
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS="${CAMERA_V2_MAX_DETECT_RESULT_AGE_MS:-400}"
export CAMERA_V2_DETECT_ENABLED="${CAMERA_V2_DETECT_ENABLED:-1}"
export CAMERA_V2_ANALYTICS_ENABLED="${CAMERA_V2_ANALYTICS_ENABLED:-1}"
export CAMERA_V84_GLOBAL_INITIAL_HZ="${CAMERA_V84_GLOBAL_INITIAL_HZ:-4.50}"
export CAMERA_V84_GLOBAL_MIN_HZ="${CAMERA_V84_GLOBAL_MIN_HZ:-1.50}"
export CAMERA_V84_GLOBAL_MAX_HZ="${CAMERA_V84_GLOBAL_MAX_HZ:-6.00}"
export CAMERA_V84_DETECT_GPU_BUDGET="${CAMERA_V84_DETECT_GPU_BUDGET:-0.30}"
export CAMERA_V84_CAPTURE_TIMEOUT="${CAMERA_V84_CAPTURE_TIMEOUT:-0.12}"
export CAMERA_V84_EMA_ALPHA="${CAMERA_V84_EMA_ALPHA:-0.20}"

# Revert prior A/B knobs. V8.9 isolates display compositor cost only.
unset NVDS_DISABLE_CUDADEV_BLOCKINGSYNC || true
unset CAMERA_V88_QUIET_BEFORE_MS || true

export CAMERA_V2_STARTUP_STAGGER_SEC="${CAMERA_V2_STARTUP_STAGGER_SEC:-0.40}"
export CAMERA_V2_SOURCE_STALL_SEC="${CAMERA_V2_SOURCE_STALL_SEC:-12}"
export CAMERA_V2_TRT86_PYTHON="${CAMERA_V2_TRT86_PYTHON:-$ROOT/.venv-trt86/bin/python}"
export CAMERA_V2_TRT86_SHM_WORKER="${CAMERA_V2_TRT86_SHM_WORKER:-$ROOT/scripts/yolo26_trt86_shm_worker_v4.py}"
export CAMERA_V2_TRT86_ENGINE="${CAMERA_V2_TRT86_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
export QWEN_REID_ENABLED=0
unset QWEN_REID_URL QWEN_REID_MODEL QWEN_REID_TIMEOUT_SEC || true

for plugin in nvurisrcbin tee queue nvstreammux nvvideoconvert appsink nvtracker fakesink; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing plugin: $plugin"
done
[[ -x "$CAMERA_V2_TRT86_PYTHON" ]] || fail "TRT8.6 python missing: $CAMERA_V2_TRT86_PYTHON"
[[ -f "$CAMERA_V2_TRT86_SHM_WORKER" ]] || fail "batch1 worker missing: $CAMERA_V2_TRT86_SHM_WORKER"
[[ -s "$CAMERA_V2_TRT86_ENGINE" ]] || fail "batch1 engine missing: $CAMERA_V2_TRT86_ENGINE"

MAIN_PYTHON=""
for candidate in "${CAMERA_V2_MAIN_PYTHON:-}" "$ROOT/.venv/bin/python" "$(command -v python3 2>/dev/null || true)"; do
  [[ -n "$candidate" && -x "$candidate" ]] || continue
  if PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$candidate" - <<'PY' >/dev/null 2>&1
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)
from services.camera_v2.runtime_v89_display_cost import PascalDisplayCostProbeRuntime  # noqa
PY
  then MAIN_PYTHON="$candidate"; break; fi
done
[[ -n "$MAIN_PYTHON" ]] || fail "no Python can import V8.9 runtime"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
printf '%s\n' \
  "CAMERA_V89_PREFLIGHT status=OK python=$MAIN_PYTHON single_owner=1" \
  "CAMERA_V89_PROFILE source=${CAMERA_V2_SOURCE_FPS}fps tracker=${CAMERA_V2_TRACK_WIDTH}x${CAMERA_V2_TRACK_HEIGHT}@${CAMERA_V2_TRACK_FPS}Hz feature_level=${CAMERA_V85_NVDCF_FEATURE_LEVEL} detector=batch1-v84" \
  "CAMERA_V89_POLICY one_change=remove-tiler-osd-egl display_mux_kept=1 bbox_unchanged=1 detector_unchanged=1 tracker_unchanged=1 production_ui=0" \
  "CAMERA_V89_NOTE no_video_window_expected=1 this_is_diagnostic_only=1"

restart_count=0
while true; do
  set +e
  "$MAIN_PYTHON" -u -m services.camera_v2.runtime_v89_display_cost
  rc=$?
  set -e
  [[ $rc -eq 75 ]] || exit "$rc"
  restart_count=$((restart_count + 1))
  delay=$restart_count
  (( delay > 10 )) && delay=10
  echo "CAMERA_V89_SUPERVISOR restart=$restart_count delay=${delay}s" >&2
  sleep "$delay"
done
