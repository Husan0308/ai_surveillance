#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
LOCK_FILE="/tmp/ai_surveillance_camera_v2_gpu.lock"
PY="${CAMERA_V94_PYTHON:-$ROOT/.venv-trt86/bin/python}"

fail() { printf 'CAMERA_V94_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }
command -v flock >/dev/null 2>&1 || fail "flock missing"
exec 9>"$LOCK_FILE"
flock -n 9 || fail "another Camera V2 owner holds $LOCK_FILE; stop it first"
[[ -x "$PY" ]] || fail "Python 3.10 TRT8.6 environment missing: $PY"

# V9.4 is a one-variable A/B over V9.3: only detector X coordinates are scaled
# from the 672px detector space into the 512px tracker-mux space.
export CAMERA_V2_RTSP_TRANSPORT="${CAMERA_V2_RTSP_TRANSPORT:-tcp}"
export CAMERA_V2_RTSP_LATENCY_MS="${CAMERA_V2_RTSP_LATENCY_MS:-60}"
export CAMERA_V2_SOURCE_FPS="${CAMERA_V2_SOURCE_FPS:-20}"
export CAMERA_V2_EXTRA_SURFACES="${CAMERA_V2_EXTRA_SURFACES:-6}"
export CAMERA_V2_DISPLAY_WIDTH="${CAMERA_V2_DISPLAY_WIDTH:-1280}"
export CAMERA_V2_DISPLAY_HEIGHT="${CAMERA_V2_DISPLAY_HEIGHT:-720}"
export CAMERA_V2_WALL_WIDTH="${CAMERA_V2_WALL_WIDTH:-1920}"
export CAMERA_V2_WALL_HEIGHT="${CAMERA_V2_WALL_HEIGHT:-720}"

# Exactly the V9.3 tracker/display settings.
export CAMERA_V2_TRACK_WIDTH="${CAMERA_V2_TRACK_WIDTH:-512}"
export CAMERA_V2_TRACK_HEIGHT="${CAMERA_V2_TRACK_HEIGHT:-288}"
export CAMERA_V2_TRACK_FPS="${CAMERA_V2_TRACK_FPS:-10}"
export CAMERA_V2_MIN_DISPLAY_TRACK_CONF="${CAMERA_V2_MIN_DISPLAY_TRACK_CONF:-0.10}"
export CAMERA_V2_DISPLAY_EMPTY_HOLD_MS="${CAMERA_V2_DISPLAY_EMPTY_HOLD_MS:-180}"
export CAMERA_V2_DISPLAY_TRACK_MAX_AGE_MS="${CAMERA_V2_DISPLAY_TRACK_MAX_AGE_MS:-210}"
export CAMERA_V2_DISPLAY_BOX_SIDE_MARGIN="${CAMERA_V2_DISPLAY_BOX_SIDE_MARGIN:-0.06}"
export CAMERA_V2_DISPLAY_BOX_TOP_MARGIN="${CAMERA_V2_DISPLAY_BOX_TOP_MARGIN:-0.04}"
export CAMERA_V2_DISPLAY_BOX_BOTTOM_MARGIN="${CAMERA_V2_DISPLAY_BOX_BOTTOM_MARGIN:-0.07}"
export CAMERA_V2_DISPLAY_SIZE_HOLD_SEC="${CAMERA_V2_DISPLAY_SIZE_HOLD_SEC:-0.18}"
export CAMERA_V2_DISPLAY_SHRINK_ALPHA="${CAMERA_V2_DISPLAY_SHRINK_ALPHA:-0.48}"
export CAMERA_V2_TRACK_JUMP_DIAG_LIMIT="${CAMERA_V2_TRACK_JUMP_DIAG_LIMIT:-1.00}"
export CAMERA_V85_NVDCF_FEATURE_LEVEL="${CAMERA_V85_NVDCF_FEATURE_LEVEL:-1}"

# Exactly the V9.2 stale-result semantics retained by V9.3.
export CAMERA_V92_CURRENTIZE_AFTER_MS="${CAMERA_V92_CURRENTIZE_AFTER_MS:-55}"
export CAMERA_V92_NEW_TARGET_MAX_AGE_MS="${CAMERA_V92_NEW_TARGET_MAX_AGE_MS:-210}"
export CAMERA_V92_RAW_TRACK_MAX_AGE_MS="${CAMERA_V92_RAW_TRACK_MAX_AGE_MS:-150}"
export CAMERA_V92_EMPTY_DETECTOR_SKIP_AGE_MS="${CAMERA_V92_EMPTY_DETECTOR_SKIP_AGE_MS:-70}"

# Exactly the V9.3 display-only bounded center compensation.
export CAMERA_V93_DISPLAY_COMP_MS="${CAMERA_V93_DISPLAY_COMP_MS:-55}"
export CAMERA_V93_DISPLAY_COMP_GAIN="${CAMERA_V93_DISPLAY_COMP_GAIN:-0.85}"
export CAMERA_V93_MAX_SHIFT_FRAC="${CAMERA_V93_MAX_SHIFT_FRAC:-0.20}"
export CAMERA_V93_MIN_SAMPLE_DT="${CAMERA_V93_MIN_SAMPLE_DT:-0.045}"
export CAMERA_V93_MAX_SAMPLE_DT="${CAMERA_V93_MAX_SAMPLE_DT:-0.30}"

# Exactly the V9.3 detector policy.
export CAMERA_V2_DETECT_CONF="${CAMERA_V2_DETECT_CONF:-0.18}"
export CAMERA_V2_MAX_DET="${CAMERA_V2_MAX_DET:-20}"
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS="${CAMERA_V2_MAX_DETECT_RESULT_AGE_MS:-250}"
export CAMERA_V2_DETECT_ENABLED="${CAMERA_V2_DETECT_ENABLED:-1}"
export CAMERA_V2_ANALYTICS_ENABLED="${CAMERA_V2_ANALYTICS_ENABLED:-1}"
export CAMERA_V84_GLOBAL_INITIAL_HZ="${CAMERA_V84_GLOBAL_INITIAL_HZ:-3.80}"
export CAMERA_V84_GLOBAL_MIN_HZ="${CAMERA_V84_GLOBAL_MIN_HZ:-1.40}"
export CAMERA_V84_GLOBAL_MAX_HZ="${CAMERA_V84_GLOBAL_MAX_HZ:-5.00}"
export CAMERA_V84_DETECT_GPU_BUDGET="${CAMERA_V84_DETECT_GPU_BUDGET:-0.26}"
export CAMERA_V84_CAPTURE_TIMEOUT="${CAMERA_V84_CAPTURE_TIMEOUT:-0.12}"
export CAMERA_V84_EMA_ALPHA="${CAMERA_V84_EMA_ALPHA:-0.20}"

export CAMERA_V2_STARTUP_STAGGER_SEC="${CAMERA_V2_STARTUP_STAGGER_SEC:-0.40}"
export CAMERA_V2_SOURCE_STALL_SEC="${CAMERA_V2_SOURCE_STALL_SEC:-12}"
export CAMERA_V2_TRT86_ENGINE="${CAMERA_V2_TRT86_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
unset NVDS_DISABLE_CUDADEV_BLOCKINGSYNC || true
export QWEN_REID_ENABLED=0
unset QWEN_REID_URL QWEN_REID_MODEL QWEN_REID_TIMEOUT_SEC || true

for plugin in nvurisrcbin tee queue nvstreammux nvmultistreamtiler nvvideoconvert appsink nvtracker nvdsosd nveglglessink fakesink; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing plugin: $plugin"
done
[[ -s "$CAMERA_V2_TRT86_ENGINE" ]] || fail "TRT8.6 batch1 engine missing: $CAMERA_V2_TRT86_ENGINE"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if ! "$PY" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(10)
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)
import tensorrt as trt
if not str(trt.__version__).startswith("8.6.1"):
    raise SystemExit(11)
from services.camera_v2.runtime_v94_xmap import PascalXMapRuntime  # noqa
print(f"CAMERA_V94_IMPORT python={sys.version.split()[0]} trt={trt.__version__} gi=OK runtime=OK")
PY
then
  fail "Python 3.10 V9.4 runtime import failed"
fi

printf '%s\n' \
  "CAMERA_V94_PREFLIGHT status=OK python=$PY single_owner=1" \
  "CAMERA_V94_PROFILE source=${CAMERA_V2_SOURCE_FPS}fps tracker=${CAMERA_V2_TRACK_WIDTH}x${CAMERA_V2_TRACK_HEIGHT}@${CAMERA_V2_TRACK_FPS}Hz detector=672x384 TRT8.6 inprocess" \
  "CAMERA_V94_POLICY only_change=detector-x-scale-672-to-${CAMERA_V2_TRACK_WIDTH} v93_other_settings=unchanged"

restart_count=0
while true; do
  set +e
  "$PY" -u -m services.camera_v2.runtime_v94_xmap
  rc=$?
  set -e
  [[ $rc -eq 75 ]] || exit "$rc"
  restart_count=$((restart_count + 1))
  delay=$restart_count
  (( delay > 10 )) && delay=10
  echo "CAMERA_V94_SUPERVISOR restart=$restart_count delay=${delay}s" >&2
  sleep "$delay"
done
