#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
LOCK_FILE="/tmp/ai_surveillance_camera_v2_gpu.lock"

fail() { printf 'CAMERA_V8_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }
command -v flock >/dev/null 2>&1 || fail "flock missing (install util-linux)"
exec 9>"$LOCK_FILE"
flock -n 9 || fail "another Camera V2 owner holds $LOCK_FILE; stop it first"

# Presentation remains independent and latest-only. A modest LAN jitter buffer keeps
# RTSP stable without letting old frames accumulate in the visible wall.
export CAMERA_V2_RTSP_TRANSPORT="${CAMERA_V2_RTSP_TRANSPORT:-tcp}"
export CAMERA_V2_RTSP_LATENCY_MS="${CAMERA_V2_RTSP_LATENCY_MS:-60}"
export CAMERA_V2_SOURCE_FPS="${CAMERA_V2_SOURCE_FPS:-20}"
export CAMERA_V2_EXTRA_SURFACES="${CAMERA_V2_EXTRA_SURFACES:-6}"
export CAMERA_V2_DISPLAY_WIDTH="${CAMERA_V2_DISPLAY_WIDTH:-1280}"
export CAMERA_V2_DISPLAY_HEIGHT="${CAMERA_V2_DISPLAY_HEIGHT:-720}"
export CAMERA_V2_WALL_WIDTH="${CAMERA_V2_WALL_WIDTH:-1920}"
export CAMERA_V2_WALL_HEIGHT="${CAMERA_V2_WALL_HEIGHT:-720}"

# NvDCF sees real frames at 8 Hz. V8 never drops these frames because TensorRT owns a
# Python lock; there is no shared GPU lane at all.
export CAMERA_V2_TRACK_WIDTH="${CAMERA_V2_TRACK_WIDTH:-512}"
export CAMERA_V2_TRACK_HEIGHT="${CAMERA_V2_TRACK_HEIGHT:-288}"
export CAMERA_V2_TRACK_FPS="${CAMERA_V2_TRACK_FPS:-8}"
export CAMERA_V2_MIN_DISPLAY_TRACK_CONF="${CAMERA_V2_MIN_DISPLAY_TRACK_CONF:-0.28}"
export CAMERA_V2_DISPLAY_TRACK_MAX_AGE_MS="${CAMERA_V2_DISPLAY_TRACK_MAX_AGE_MS:-420}"
export CAMERA_V2_DISPLAY_EMPTY_HOLD_MS="${CAMERA_V2_DISPLAY_EMPTY_HOLD_MS:-350}"

# Display-only body envelope: fast open, short hold, slow close. Association continues
# to use raw detector/NvDCF geometry; these margins cannot mint/merge tracker IDs.
export CAMERA_V2_DISPLAY_BOX_SIDE_MARGIN="${CAMERA_V2_DISPLAY_BOX_SIDE_MARGIN:-0.06}"
export CAMERA_V2_DISPLAY_BOX_TOP_MARGIN="${CAMERA_V2_DISPLAY_BOX_TOP_MARGIN:-0.04}"
export CAMERA_V2_DISPLAY_BOX_BOTTOM_MARGIN="${CAMERA_V2_DISPLAY_BOX_BOTTOM_MARGIN:-0.07}"
export CAMERA_V2_DISPLAY_SIZE_HOLD_SEC="${CAMERA_V2_DISPLAY_SIZE_HOLD_SEC:-0.22}"
export CAMERA_V2_DISPLAY_SHRINK_ALPHA="${CAMERA_V2_DISPLAY_SHRINK_ALPHA:-0.42}"
export CAMERA_V2_TRACK_JUMP_DIAG_LIMIT="${CAMERA_V2_TRACK_JUMP_DIAG_LIMIT:-1.00}"

# One detector batch contains all six cameras. The initial rate is 1 Hz/camera and is
# adapted from integrated batch latency. 28% GPU duty is intentionally conservative on
# GP107 so NvDCF and the wall retain latency headroom.
export CAMERA_V2_DETECT_HZ="${CAMERA_V2_DETECT_HZ:-1.00}"
export CAMERA_V2_DETECT_CONF="${CAMERA_V2_DETECT_CONF:-0.18}"
export CAMERA_V2_MAX_DET="${CAMERA_V2_MAX_DET:-20}"
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS="${CAMERA_V2_MAX_DETECT_RESULT_AGE_MS:-650}"
export CAMERA_V2_DETECT_ENABLED="${CAMERA_V2_DETECT_ENABLED:-1}"
export CAMERA_V2_ANALYTICS_ENABLED="${CAMERA_V2_ANALYTICS_ENABLED:-1}"
export CAMERA_V8_DETECT_INITIAL_HZ="${CAMERA_V8_DETECT_INITIAL_HZ:-1.00}"
export CAMERA_V8_DETECT_MIN_HZ="${CAMERA_V8_DETECT_MIN_HZ:-0.70}"
export CAMERA_V8_DETECT_MAX_HZ="${CAMERA_V8_DETECT_MAX_HZ:-2.00}"
export CAMERA_V8_DETECT_GPU_BUDGET="${CAMERA_V8_DETECT_GPU_BUDGET:-0.28}"
export CAMERA_V8_CAPTURE_BATCH_TIMEOUT="${CAMERA_V8_CAPTURE_BATCH_TIMEOUT:-0.30}"
export CAMERA_V8_DETECT_EMA_ALPHA="${CAMERA_V8_DETECT_EMA_ALPHA:-0.20}"

export CAMERA_V2_STARTUP_STAGGER_SEC="${CAMERA_V2_STARTUP_STAGGER_SEC:-0.40}"
export CAMERA_V2_SOURCE_STALL_SEC="${CAMERA_V2_SOURCE_STALL_SEC:-12}"

export CAMERA_V8_TRT86_PYTHON="${CAMERA_V8_TRT86_PYTHON:-$ROOT/.venv-trt86/bin/python}"
export CAMERA_V8_TRT86_BATCH_WORKER="${CAMERA_V8_TRT86_BATCH_WORKER:-$ROOT/scripts/yolo26_trt86_batch6_worker_v8.py}"
export CAMERA_V8_TRT86_ENGINE="${CAMERA_V8_TRT86_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b6-fp32-trt86.engine}"

export QWEN_REID_ENABLED=0
unset QWEN_REID_URL QWEN_REID_MODEL QWEN_REID_TIMEOUT_SEC || true

for plugin in nvurisrcbin tee queue nvstreammux nvmultistreamtiler nvvideoconvert appsink nvtracker nvdsosd nveglglessink fakesink; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing DeepStream/GStreamer plugin: $plugin"
done

GPU_LINE="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -n1 || true)"
[[ -n "$GPU_LINE" ]] || GPU_LINE="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
echo "CAMERA_V8_GPU ${GPU_LINE:-unknown}"

# A batch-6 engine cannot be derived from the old fixed-batch-1 plan. Prepare it from
# a batch-capable ONNX or the original yolo26s.pt when needed.
if [[ "$CAMERA_V2_DETECT_ENABLED" == "1" ]]; then
  bash "$ROOT/scripts/prepare_yolo26_batch6_v8.sh"
fi

[[ -x "$CAMERA_V8_TRT86_PYTHON" ]] || fail "TRT8.6 python missing: $CAMERA_V8_TRT86_PYTHON"
"$CAMERA_V8_TRT86_PYTHON" - <<'PY'
import tensorrt as trt
if not str(trt.__version__).startswith("8.6.1"):
    raise SystemExit(f"CAMERA_V8_PREFLIGHT ERROR: TensorRT 8.6.1 required, got {trt.__version__}")
print(f"CAMERA_V8_TRT_ENV trt={trt.__version__}", flush=True)
PY

MAIN_PYTHON=""
for candidate in "${CAMERA_V2_MAIN_PYTHON:-}" "$ROOT/.venv/bin/python" "$(command -v python3 2>/dev/null || true)"; do
  [[ -n "$candidate" && -x "$candidate" ]] || continue
  if PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$candidate" - <<'PY' >/dev/null 2>&1
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)
import numpy, yaml, dotenv  # noqa: F401
from services.camera_v2.runtime_v8_pascal import PascalBatchLowLatencyRuntime  # noqa: F401
PY
  then MAIN_PYTHON="$candidate"; break; fi
done
[[ -n "$MAIN_PYTHON" ]] || fail "no Python can import V8 Pascal runtime"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
printf '%s\n' \
  "CAMERA_V8_PREFLIGHT status=OK python=$MAIN_PYTHON single_owner=1" \
  "CAMERA_V8_PROFILE source=${CAMERA_V2_SOURCE_FPS}fps rtsp_latency=${CAMERA_V2_RTSP_LATENCY_MS}ms display=${CAMERA_V2_DISPLAY_WIDTH}x${CAMERA_V2_DISPLAY_HEIGHT} tracker=${CAMERA_V2_TRACK_WIDTH}x${CAMERA_V2_TRACK_HEIGHT}@${CAMERA_V2_TRACK_FPS}Hz detector=batch6@${CAMERA_V8_DETECT_INITIAL_HZ}Hz adaptive=${CAMERA_V8_DETECT_MIN_HZ}-${CAMERA_V8_DETECT_MAX_HZ}Hz budget=${CAMERA_V8_DETECT_GPU_BUDGET}" \
  "CAMERA_V8_POLICY pascal_trt86=1 native_nvinfer=0 gpu_lane=0 tracker_drop_for_detector=0 latest_only_queues=1 predictor=0 bbox_hold=${CAMERA_V2_DISPLAY_EMPTY_HOLD_MS}ms" \
  "CAMERA_V8_PIPELINE decode-once->tee->{display-independent,tracker/NvDCF,detector-coalesced-batch6}"

restart_count=0
while true; do
  set +e
  "$MAIN_PYTHON" -u -m services.camera_v2.runtime_v8_pascal
  rc=$?
  set -e
  [[ $rc -eq 75 ]] || exit "$rc"
  restart_count=$((restart_count + 1))
  delay=$restart_count
  (( delay > 10 )) && delay=10
  echo "CAMERA_V8_SUPERVISOR restart=$restart_count delay=${delay}s" >&2
  sleep "$delay"
done
