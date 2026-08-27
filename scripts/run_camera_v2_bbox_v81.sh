#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
LOCK_FILE="/tmp/ai_surveillance_camera_v2_gpu.lock"

fail() { printf 'CAMERA_V81_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }
command -v flock >/dev/null 2>&1 || fail "flock missing (install util-linux)"
exec 9>"$LOCK_FILE"
flock -n 9 || fail "another Camera V2 owner holds $LOCK_FILE; stop it first"

# Keep V8's proven low-latency presentation path.
export CAMERA_V2_RTSP_TRANSPORT="${CAMERA_V2_RTSP_TRANSPORT:-tcp}"
export CAMERA_V2_RTSP_LATENCY_MS="${CAMERA_V2_RTSP_LATENCY_MS:-60}"
export CAMERA_V2_SOURCE_FPS="${CAMERA_V2_SOURCE_FPS:-20}"
export CAMERA_V2_EXTRA_SURFACES="${CAMERA_V2_EXTRA_SURFACES:-6}"
export CAMERA_V2_DISPLAY_WIDTH="${CAMERA_V2_DISPLAY_WIDTH:-1280}"
export CAMERA_V2_DISPLAY_HEIGHT="${CAMERA_V2_DISPLAY_HEIGHT:-720}"
export CAMERA_V2_WALL_WIDTH="${CAMERA_V2_WALL_WIDTH:-1920}"
export CAMERA_V2_WALL_HEIGHT="${CAMERA_V2_WALL_HEIGHT:-720}"

# V8.1 spends the GPU headroom recovered by batch-6 on temporal tracker cadence.
# 12 Hz cuts real-tracker sampling interval from 125 ms to ~83 ms. The native copy
# floor is lowered so valid moving NvDCF tracks are not discarded before Python sees them.
export CAMERA_V2_TRACK_WIDTH="${CAMERA_V2_TRACK_WIDTH:-512}"
export CAMERA_V2_TRACK_HEIGHT="${CAMERA_V2_TRACK_HEIGHT:-288}"
export CAMERA_V2_TRACK_FPS="${CAMERA_V2_TRACK_FPS:-12}"
export CAMERA_V2_MIN_DISPLAY_TRACK_CONF="${CAMERA_V2_MIN_DISPLAY_TRACK_CONF:-0.10}"
export CAMERA_V2_DISPLAY_TRACK_MAX_AGE_MS="${CAMERA_V2_DISPLAY_TRACK_MAX_AGE_MS:-280}"
export CAMERA_V2_DISPLAY_EMPTY_HOLD_MS="${CAMERA_V2_DISPLAY_EMPTY_HOLD_MS:-220}"

export CAMERA_V2_DISPLAY_BOX_SIDE_MARGIN="${CAMERA_V2_DISPLAY_BOX_SIDE_MARGIN:-0.06}"
export CAMERA_V2_DISPLAY_BOX_TOP_MARGIN="${CAMERA_V2_DISPLAY_BOX_TOP_MARGIN:-0.04}"
export CAMERA_V2_DISPLAY_BOX_BOTTOM_MARGIN="${CAMERA_V2_DISPLAY_BOX_BOTTOM_MARGIN:-0.07}"
export CAMERA_V2_DISPLAY_SIZE_HOLD_SEC="${CAMERA_V2_DISPLAY_SIZE_HOLD_SEC:-0.14}"
export CAMERA_V2_DISPLAY_SHRINK_ALPHA="${CAMERA_V2_DISPLAY_SHRINK_ALPHA:-0.55}"
export CAMERA_V2_TRACK_JUMP_DIAG_LIMIT="${CAMERA_V2_TRACK_JUMP_DIAG_LIMIT:-1.00}"

# Batch-6 detector remains independent. Old detector geometry beyond 70 ms is aligned
# to the newest real NvDCF target when it is an existing person. Very old unmatched
# rectangles are never allowed to create a new target on a current video frame.
export CAMERA_V2_DETECT_HZ="${CAMERA_V2_DETECT_HZ:-2.00}"
export CAMERA_V2_DETECT_CONF="${CAMERA_V2_DETECT_CONF:-0.18}"
export CAMERA_V2_MAX_DET="${CAMERA_V2_MAX_DET:-20}"
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS="${CAMERA_V2_MAX_DETECT_RESULT_AGE_MS:-300}"
export CAMERA_V2_DETECT_ENABLED="${CAMERA_V2_DETECT_ENABLED:-1}"
export CAMERA_V2_ANALYTICS_ENABLED="${CAMERA_V2_ANALYTICS_ENABLED:-1}"
export CAMERA_V8_DETECT_INITIAL_HZ="${CAMERA_V8_DETECT_INITIAL_HZ:-2.00}"
export CAMERA_V8_DETECT_MIN_HZ="${CAMERA_V8_DETECT_MIN_HZ:-0.70}"
export CAMERA_V8_DETECT_MAX_HZ="${CAMERA_V8_DETECT_MAX_HZ:-2.00}"
export CAMERA_V8_DETECT_GPU_BUDGET="${CAMERA_V8_DETECT_GPU_BUDGET:-0.28}"
export CAMERA_V8_CAPTURE_BATCH_TIMEOUT="${CAMERA_V8_CAPTURE_BATCH_TIMEOUT:-0.30}"
export CAMERA_V8_DETECT_EMA_ALPHA="${CAMERA_V8_DETECT_EMA_ALPHA:-0.20}"

export CAMERA_V81_CURRENTIZE_AFTER_MS="${CAMERA_V81_CURRENTIZE_AFTER_MS:-70}"
export CAMERA_V81_NEW_TARGET_MAX_AGE_MS="${CAMERA_V81_NEW_TARGET_MAX_AGE_MS:-240}"
export CAMERA_V81_RAW_TRACK_MAX_AGE_MS="${CAMERA_V81_RAW_TRACK_MAX_AGE_MS:-180}"
export CAMERA_V81_EMPTY_DETECTOR_SKIP_AGE_MS="${CAMERA_V81_EMPTY_DETECTOR_SKIP_AGE_MS:-80}"

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

[[ -s "$CAMERA_V8_TRT86_ENGINE" ]] || fail "batch-6 engine missing: $CAMERA_V8_TRT86_ENGINE; run bash scripts/prepare_yolo26_batch6_v8.sh first"
[[ -x "$CAMERA_V8_TRT86_PYTHON" ]] || fail "TRT8.6 python missing: $CAMERA_V8_TRT86_PYTHON"

GPU_LINE="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -n1 || true)"
echo "CAMERA_V81_GPU ${GPU_LINE:-unknown}"

"$CAMERA_V8_TRT86_PYTHON" - "$CAMERA_V8_TRT86_ENGINE" <<'PY'
import sys
from pathlib import Path
import tensorrt as trt
p = Path(sys.argv[1])
if not str(trt.__version__).startswith("8.6.1"):
    raise SystemExit(f"CAMERA_V81_PREFLIGHT ERROR: TensorRT 8.6.1 required, got {trt.__version__}")
logger = trt.Logger(trt.Logger.ERROR)
engine = trt.Runtime(logger).deserialize_cuda_engine(p.read_bytes())
if engine is None:
    raise SystemExit("CAMERA_V81_PREFLIGHT ERROR: engine deserialize failed")
ctx = engine.create_execution_context()
ins = [i for i in range(engine.num_bindings) if engine.binding_is_input(i)]
outs = [i for i in range(engine.num_bindings) if not engine.binding_is_input(i)]
i = tuple(int(v) for v in ctx.get_binding_shape(ins[0]))
o = tuple(int(v) for v in ctx.get_binding_shape(outs[0]))
if i != (6,3,384,672) or o != (6,300,6):
    raise SystemExit(f"CAMERA_V81_PREFLIGHT ERROR: wrong engine shapes input={i} output={o}")
print(f"CAMERA_V81_ENGINE PASS input={i} output={o} trt={trt.__version__}")
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
from services.camera_v2.runtime_v81_sync import PascalStickySyncRuntime  # noqa: F401
PY
  then MAIN_PYTHON="$candidate"; break; fi
done
[[ -n "$MAIN_PYTHON" ]] || fail "no Python can import V8.1 runtime"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
printf '%s\n' \
  "CAMERA_V81_PREFLIGHT status=OK python=$MAIN_PYTHON single_owner=1" \
  "CAMERA_V81_PROFILE source=${CAMERA_V2_SOURCE_FPS}fps rtsp=${CAMERA_V2_RTSP_LATENCY_MS}ms tracker=${CAMERA_V2_TRACK_WIDTH}x${CAMERA_V2_TRACK_HEIGHT}@${CAMERA_V2_TRACK_FPS}Hz display_conf=${CAMERA_V2_MIN_DISPLAY_TRACK_CONF} detector=batch6@${CAMERA_V8_DETECT_INITIAL_HZ}Hz" \
  "CAMERA_V81_POLICY false_empty_fabrication=0 stale_detector_geometry=currentized empty_detector=skip-if-active hold=${CAMERA_V2_DISPLAY_EMPTY_HOLD_MS}ms predictor=0 gpu_lane=0" \
  "CAMERA_V81_PIPELINE decode-once->tee->{display-20fps,tracker-real-NvDCF,detector-batch6}"

restart_count=0
while true; do
  set +e
  "$MAIN_PYTHON" -u -m services.camera_v2.runtime_v81_sync
  rc=$?
  set -e
  [[ $rc -eq 75 ]] || exit "$rc"
  restart_count=$((restart_count + 1))
  delay=$restart_count
  (( delay > 10 )) && delay=10
  echo "CAMERA_V81_SUPERVISOR restart=$restart_count delay=${delay}s" >&2
  sleep "$delay"
done
