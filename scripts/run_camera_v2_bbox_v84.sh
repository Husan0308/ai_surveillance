#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
LOCK_FILE="/tmp/ai_surveillance_camera_v2_gpu.lock"

fail() { printf 'CAMERA_V84_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }
command -v flock >/dev/null 2>&1 || fail "flock missing (install util-linux)"
exec 9>"$LOCK_FILE"
flock -n 9 || fail "another Camera V2 owner holds $LOCK_FILE; stop it first"

# Keep the proven V8 presentation path. The detector experiment changes only batching.
export CAMERA_V2_RTSP_TRANSPORT="${CAMERA_V2_RTSP_TRANSPORT:-tcp}"
export CAMERA_V2_RTSP_LATENCY_MS="${CAMERA_V2_RTSP_LATENCY_MS:-60}"
export CAMERA_V2_SOURCE_FPS="${CAMERA_V2_SOURCE_FPS:-20}"
export CAMERA_V2_EXTRA_SURFACES="${CAMERA_V2_EXTRA_SURFACES:-6}"
export CAMERA_V2_DISPLAY_WIDTH="${CAMERA_V2_DISPLAY_WIDTH:-1280}"
export CAMERA_V2_DISPLAY_HEIGHT="${CAMERA_V2_DISPLAY_HEIGHT:-720}"
export CAMERA_V2_WALL_WIDTH="${CAMERA_V2_WALL_WIDTH:-1920}"
export CAMERA_V2_WALL_HEIGHT="${CAMERA_V2_WALL_HEIGHT:-720}"

# Exact original V8 NvDCF A/B baseline. Do not tune bbox/tracker in this detector step.
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

# Batch-1 detector: one latest camera frame at a time, persistent TRT8.6 context,
# fair round robin across the six cameras. 4.5 Hz global ~= 0.75 Hz/camera initially.
# The runtime adapts the GLOBAL rate to keep detector GPU duty near 30%.
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

export CAMERA_V2_STARTUP_STAGGER_SEC="${CAMERA_V2_STARTUP_STAGGER_SEC:-0.40}"
export CAMERA_V2_SOURCE_STALL_SEC="${CAMERA_V2_SOURCE_STALL_SEC:-12}"

# Reuse the already-proven batch-1 TRT8.6 engine and production pinned/async worker.
export CAMERA_V2_TRT86_PYTHON="${CAMERA_V2_TRT86_PYTHON:-$ROOT/.venv-trt86/bin/python}"
export CAMERA_V2_TRT86_SHM_WORKER="${CAMERA_V2_TRT86_SHM_WORKER:-$ROOT/scripts/yolo26_trt86_shm_worker_v4.py}"
export CAMERA_V2_TRT86_ENGINE="${CAMERA_V2_TRT86_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"

export QWEN_REID_ENABLED=0
unset QWEN_REID_URL QWEN_REID_MODEL QWEN_REID_TIMEOUT_SEC || true

for plugin in nvurisrcbin tee queue nvstreammux nvmultistreamtiler nvvideoconvert appsink nvtracker nvdsosd nveglglessink fakesink; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing DeepStream/GStreamer plugin: $plugin"
done

GPU_LINE="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -n1 || true)"
[[ -n "$GPU_LINE" ]] || GPU_LINE="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
echo "CAMERA_V84_GPU ${GPU_LINE:-unknown}"

[[ -x "$CAMERA_V2_TRT86_PYTHON" ]] || fail "TRT8.6 python missing: $CAMERA_V2_TRT86_PYTHON"
[[ -f "$CAMERA_V2_TRT86_SHM_WORKER" ]] || fail "batch1 worker missing: $CAMERA_V2_TRT86_SHM_WORKER"
[[ -s "$CAMERA_V2_TRT86_ENGINE" ]] || fail "batch1 engine missing: $CAMERA_V2_TRT86_ENGINE"

"$CAMERA_V2_TRT86_PYTHON" - "$CAMERA_V2_TRT86_ENGINE" <<'PY'
import sys
from pathlib import Path
import tensorrt as trt
p=Path(sys.argv[1])
if not str(trt.__version__).startswith("8.6.1"):
    raise SystemExit(f"CAMERA_V84_PREFLIGHT ERROR: TensorRT 8.6.1 required, got {trt.__version__}")
log=trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(log, "")
e=trt.Runtime(log).deserialize_cuda_engine(p.read_bytes())
if e is None: raise SystemExit("CAMERA_V84_PREFLIGHT ERROR: deserialize failed")
c=e.create_execution_context()
ins=[i for i in range(e.num_bindings) if e.binding_is_input(i)]
outs=[i for i in range(e.num_bindings) if not e.binding_is_input(i)]
i=tuple(int(v) for v in c.get_binding_shape(ins[0])); o=tuple(int(v) for v in c.get_binding_shape(outs[0]))
if i != (1,3,384,672) or o != (1,300,6):
    raise SystemExit(f"CAMERA_V84_PREFLIGHT ERROR: wrong shapes input={i} output={o}")
print(f"CAMERA_V84_TRT_ENV trt={trt.__version__} engine={p.name} input={i} output={o}", flush=True)
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
from services.camera_v2.runtime_v84_batch1 import PascalBatch1LowLatencyRuntime  # noqa: F401
PY
  then MAIN_PYTHON="$candidate"; break; fi
done
[[ -n "$MAIN_PYTHON" ]] || fail "no Python can import V8.4 batch1 runtime"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
printf '%s\n' \
  "CAMERA_V84_PREFLIGHT status=OK python=$MAIN_PYTHON single_owner=1 batch1_engine=1" \
  "CAMERA_V84_PROFILE source=${CAMERA_V2_SOURCE_FPS}fps rtsp_latency=${CAMERA_V2_RTSP_LATENCY_MS}ms tracker=${CAMERA_V2_TRACK_WIDTH}x${CAMERA_V2_TRACK_HEIGHT}@${CAMERA_V2_TRACK_FPS}Hz detector=batch1-roundrobin global=${CAMERA_V84_GLOBAL_INITIAL_HZ}Hz adaptive=${CAMERA_V84_GLOBAL_MIN_HZ}-${CAMERA_V84_GLOBAL_MAX_HZ}Hz budget=${CAMERA_V84_DETECT_GPU_BUDGET}" \
  "CAMERA_V84_POLICY batch6=0 coalesced_six_wait=0 latest_one_camera=1 queue_depth=1 gpu_lane=0 tracker_drop_for_detector=0 bbox_unchanged=1" \
  "CAMERA_V84_PIPELINE decode-once->tee->{display-independent,tracker/NvDCF,detector-latest-roundrobin-batch1}"

restart_count=0
while true; do
  set +e
  "$MAIN_PYTHON" -u -m services.camera_v2.runtime_v84_batch1
  rc=$?
  set -e
  [[ $rc -eq 75 ]] || exit "$rc"
  restart_count=$((restart_count + 1))
  delay=$restart_count
  (( delay > 10 )) && delay=10
  echo "CAMERA_V84_SUPERVISOR restart=$restart_count delay=${delay}s" >&2
  sleep "$delay"
done
