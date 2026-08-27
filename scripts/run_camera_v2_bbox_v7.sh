#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

# V7 restores the old visual-tracking contract: camera/display stays independent,
# sparse YOLO only corrects targets, and NvDCF localizes the person on video frames.
export CAMERA_V2_RTSP_TRANSPORT="${CAMERA_V2_RTSP_TRANSPORT:-tcp}"
export CAMERA_V2_RTSP_LATENCY_MS="${CAMERA_V2_RTSP_LATENCY_MS:-80}"
export CAMERA_V2_SOURCE_FPS="${CAMERA_V2_SOURCE_FPS:-20}"
export CAMERA_V2_EXTRA_SURFACES="${CAMERA_V2_EXTRA_SURFACES:-8}"

export CAMERA_V2_DISPLAY_WIDTH="${CAMERA_V2_DISPLAY_WIDTH:-1280}"
export CAMERA_V2_DISPLAY_HEIGHT="${CAMERA_V2_DISPLAY_HEIGHT:-720}"
export CAMERA_V2_WALL_WIDTH="${CAMERA_V2_WALL_WIDTH:-1920}"
export CAMERA_V2_WALL_HEIGHT="${CAMERA_V2_WALL_HEIGHT:-720}"

# Known-good Pascal NvDCF geometry, now at source cadence rather than a 10 Hz gate.
# This is the key difference from Step-4 V6: no application-side velocity predictor.
export CAMERA_V2_TRACK_WIDTH="${CAMERA_V2_TRACK_WIDTH:-512}"
export CAMERA_V2_TRACK_HEIGHT="${CAMERA_V2_TRACK_HEIGHT:-288}"
export CAMERA_V2_TRACK_FPS="${CAMERA_V2_TRACK_FPS:-20}"
export CAMERA_V2_MIN_DISPLAY_TRACK_CONF="${CAMERA_V2_MIN_DISPLAY_TRACK_CONF:-0.28}"
export CAMERA_V2_DISPLAY_TRACK_MAX_AGE_MS="${CAMERA_V2_DISPLAY_TRACK_MAX_AGE_MS:-120}"

# Display-only full-body safety envelope from the earlier good camera tracker.
# It never feeds back into NvDCF association.
export CAMERA_V2_DISPLAY_BOX_SIDE_MARGIN="${CAMERA_V2_DISPLAY_BOX_SIDE_MARGIN:-0.06}"
export CAMERA_V2_DISPLAY_BOX_TOP_MARGIN="${CAMERA_V2_DISPLAY_BOX_TOP_MARGIN:-0.04}"
export CAMERA_V2_DISPLAY_BOX_BOTTOM_MARGIN="${CAMERA_V2_DISPLAY_BOX_BOTTOM_MARGIN:-0.07}"
export CAMERA_V2_DISPLAY_SIZE_HOLD_SEC="${CAMERA_V2_DISPLAY_SIZE_HOLD_SEC:-0.22}"
export CAMERA_V2_DISPLAY_SHRINK_ALPHA="${CAMERA_V2_DISPLAY_SHRINK_ALPHA:-0.42}"
export CAMERA_V2_TRACK_JUMP_DIAG_LIMIT="${CAMERA_V2_TRACK_JUMP_DIAG_LIMIT:-1.00}"

# 1.5 Hz/camera is deliberate for the GTX 1050 Ti: NvDCF owns skipped-frame motion.
# Raise to 2.0 only after the V7 acceptance log proves tracker_rate/display FPS remain healthy.
export CAMERA_V2_DETECT_HZ="${CAMERA_V2_DETECT_HZ:-1.50}"
export CAMERA_V2_DETECT_CONF="${CAMERA_V2_DETECT_CONF:-0.08}"
export CAMERA_V2_MAX_DET="${CAMERA_V2_MAX_DET:-20}"
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS="${CAMERA_V2_MAX_DETECT_RESULT_AGE_MS:-260}"
export CAMERA_V2_DETECT_ENABLED="${CAMERA_V2_DETECT_ENABLED:-1}"
export CAMERA_V2_ANALYTICS_ENABLED="${CAMERA_V2_ANALYTICS_ENABLED:-1}"

export CAMERA_V2_STARTUP_STAGGER_SEC="${CAMERA_V2_STARTUP_STAGGER_SEC:-0.50}"
export CAMERA_V2_SOURCE_STALL_SEC="${CAMERA_V2_SOURCE_STALL_SEC:-12}"

export CAMERA_V2_TRT86_PYTHON="${CAMERA_V2_TRT86_PYTHON:-$ROOT/.venv-trt86/bin/python}"
export CAMERA_V2_TRT86_ENGINE="${CAMERA_V2_TRT86_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
export CAMERA_V2_TRT86_SHM_WORKER="${CAMERA_V2_TRT86_SHM_WORKER:-$ROOT/scripts/yolo26_trt86_shm_worker_v4.py}"
RESTORE_HELPER="$ROOT/scripts/restore_cam01_trt86_engine.sh"

export QWEN_REID_ENABLED=0
unset QWEN_REID_URL QWEN_REID_MODEL QWEN_REID_TIMEOUT_SEC || true

fail() { printf 'CAMERA_BBOX_V7_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }

for plugin in nvurisrcbin tee queue nvstreammux nvmultistreamtiler nvvideoconvert appsink nvtracker nvdsosd nveglglessink fakesink; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing DeepStream/GStreamer plugin: $plugin"
done

GPU_LINE="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -n1 || true)"
[[ -n "$GPU_LINE" ]] || GPU_LINE="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
echo "CAMERA_BBOX_V7_GPU ${GPU_LINE:-unknown}"

if [[ "$CAMERA_V2_DETECT_ENABLED" == "1" ]]; then
  [[ -x "$CAMERA_V2_TRT86_PYTHON" ]] || fail "TRT86 python missing: $CAMERA_V2_TRT86_PYTHON"
  if [[ ! -s "$CAMERA_V2_TRT86_ENGINE" && -f "$RESTORE_HELPER" ]]; then
    echo "CAMERA_BBOX_V7_ENGINE recovery=stash/local-search" >&2
    bash "$RESTORE_HELPER" "$CAMERA_V2_TRT86_ENGINE" || true
  fi
  [[ -s "$CAMERA_V2_TRT86_ENGINE" ]] || fail "TRT8.6 engine missing: $CAMERA_V2_TRT86_ENGINE"
  [[ -f "$CAMERA_V2_TRT86_SHM_WORKER" ]] || fail "TRT86 worker missing: $CAMERA_V2_TRT86_SHM_WORKER"

  "$CAMERA_V2_TRT86_PYTHON" - <<'PY'
import sys
import numpy as np
import tensorrt as trt
if not str(trt.__version__).startswith("8.6.1"):
    raise SystemExit(f"CAMERA_BBOX_V7_PREFLIGHT ERROR: TensorRT 8.6.1 required, got {trt.__version__}")
print(f"CAMERA_BBOX_V7_TRT_ENV python={sys.executable} trt={trt.__version__} numpy={np.__version__}", flush=True)
PY
fi

MAIN_PYTHON=""
for candidate in "${CAMERA_V2_MAIN_PYTHON:-}" "$ROOT/.venv/bin/python" "$(command -v python3 2>/dev/null || true)"; do
  [[ -n "$candidate" && -x "$candidate" ]] || continue
  if PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$candidate" - <<'PY' >/dev/null 2>&1
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)
import numpy, yaml, dotenv  # noqa: F401
from services.camera_v2.runtime_bbox_v7 import NvDCFStickyBBoxRuntime  # noqa: F401
PY
  then MAIN_PYTHON="$candidate"; break; fi
done
[[ -n "$MAIN_PYTHON" ]] || fail "no Python can import V7 NvDCF bbox runtime"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
printf '%s\n' \
  "CAMERA_BBOX_V7_PREFLIGHT status=OK python=$MAIN_PYTHON" \
  "CAMERA_BBOX_V7_PROFILE source=${CAMERA_V2_SOURCE_FPS}fps display=${CAMERA_V2_DISPLAY_WIDTH}x${CAMERA_V2_DISPLAY_HEIGHT} tracker=${CAMERA_V2_TRACK_WIDTH}x${CAMERA_V2_TRACK_HEIGHT}@${CAMERA_V2_TRACK_FPS}Hz detector=672x384@${CAMERA_V2_DETECT_HZ}Hz/cam conf=${CAMERA_V2_DETECT_CONF}" \
  "CAMERA_BBOX_V7_POLICY nvdcf_current_frame=1 cpu_velocity_predictor=0 shadow_render=0 min_conf=${CAMERA_V2_MIN_DISPLAY_TRACK_CONF} margin=${CAMERA_V2_DISPLAY_BOX_SIDE_MARGIN}/${CAMERA_V2_DISPLAY_BOX_TOP_MARGIN}/${CAMERA_V2_DISPLAY_BOX_BOTTOM_MARGIN} fast_open=1 shrink_hold=${CAMERA_V2_DISPLAY_SIZE_HOLD_SEC}s" \
  "CAMERA_BBOX_V7_PIPELINE decode-once->tee->{display,tracker/NvDCF,detector/TRT86} display_never_waits_for_analytics=1 gpu_lane=serialized"

restart_count=0
while true; do
  set +e
  "$MAIN_PYTHON" -u -m services.camera_v2.runtime_bbox_v7
  rc=$?
  set -e
  [[ $rc -eq 75 ]] || exit "$rc"
  restart_count=$((restart_count + 1))
  delay=$restart_count
  (( delay > 10 )) && delay=10
  echo "CAMERA_BBOX_V7_SUPERVISOR restart=$restart_count delay=${delay}s" >&2
  sleep "$delay"
done
