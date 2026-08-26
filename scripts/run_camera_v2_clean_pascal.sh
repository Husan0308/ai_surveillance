#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

# One production profile. Display, NvDCF and detector rates are independent.
export CAMERA_V2_RTSP_TRANSPORT="${CAMERA_V2_RTSP_TRANSPORT:-tcp}"
export CAMERA_V2_RTSP_LATENCY_MS="${CAMERA_V2_RTSP_LATENCY_MS:-80}"
export CAMERA_V2_SOURCE_FPS="${CAMERA_V2_SOURCE_FPS:-20}"
export CAMERA_V2_EXTRA_SURFACES="${CAMERA_V2_EXTRA_SURFACES:-4}"

export CAMERA_V2_DISPLAY_WIDTH="${CAMERA_V2_DISPLAY_WIDTH:-1280}"
export CAMERA_V2_DISPLAY_HEIGHT="${CAMERA_V2_DISPLAY_HEIGHT:-720}"
export CAMERA_V2_WALL_WIDTH="${CAMERA_V2_WALL_WIDTH:-1920}"
export CAMERA_V2_WALL_HEIGHT="${CAMERA_V2_WALL_HEIGHT:-720}"

# The clean display path is healthy with analytics disabled. Keep NvDCF below
# source/display cadence so it cannot monopolize the GTX 1050 Ti. Eight Hz gives
# a 125 ms local-track update period while freeing GPU time for a faster detector.
export CAMERA_V2_TRACK_FPS="${CAMERA_V2_TRACK_FPS:-8}"
export CAMERA_V2_MIN_DISPLAY_TRACK_CONF="${CAMERA_V2_MIN_DISPLAY_TRACK_CONF:-0.12}"
export CAMERA_V2_DISPLAY_TRACK_MAX_AGE_MS="${CAMERA_V2_DISPLAY_TRACK_MAX_AGE_MS:-300}"

# Measured TRT8.6 B1 FP32 is usually ~140-180 ms after warm-up. 0.50 Hz/camera
# cuts worst-case round-robin first-detection wait from ~2.5 s to ~2.0 s. The v5
# worker uses a low-priority CUDA stream so display/tracker kernels win contention.
export CAMERA_V2_DETECT_HZ="${CAMERA_V2_DETECT_HZ:-0.50}"
export CAMERA_V2_DETECT_CONF="${CAMERA_V2_DETECT_CONF:-0.18}"
export CAMERA_V2_MAX_DET="${CAMERA_V2_MAX_DET:-20}"
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS="${CAMERA_V2_MAX_DETECT_RESULT_AGE_MS:-350}"
export CAMERA_V2_DETECT_ENABLED="${CAMERA_V2_DETECT_ENABLED:-1}"
export CAMERA_V2_ANALYTICS_ENABLED="${CAMERA_V2_ANALYTICS_ENABLED:-1}"

export CAMERA_V2_STARTUP_STAGGER_SEC="${CAMERA_V2_STARTUP_STAGGER_SEC:-0.50}"
export CAMERA_V2_SOURCE_STALL_SEC="${CAMERA_V2_SOURCE_STALL_SEC:-12}"

export CAMERA_V2_TRT86_PYTHON="${CAMERA_V2_TRT86_PYTHON:-$ROOT/.venv-trt86/bin/python}"
export CAMERA_V2_TRT86_ENGINE="${CAMERA_V2_TRT86_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
export CAMERA_V2_TRT86_SHM_WORKER="${CAMERA_V2_TRT86_SHM_WORKER:-$ROOT/scripts/yolo26_trt86_shm_worker_v5.py}"
RESTORE_HELPER="$ROOT/scripts/restore_cam01_trt86_engine.sh"

# Keep optional identity/LLM work out of the camera hot path until the camera
# acceptance test passes. Global ID utilities remain in the repo for phase two.
export QWEN_REID_ENABLED=0
unset QWEN_REID_URL QWEN_REID_MODEL QWEN_REID_TIMEOUT_SEC || true

fail() { printf 'CAMERA_CLEAN_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }

for plugin in nvurisrcbin tee queue nvstreammux nvmultistreamtiler nvvideoconvert appsink nvtracker nvdsosd nveglglessink fakesink; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing GStreamer/DeepStream plugin: $plugin"
done

GPU_LINE="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -n1 || true)"
[[ -n "$GPU_LINE" ]] || GPU_LINE="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
echo "CAMERA_CLEAN_GPU ${GPU_LINE:-unknown}"

if [[ "$CAMERA_V2_DETECT_ENABLED" == "1" ]]; then
  [[ -x "$CAMERA_V2_TRT86_PYTHON" ]] || fail "TRT86 python missing: $CAMERA_V2_TRT86_PYTHON"
  if [[ ! -s "$CAMERA_V2_TRT86_ENGINE" && -f "$RESTORE_HELPER" ]]; then
    echo "CAMERA_CLEAN_ENGINE recovery=stash/local-search" >&2
    bash "$RESTORE_HELPER" "$CAMERA_V2_TRT86_ENGINE" || true
  fi
  [[ -s "$CAMERA_V2_TRT86_ENGINE" ]] || fail "TRT8.6 engine missing: $CAMERA_V2_TRT86_ENGINE"
  [[ -f "$CAMERA_V2_TRT86_SHM_WORKER" ]] || fail "TRT86 worker missing: $CAMERA_V2_TRT86_SHM_WORKER"

  "$CAMERA_V2_TRT86_PYTHON" - <<'PY'
import sys
import numpy as np
import tensorrt as trt
if not str(trt.__version__).startswith("8.6.1"):
    raise SystemExit(f"TensorRT 8.6.1 required, got {trt.__version__}")
print(f"CAMERA_CLEAN_TRT_ENV python={sys.executable} trt={trt.__version__} numpy={np.__version__}")
PY
fi

MAIN_PYTHON=""
for candidate in "${CAMERA_V2_MAIN_PYTHON:-}" "$ROOT/.venv/bin/python" "$(command -v python3 2>/dev/null || true)"; do
  [[ -n "$candidate" && -x "$candidate" ]] || continue
  if "$candidate" - <<'PY' >/dev/null 2>&1
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: F401
import numpy, yaml, dotenv  # noqa: F401
import services.camera_v2.runtime_quality  # noqa: F401
PY
  then MAIN_PYTHON="$candidate"; break; fi
done
[[ -n "$MAIN_PYTHON" ]] || fail "no Python can import clean Camera V2 quality runtime"

printf '%s\n' \
  "CAMERA_CLEAN_PROFILE source=${CAMERA_V2_SOURCE_FPS}fps rtsp=${CAMERA_V2_RTSP_LATENCY_MS}ms display=${CAMERA_V2_DISPLAY_WIDTH}x${CAMERA_V2_DISPLAY_HEIGHT} wall=${CAMERA_V2_WALL_WIDTH}x${CAMERA_V2_WALL_HEIGHT}" \
  "CAMERA_CLEAN_PROFILE tracker=672x384@${CAMERA_V2_TRACK_FPS}Hz detector=672x384@${CAMERA_V2_DETECT_HZ}Hz/cam conf=${CAMERA_V2_DETECT_CONF} analytics=${CAMERA_V2_ANALYTICS_ENABLED} detector_enabled=${CAMERA_V2_DETECT_ENABLED}" \
  "CAMERA_CLEAN_PIPELINE decode-once->tee->{display/latest,tracker/latest+rate-gate,detector/latest+JIT-gate} display-never-waits-for-analytics=1" \
  "CAMERA_CLEAN_QUALITY detector_nms=1 nvdcf_duplicate_guard=1 detector_cuda_priority=low" \
  "CAMERA_CLEAN_MAIN executable=$MAIN_PYTHON module=services.camera_v2.runtime_quality"

restart_count=0
while true; do
  set +e
  "$MAIN_PYTHON" -u -m services.camera_v2.runtime_quality
  rc=$?
  set -e
  [[ $rc -eq 75 ]] || exit "$rc"
  restart_count=$((restart_count + 1))
  delay=$restart_count
  (( delay > 10 )) && delay=10
  echo "CAMERA_CLEAN_SUPERVISOR restart=$restart_count delay=${delay}s" >&2
  sleep "$delay"
done
