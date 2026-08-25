#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

# Smoothness-first RTSP profile. DeepStream's RTSP latency is the jitterbuffer
# size; 100 ms is the NVIDIA default and is safer than the previous 50 ms with
# drop-on-latency=true on six live NVR streams.
export CAMERA_V2_RTSP_TRANSPORT="${CAMERA_V2_RTSP_TRANSPORT:-tcp}"
export CAMERA_V2_RTSP_LATENCY_MS="${CAMERA_V2_RTSP_LATENCY_MS:-100}"

export CAMERA_V2_DETECT_WIDTH=672
export CAMERA_V2_DETECT_HEIGHT=384
export CAMERA_V2_MICRO_BATCH=1
export CAMERA_V2_DETECT_ACTIVE_CAMERAS=CAM-01
export CAMERA_V2_DETECT_CONF="${CAMERA_V2_DETECT_CONF:-0.05}"
export CAMERA_V2_DETECT_IOU="${CAMERA_V2_DETECT_IOU:-0.70}"
export CAMERA_V2_MAX_DET="${CAMERA_V2_MAX_DET:-40}"

export CAMERA_V2_DETECT_TARGET_HZ="${CAMERA_V2_DETECT_TARGET_HZ:-2.0}"
export CAMERA_V2_DETECT_MIN_HZ="${CAMERA_V2_DETECT_MIN_HZ:-1.8}"
export CAMERA_V2_DETECT_MAX_HZ="${CAMERA_V2_DETECT_MAX_HZ:-2.3}"
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS="${CAMERA_V2_MAX_DETECT_RESULT_AGE_MS:-350}"

export CAMERA_V2_TRACKER_WIDTH="${CAMERA_V2_TRACKER_WIDTH:-512}"
export CAMERA_V2_TRACKER_HEIGHT="${CAMERA_V2_TRACKER_HEIGHT:-288}"
export CAMERA_V2_MIN_DISPLAY_TRACK_CONF="${CAMERA_V2_MIN_DISPLAY_TRACK_CONF:-0.12}"

export CAMERA_V2_TRT86_PYTHON="${CAMERA_V2_TRT86_PYTHON:-$ROOT/.venv-trt86/bin/python}"
export CAMERA_V2_TRT86_ENGINE="${CAMERA_V2_TRT86_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
export CAMERA_V2_TRT86_SHM_WORKER="${CAMERA_V2_TRT86_SHM_WORKER:-$ROOT/scripts/yolo26_trt86_shm_worker_v3.py}"

export QWEN_REID_ENABLED=0
unset QWEN_REID_URL QWEN_REID_MODEL QWEN_REID_TIMEOUT_SEC || true

fail() {
  printf 'CAM01_TRT86_PREFLIGHT ERROR: %s\n' "$*" >&2
  exit 1
}

[[ -x "$CAMERA_V2_TRT86_PYTHON" ]] || fail "TRT86 python missing/not executable: $CAMERA_V2_TRT86_PYTHON"
[[ -f "$CAMERA_V2_TRT86_ENGINE" ]] || fail "TensorRT engine missing: $CAMERA_V2_TRT86_ENGINE"
[[ -f "$CAMERA_V2_TRT86_SHM_WORKER" ]] || fail "TRT86 worker missing: $CAMERA_V2_TRT86_SHM_WORKER"

"$CAMERA_V2_TRT86_PYTHON" - <<'PY'
import sys
import numpy as np
import tensorrt as trt
if not str(trt.__version__).startswith("8.6.1"):
    raise SystemExit(f"CAM01_TRT86_PREFLIGHT ERROR: TensorRT 8.6.1 required, got {trt.__version__}")
print(f"CAM01_TRT86_PREFLIGHT trt_python={sys.executable} tensorrt={trt.__version__} numpy={np.__version__}", flush=True)
PY

MAIN_PYTHON=""
CANDIDATES=()
if [[ -n "${CAMERA_V2_MAIN_PYTHON:-}" ]]; then CANDIDATES+=("$CAMERA_V2_MAIN_PYTHON"); fi
if [[ -x "$ROOT/.venv/bin/python" ]]; then CANDIDATES+=("$ROOT/.venv/bin/python"); fi
if command -v python3 >/dev/null 2>&1; then CANDIDATES+=("$(command -v python3)"); fi
if command -v python >/dev/null 2>&1; then CANDIDATES+=("$(command -v python)"); fi

for candidate in "${CANDIDATES[@]}"; do
  [[ -x "$candidate" ]] || continue
  if "$candidate" - <<'PY' >/dev/null 2>&1
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: F401
import numpy  # noqa: F401
import yaml  # noqa: F401
import dotenv  # noqa: F401
import services.camera_v2.person_tracking_trt86_audited  # noqa: F401
PY
  then
    MAIN_PYTHON="$candidate"
    break
  fi
done

[[ -n "$MAIN_PYTHON" ]] || fail "no Python can import Camera V2 audited runtime"

printf '%s\n' \
  "CAM01_TRT86_PROFILE engine=$(basename "$CAMERA_V2_TRT86_ENGINE") input=672x384/b1/fp32 active=CAM-01 target=${CAMERA_V2_DETECT_TARGET_HZ}Hz max_result_age>=${CAMERA_V2_MAX_DETECT_RESULT_AGE_MS}ms tracker=${CAMERA_V2_TRACKER_WIDTH}x${CAMERA_V2_TRACKER_HEIGHT} qwen=0" \
  "CAM01_TRT86_PIPELINE backend=trt86-sidecar-shm-bgr-v3 capture=jit-latest no_prefetch=1 appsink_async=0 queue_depth=1 letterbox=672x378+3+3 nvdcf=per-frame rtsp=${CAMERA_V2_RTSP_TRANSPORT}/${CAMERA_V2_RTSP_LATENCY_MS}ms" \
  "CAM01_TRT86_MAIN_PYTHON executable=$MAIN_PYTHON"

exec "$MAIN_PYTHON" -u -m services.camera_v2.person_tracking_trt86_audited
