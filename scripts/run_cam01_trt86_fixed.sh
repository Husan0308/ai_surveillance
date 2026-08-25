#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

export CAMERA_V2_RTSP_TRANSPORT="${CAMERA_V2_RTSP_TRANSPORT:-tcp}"
export CAMERA_V2_RTSP_LATENCY_MS="${CAMERA_V2_RTSP_LATENCY_MS:-50}"

export CAMERA_V2_DETECT_WIDTH=672
export CAMERA_V2_DETECT_HEIGHT=384
export CAMERA_V2_MICRO_BATCH=1
export CAMERA_V2_DETECT_ACTIVE_CAMERAS=CAM-01
export CAMERA_V2_DETECT_CONF="${CAMERA_V2_DETECT_CONF:-0.05}"
# YOLO26 E2E output is already NMS-free. This is kept only because inherited
# tracking/status code reads the setting; the TRT86 worker does not run NMS.
export CAMERA_V2_DETECT_IOU="${CAMERA_V2_DETECT_IOU:-0.70}"
export CAMERA_V2_MAX_DET="${CAMERA_V2_MAX_DET:-40}"

export CAMERA_V2_DETECT_TARGET_HZ="${CAMERA_V2_DETECT_TARGET_HZ:-2.0}"
export CAMERA_V2_DETECT_MIN_HZ="${CAMERA_V2_DETECT_MIN_HZ:-1.8}"
export CAMERA_V2_DETECT_MAX_HZ="${CAMERA_V2_DETECT_MAX_HZ:-2.3}"

# Correctness floor: the measured TRT86 round-trip is ~154-190 ms, so 160 ms
# can discard valid detector refreshes before NvDCF sees them.
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS="${CAMERA_V2_MAX_DETECT_RESULT_AGE_MS:-350}"

export CAMERA_V2_TRACKER_WIDTH="${CAMERA_V2_TRACKER_WIDTH:-512}"
export CAMERA_V2_TRACKER_HEIGHT="${CAMERA_V2_TRACKER_HEIGHT:-288}"
export CAMERA_V2_MIN_DISPLAY_TRACK_CONF="${CAMERA_V2_MIN_DISPLAY_TRACK_CONF:-0.12}"

export CAMERA_V2_TRT86_PYTHON="${CAMERA_V2_TRT86_PYTHON:-$ROOT/.venv-trt86/bin/python}"
export CAMERA_V2_TRT86_ENGINE="${CAMERA_V2_TRT86_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
export CAMERA_V2_TRT86_SHM_WORKER="${CAMERA_V2_TRT86_SHM_WORKER:-$ROOT/scripts/yolo26_trt86_shm_worker.py}"

export QWEN_REID_ENABLED=0
unset QWEN_REID_URL QWEN_REID_MODEL QWEN_REID_TIMEOUT_SEC || true

fail() {
  printf 'CAM01_TRT86_PREFLIGHT ERROR: %s\n' "$*" >&2
  exit 1
}

[[ -x "$CAMERA_V2_TRT86_PYTHON" ]] || \
  fail "TRT86 python is missing/not executable: $CAMERA_V2_TRT86_PYTHON"
[[ -f "$CAMERA_V2_TRT86_ENGINE" ]] || \
  fail "TensorRT engine is missing: $CAMERA_V2_TRT86_ENGINE"
[[ -f "$CAMERA_V2_TRT86_SHM_WORKER" ]] || \
  fail "TRT86 SHM worker is missing: $CAMERA_V2_TRT86_SHM_WORKER"

# Verify the isolated TensorRT environment before starting six camera streams.
"$CAMERA_V2_TRT86_PYTHON" - <<'PY' || exit 1
import sys
try:
    import numpy as np
    import tensorrt as trt
except Exception as exc:
    raise SystemExit(f"CAM01_TRT86_PREFLIGHT ERROR: TRT86 env import failed: {type(exc).__name__}: {exc}")
if not str(trt.__version__).startswith("8.6.1"):
    raise SystemExit(f"CAM01_TRT86_PREFLIGHT ERROR: TensorRT 8.6.1 required, got {trt.__version__}")
print(f"CAM01_TRT86_PREFLIGHT trt_python={sys.executable} tensorrt={trt.__version__} numpy={np.__version__}", flush=True)
PY

# The main DeepStream process must use the project/system Python, not the
# TensorRT-only venv. Ubuntu guarantees python3; a bare `python` command may not
# exist. Prefer an explicit override, then project .venv, then python3/python.
MAIN_PYTHON=""
CANDIDATES=()
if [[ -n "${CAMERA_V2_MAIN_PYTHON:-}" ]]; then
  CANDIDATES+=("$CAMERA_V2_MAIN_PYTHON")
fi
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  CANDIDATES+=("$ROOT/.venv/bin/python")
fi
if command -v python3 >/dev/null 2>&1; then
  CANDIDATES+=("$(command -v python3)")
fi
if command -v python >/dev/null 2>&1; then
  CANDIDATES+=("$(command -v python)")
fi

for candidate in "${CANDIDATES[@]}"; do
  [[ -x "$candidate" ]] || continue
  if "$candidate" - <<'PY' >/dev/null 2>&1
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: F401
import numpy  # noqa: F401
import yaml  # noqa: F401
import dotenv  # noqa: F401
import services.camera_v2.person_tracking_trt86  # noqa: F401
PY
  then
    MAIN_PYTHON="$candidate"
    break
  fi
done

if [[ -z "$MAIN_PYTHON" ]]; then
  printf '%s\n' \
    'CAM01_TRT86_PREFLIGHT ERROR: no Python interpreter can import the Camera V2 runtime.' \
    'Checked CAMERA_V2_MAIN_PYTHON, .venv/bin/python, python3, and python.' \
    'The interpreter needs gi/GStreamer, numpy, PyYAML, python-dotenv, and this project.' >&2
  exit 1
fi

printf '%s\n' \
  "CAM01_TRT86_PROFILE engine=$(basename "$CAMERA_V2_TRT86_ENGINE") input=672x384/b1/fp32 active=CAM-01 target=${CAMERA_V2_DETECT_TARGET_HZ}Hz max_result_age>=${CAMERA_V2_MAX_DETECT_RESULT_AGE_MS}ms tracker=${CAMERA_V2_TRACKER_WIDTH}x${CAMERA_V2_TRACKER_HEIGHT} qwen=0" \
  "CAM01_TRT86_PIPELINE backend=trt86-sidecar-shm-bgr base64=0 jpeg=0 queue_depth=1 nvdcf=per-frame" \
  "CAM01_TRT86_MAIN_PYTHON executable=$MAIN_PYTHON"

exec "$MAIN_PYTHON" -u -m services.camera_v2.person_tracking_trt86
