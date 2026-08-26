#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

export ML_DETECTOR_SHM_DIR="${ML_DETECTOR_SHM_DIR:-/dev/shm/ai_surveillance}"
export ML_DETECTOR_CAMERAS="${ML_DETECTOR_CAMERAS:-CAM-01,CAM-02,CAM-03,CAM-04,CAM-05,CAM-06}"
export ML_DETECTOR_CONF="${ML_DETECTOR_CONF:-0.18}"
export ML_DETECTOR_MAX_DET="${ML_DETECTOR_MAX_DET:-20}"
export ML_DETECTOR_MAX_INPUT_AGE_MS="${ML_DETECTOR_MAX_INPUT_AGE_MS:-300}"
export ML_DETECTOR_ATTACH_TIMEOUT_SEC="${ML_DETECTOR_ATTACH_TIMEOUT_SEC:-30}"
export ML_DETECTOR_TRT86_PYTHON="${ML_DETECTOR_TRT86_PYTHON:-$ROOT/.venv-trt86/bin/python}"
export ML_DETECTOR_TRT86_ENGINE="${ML_DETECTOR_TRT86_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
export ML_DETECTOR_TRT86_WORKER="${ML_DETECTOR_TRT86_WORKER:-$ROOT/scripts/yolo26_trt86_shm_worker_v4.py}"

fail() { printf 'ML_DETECTOR_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }

[[ -x "$ML_DETECTOR_TRT86_PYTHON" ]] || fail "TRT8.6 python missing: $ML_DETECTOR_TRT86_PYTHON"
[[ -s "$ML_DETECTOR_TRT86_ENGINE" ]] || fail "TRT8.6 engine missing: $ML_DETECTOR_TRT86_ENGINE"
[[ -f "$ML_DETECTOR_TRT86_WORKER" ]] || fail "TRT8.6 worker missing: $ML_DETECTOR_TRT86_WORKER"

# Run the exact same venv path that the detector child will use. Do not replace
# this path with readlink -f: venv/bin/python is commonly a symlink and the venv
# identity is carried by invoking that symlink path, not the base interpreter.
"$ML_DETECTOR_TRT86_PYTHON" -I - <<'PY'
import sys
import numpy as np
import tensorrt as trt
if not str(trt.__version__).startswith("8.6.1"):
    raise SystemExit(f"ML_DETECTOR_PREFLIGHT ERROR: TensorRT 8.6.1 required, got {trt.__version__}")
if sys.prefix == sys.base_prefix:
    raise SystemExit(
        "ML_DETECTOR_PREFLIGHT ERROR: TRT86 interpreter is not running inside a virtual environment: "
        f"executable={sys.executable} prefix={sys.prefix} base_prefix={sys.base_prefix}"
    )
print(
    "ML_DETECTOR_TRT_ENV "
    f"python={sys.executable} prefix={sys.prefix} base_prefix={sys.base_prefix} "
    f"trt={trt.__version__} numpy={np.__version__} numpy_file={np.__file__}",
    flush=True,
)
PY

MAIN_PYTHON="${ML_DETECTOR_MAIN_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$MAIN_PYTHON" ]] || fail "main python missing: $MAIN_PYTHON"
"$MAIN_PYTHON" - <<'PY'
import numpy
from services.camera_service.app.shm_frame import LatestFrameMmapReader
from services.ml_service.app.detector_only_shm import DetectorOnlyShmService
print("ML_DETECTOR_IMPORTS status=OK", flush=True)
PY

printf '%s\n' \
  "ML_DETECTOR_ARCH camera_rtsp=0 camera_nvdec=0 deepstream=0 tracker=0 api=0 ui=0 input=SHM-latest output=detections-only" \
  "ML_DETECTOR_INPUT dir=$ML_DETECTOR_SHM_DIR cameras=$ML_DETECTOR_CAMERAS producer_expected=2Hz geometry=672x378x3" \
  "ML_DETECTOR_MODEL engine=$ML_DETECTOR_TRT86_ENGINE conf=$ML_DETECTOR_CONF max_det=$ML_DETECTOR_MAX_DET"

exec "$MAIN_PYTHON" -u -m services.ml_service.app.detector_only_shm
