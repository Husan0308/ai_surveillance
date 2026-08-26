#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

export ML_SUBSTREAM_RTSP_TRANSPORT="${ML_SUBSTREAM_RTSP_TRANSPORT:-tcp}"
export ML_SUBSTREAM_RTSP_LATENCY_MS="${ML_SUBSTREAM_RTSP_LATENCY_MS:-80}"
export ML_SUBSTREAM_EXTRA_SURFACES="${ML_SUBSTREAM_EXTRA_SURFACES:-4}"
export ML_SUBSTREAM_STARTUP_STAGGER_SEC="${ML_SUBSTREAM_STARTUP_STAGGER_SEC:-0.35}"
export ML_SUBSTREAM_CAPTURE_TIMEOUT_MS="${ML_SUBSTREAM_CAPTURE_TIMEOUT_MS:-300}"
export ML_SUBSTREAM_MAX_INPUT_AGE_MS="${ML_SUBSTREAM_MAX_INPUT_AGE_MS:-180}"
export ML_DETECTOR_CONF="${ML_DETECTOR_CONF:-0.18}"
export ML_DETECTOR_MAX_DET="${ML_DETECTOR_MAX_DET:-20}"
export ML_DETECTOR_TARGET_HZ="${ML_DETECTOR_TARGET_HZ:-2.0}"
export ML_DETECTOR_TRT86_PYTHON="${ML_DETECTOR_TRT86_PYTHON:-$ROOT/.venv-trt86/bin/python}"
export ML_DETECTOR_TRT86_ENGINE="${ML_DETECTOR_TRT86_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
export ML_DETECTOR_TRT86_WORKER="${ML_DETECTOR_TRT86_WORKER:-$ROOT/scripts/yolo26_trt86_shm_worker_v4.py}"

fail() { printf 'ML_SUBSTREAM_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }

for plugin in nvurisrcbin queue nvvideoconvert capsfilter appsink; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing GStreamer/DeepStream plugin: $plugin"
done

[[ -x "$ML_DETECTOR_TRT86_PYTHON" ]] || fail "TRT8.6 python missing: $ML_DETECTOR_TRT86_PYTHON"
[[ -s "$ML_DETECTOR_TRT86_ENGINE" ]] || fail "TRT8.6 engine missing: $ML_DETECTOR_TRT86_ENGINE"
[[ -f "$ML_DETECTOR_TRT86_WORKER" ]] || fail "TRT8.6 worker missing: $ML_DETECTOR_TRT86_WORKER"

"$ML_DETECTOR_TRT86_PYTHON" -I - <<'PY'
import sys
import numpy as np
import tensorrt as trt
if not str(trt.__version__).startswith("8.6.1"):
    raise SystemExit(f"ML_SUBSTREAM_PREFLIGHT ERROR: TensorRT 8.6.1 required, got {trt.__version__}")
if sys.prefix == sys.base_prefix:
    raise SystemExit(
        "ML_SUBSTREAM_PREFLIGHT ERROR: TRT86 interpreter is not inside venv: "
        f"executable={sys.executable} prefix={sys.prefix} base_prefix={sys.base_prefix}"
    )
print(
    "ML_SUBSTREAM_TRT_ENV "
    f"python={sys.executable} prefix={sys.prefix} base_prefix={sys.base_prefix} "
    f"trt={trt.__version__} numpy={np.__version__} numpy_file={np.__file__}",
    flush=True,
)
PY

MAIN_PYTHON="${ML_SUBSTREAM_MAIN_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$MAIN_PYTHON" ]] || fail "main python missing: $MAIN_PYTHON"
"$MAIN_PYTHON" - <<'PY'
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: F401
import numpy, yaml, dotenv  # noqa: F401
from services.ml_service.app.detector_substream_paced import DetectorSubstreamPacedService  # noqa: F401
print("ML_SUBSTREAM_IMPORTS status=OK pts_paced_ready_first=1 live_preroll_safe=1", flush=True)
PY

printf '%s\n' \
  "ML_SUBSTREAM_PROFILE source=Hikvision-substream-direct rtsp=${ML_SUBSTREAM_RTSP_LATENCY_MS}ms extra_surfaces=${ML_SUBSTREAM_EXTRA_SURFACES}" \
  "ML_SUBSTREAM_PROFILE detector=TRT8.6/672x384 target=${ML_DETECTOR_TARGET_HZ}Hz/cam conf=${ML_DETECTOR_CONF} max_det=${ML_DETECTOR_MAX_DET}" \
  "ML_SUBSTREAM_BOUNDARY main_stream=0 camera_service_shm=0 tracker=0 api=0 ui=0 sparse_gate_before_convert=1 scheduler=pts-paced-ready-first blocking_capture_wait=0"

exec "$MAIN_PYTHON" -u -m services.ml_service.app.detector_substream_paced
