#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

# Detector path stays independent from Camera Service. The only detector change is
# restoring weak person candidates so the tracker's low-score recovery stage is real.
export ML_SUBSTREAM_RTSP_TRANSPORT="${ML_SUBSTREAM_RTSP_TRANSPORT:-tcp}"
export ML_SUBSTREAM_RTSP_LATENCY_MS="${ML_SUBSTREAM_RTSP_LATENCY_MS:-80}"
export ML_SUBSTREAM_EXTRA_SURFACES="${ML_SUBSTREAM_EXTRA_SURFACES:-4}"
export ML_SUBSTREAM_STARTUP_STAGGER_SEC="${ML_SUBSTREAM_STARTUP_STAGGER_SEC:-0.35}"
export ML_SUBSTREAM_CAPTURE_TIMEOUT_MS="${ML_SUBSTREAM_CAPTURE_TIMEOUT_MS:-300}"
export ML_SUBSTREAM_MAX_INPUT_AGE_MS="${ML_SUBSTREAM_MAX_INPUT_AGE_MS:-180}"
export ML_SUBSTREAM_PENDING_DEPTH="${ML_SUBSTREAM_PENDING_DEPTH:-4}"
export ML_SUBSTREAM_TOKEN_CAPACITY="${ML_SUBSTREAM_TOKEN_CAPACITY:-3}"
export ML_DETECTOR_CONF="${ML_DETECTOR_CONF:-0.08}"
export ML_DETECTOR_MAX_DET="${ML_DETECTOR_MAX_DET:-20}"
export ML_DETECTOR_TARGET_HZ="${ML_DETECTOR_TARGET_HZ:-2.0}"

# High/low split: weak boxes can continue an existing track but cannot mint a new ID.
export ML_TRACK_LOW_THRESH="${ML_TRACK_LOW_THRESH:-0.08}"
export ML_TRACK_HIGH_THRESH="${ML_TRACK_HIGH_THRESH:-0.25}"
export ML_TRACK_NEW_THRESH="${ML_TRACK_NEW_THRESH:-0.30}"
export ML_TRACK_CONFIRM_HITS="${ML_TRACK_CONFIRM_HITS:-2}"
export ML_TRACK_TENTATIVE_TTL_SEC="${ML_TRACK_TENTATIVE_TTL_SEC:-0.9}"
export ML_TRACK_SHADOW_SEC="${ML_TRACK_SHADOW_SEC:-1.1}"
export ML_TRACK_MAX_LOST_SEC="${ML_TRACK_MAX_LOST_SEC:-5.0}"
export ML_TRACK_APPEARANCE_WEIGHT="${ML_TRACK_APPEARANCE_WEIGHT:-0.22}"
export ML_TRACK_REACQUIRE_THRESH="${ML_TRACK_REACQUIRE_THRESH:-0.12}"
export ML_TRACK_LOW_RECOVERY_THRESH="${ML_TRACK_LOW_RECOVERY_THRESH:-0.10}"
export ML_TRACK_LOW_RECOVERY_SEC="${ML_TRACK_LOW_RECOVERY_SEC:-3.0}"
export ML_TRACK_DUPLICATE_IOU="${ML_TRACK_DUPLICATE_IOU:-0.60}"
export ML_TRACK_LOW_APPEARANCE_WEIGHT="${ML_TRACK_LOW_APPEARANCE_WEIGHT:-0.16}"
export ML_TRACK_LOW_APPEARANCE_FLOOR="${ML_TRACK_LOW_APPEARANCE_FLOOR:-0.45}"
export ML_TRACK_LIVE_DUPLICATE_IOU="${ML_TRACK_LIVE_DUPLICATE_IOU:-0.72}"
export ML_TRACK_LOST_VELOCITY_HALF_LIFE_SEC="${ML_TRACK_LOST_VELOCITY_HALF_LIFE_SEC:-0.9}"

# Keep the proven V4 raw-association / smoothed-render separation.
export ML_TRACK_NESTED_DUPLICATE_IOS="${ML_TRACK_NESTED_DUPLICATE_IOS:-0.82}"
export ML_TRACK_NESTED_DUPLICATE_APP_FLOOR="${ML_TRACK_NESTED_DUPLICATE_APP_FLOOR:-0.58}"
export ML_TRACK_NESTED_DUPLICATE_CENTER_FRAC="${ML_TRACK_NESTED_DUPLICATE_CENTER_FRAC:-0.28}"
export ML_TRACK_RENDER_ANCHOR_ALPHA="${ML_TRACK_RENDER_ANCHOR_ALPHA:-0.72}"
export ML_TRACK_RENDER_SIZE_ALPHA="${ML_TRACK_RENDER_SIZE_ALPHA:-0.20}"
export ML_TRACK_RENDER_RECOVERY_SIZE_ALPHA="${ML_TRACK_RENDER_RECOVERY_SIZE_ALPHA:-0.34}"
export ML_TRACK_RENDER_MAX_SIZE_STEP="${ML_TRACK_RENDER_MAX_SIZE_STEP:-0.28}"
export ML_TRACK_RENDER_VELOCITY_GAIN="${ML_TRACK_RENDER_VELOCITY_GAIN:-0.30}"
# V5 visual acceptance consumes these object rows. Disable explicitly in non-visual runs.
export ML_TRACK_LOG_OBJECTS="${ML_TRACK_LOG_OBJECTS:-1}"

export ML_DETECTOR_TRT86_PYTHON="${ML_DETECTOR_TRT86_PYTHON:-$ROOT/.venv-trt86/bin/python}"
export ML_DETECTOR_TRT86_ENGINE="${ML_DETECTOR_TRT86_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
export ML_DETECTOR_TRT86_WORKER="${ML_DETECTOR_TRT86_WORKER:-$ROOT/scripts/yolo26_trt86_shm_worker_v4.py}"

fail() { printf 'ML_STEP4_V5_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }
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
    raise SystemExit(f"ML_STEP4_V5_PREFLIGHT ERROR: TensorRT 8.6.1 required, got {trt.__version__}")
if sys.prefix == sys.base_prefix:
    raise SystemExit("ML_STEP4_V5_PREFLIGHT ERROR: TRT86 interpreter is not inside venv")
print(f"ML_STEP4_V5_TRT_ENV python={sys.executable} trt={trt.__version__} numpy={np.__version__}", flush=True)
PY

MAIN_PYTHON="${ML_SUBSTREAM_MAIN_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$MAIN_PYTHON" ]] || fail "main python missing: $MAIN_PYTHON"
"$MAIN_PYTHON" - <<'PY'
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: F401
from services.ml_service.app.detector_substream_tracking_v5 import DetectorSubstreamTrackingV5Service  # noqa: F401
print("ML_STEP4_V5_IMPORTS status=OK detector_trt86=1 low_score_recovery=1 v4_box_stability=1 pose=0 gpu_tracker=0", flush=True)
PY

printf '%s\n' \
  "ML_STEP4_V5_PROFILE detector_conf=${ML_DETECTOR_CONF} low=${ML_TRACK_LOW_THRESH} high=${ML_TRACK_HIGH_THRESH} new=${ML_TRACK_NEW_THRESH} target=${ML_DETECTOR_TARGET_HZ}Hz/cam" \
  "ML_STEP4_V5_PROFILE rtsp=${ML_SUBSTREAM_RTSP_LATENCY_MS}ms pending_depth=${ML_SUBSTREAM_PENDING_DEPTH} token_capacity=${ML_SUBSTREAM_TOKEN_CAPACITY}" \
  "ML_STEP4_V5_BOUNDARY camera_service=independent detector=substream tracker=cpu-local render_box=smoothed pose=0 nvdcf=0 reid=0 global_id=0"

exec "$MAIN_PYTHON" -u -m services.ml_service.app.detector_substream_tracking_v5
