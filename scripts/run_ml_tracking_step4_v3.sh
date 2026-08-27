#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

# Frozen Step 3 detector defaults -- do not retune in Step 4.
export ML_SUBSTREAM_RTSP_TRANSPORT="${ML_SUBSTREAM_RTSP_TRANSPORT:-tcp}"
export ML_SUBSTREAM_RTSP_LATENCY_MS="${ML_SUBSTREAM_RTSP_LATENCY_MS:-80}"
export ML_SUBSTREAM_EXTRA_SURFACES="${ML_SUBSTREAM_EXTRA_SURFACES:-4}"
export ML_SUBSTREAM_STARTUP_STAGGER_SEC="${ML_SUBSTREAM_STARTUP_STAGGER_SEC:-0.35}"
export ML_SUBSTREAM_CAPTURE_TIMEOUT_MS="${ML_SUBSTREAM_CAPTURE_TIMEOUT_MS:-300}"
export ML_SUBSTREAM_MAX_INPUT_AGE_MS="${ML_SUBSTREAM_MAX_INPUT_AGE_MS:-180}"
export ML_SUBSTREAM_PENDING_DEPTH="${ML_SUBSTREAM_PENDING_DEPTH:-4}"
export ML_SUBSTREAM_TOKEN_CAPACITY="${ML_SUBSTREAM_TOKEN_CAPACITY:-3}"
export ML_DETECTOR_CONF="${ML_DETECTOR_CONF:-0.18}"
export ML_DETECTOR_MAX_DET="${ML_DETECTOR_MAX_DET:-20}"
export ML_DETECTOR_TARGET_HZ="${ML_DETECTOR_TARGET_HZ:-2.0}"

# Step 4 v3 local tracking defaults. Visible shadow stays short; dormant recovery is longer.
export ML_TRACK_LOW_THRESH="${ML_TRACK_LOW_THRESH:-0.18}"
export ML_TRACK_HIGH_THRESH="${ML_TRACK_HIGH_THRESH:-0.30}"
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
export ML_TRACK_LOG_OBJECTS="${ML_TRACK_LOG_OBJECTS:-0}"

export ML_DETECTOR_TRT86_PYTHON="${ML_DETECTOR_TRT86_PYTHON:-$ROOT/.venv-trt86/bin/python}"
export ML_DETECTOR_TRT86_ENGINE="${ML_DETECTOR_TRT86_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
export ML_DETECTOR_TRT86_WORKER="${ML_DETECTOR_TRT86_WORKER:-$ROOT/scripts/yolo26_trt86_shm_worker_v4.py}"

fail() { printf 'ML_STEP4_V3_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }

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
    raise SystemExit(f"ML_STEP4_V3_PREFLIGHT ERROR: TensorRT 8.6.1 required, got {trt.__version__}")
if sys.prefix == sys.base_prefix:
    raise SystemExit(
        "ML_STEP4_V3_PREFLIGHT ERROR: TRT86 interpreter is not inside venv: "
        f"executable={sys.executable} prefix={sys.prefix} base_prefix={sys.base_prefix}"
    )
print(
    "ML_STEP4_V3_TRT_ENV "
    f"python={sys.executable} prefix={sys.prefix} base_prefix={sys.base_prefix} "
    f"trt={trt.__version__} numpy={np.__version__}",
    flush=True,
)
PY

MAIN_PYTHON="${ML_SUBSTREAM_MAIN_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$MAIN_PYTHON" ]] || fail "main python missing: $MAIN_PYTHON"
"$MAIN_PYTHON" - <<'PY'
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: F401
import numpy  # noqa: F401
from services.ml_service.app.local_tracker_sparse_v3 import ObservationRecoveryPersonTracker  # noqa: F401
from services.ml_service.app.detector_substream_tracking_v3 import DetectorSubstreamTrackingV3Service  # noqa: F401
print(
    "ML_STEP4_V3_IMPORTS status=OK detector_v14_frozen=1 cpu_tracker=1 "
    "observation_anchor=1 lost_velocity_decay=1 low_hijack_guard=1 live_duplicate_veto=1 gpu_tracker=0",
    flush=True,
)
PY

printf '%s\n' \
  "ML_STEP4_V3_PROFILE detector=V14-token3 target=${ML_DETECTOR_TARGET_HZ}Hz/cam rtsp=${ML_SUBSTREAM_RTSP_LATENCY_MS}ms pending_depth=${ML_SUBSTREAM_PENDING_DEPTH}" \
  "ML_STEP4_V3_PROFILE tracker=CPU-observation-recovery max_lost=${ML_TRACK_MAX_LOST_SEC}s low_recovery=${ML_TRACK_LOW_RECOVERY_SEC}s shadow=${ML_TRACK_SHADOW_SEC}s low_app=${ML_TRACK_LOW_APPEARANCE_WEIGHT}/${ML_TRACK_LOW_APPEARANCE_FLOOR} live_duplicate_iou=${ML_TRACK_LIVE_DUPLICATE_IOU}" \
  "ML_STEP4_V3_BOUNDARY camera_service=independent main_stream=0 nvdcf=0 gpu_tracker=0 reid=0 global_id=0 api=0 ui=0"

exec "$MAIN_PYTHON" -u -m services.ml_service.app.detector_substream_tracking_v3
