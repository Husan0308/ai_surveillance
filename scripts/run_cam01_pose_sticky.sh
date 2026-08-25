#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

# Stable six-camera DeepStream ingest/display profile.
export CAMERA_V2_RTSP_TRANSPORT=tcp
export CAMERA_V2_RTSP_LATENCY_MS=100
export CAMERA_V2_FRAME_WIDTH=2560
export CAMERA_V2_FRAME_HEIGHT=1440
export CAMERA_V2_WALL_WIDTH=1920
export CAMERA_V2_WALL_HEIGHT=720

# Restore the historical pose analysis geometry. Do NOT reuse the 672x384 TRT
# sidecar capture here: pose/keypoint quality on distant/seated people benefits
# from retaining the 1280x720 analysis frame before Ultralytics letterboxes to 832.
# The capture gate is before nvvideoconvert, so this conversion/copy happens only
# for the requested ~1 Hz detector frame, not for every display frame.
export CAMERA_V2_DETECT_WIDTH=1280
export CAMERA_V2_DETECT_HEIGHT=720
export CAMERA_V2_MICRO_BATCH=1
export CAMERA_V2_DETECT_ACTIVE_CAMERAS=CAM-01
export CAMERA_V2_DETECT_CONF=0.05
export CAMERA_V2_DETECT_IOU=0.80
export CAMERA_V2_MAX_DET=50
export CAMERA_V2_DETECT_STARTUP_DELAY=0.5

export CAMERA_V2_POSE_IMGSZ=832
export CAMERA_V2_POSE_CONF=0.05
export CAMERA_V2_POSE_IOU=0.80
if [[ -f "$ROOT/yolo26s-pose.pt" ]]; then
  export CAMERA_V2_POSE_MODEL="$ROOT/yolo26s-pose.pt"
else
  unset CAMERA_V2_POSE_MODEL || true
fi

# GTX 1050 Ti pose refresh. NvDCF owns every frame between these observations.
# Use deterministic values for the baseline test rather than inheriting stale
# shell overrides from previous experiments.
export CAMERA_V2_DETECT_TARGET_HZ=1.0
export CAMERA_V2_DETECT_MIN_HZ=1.0
export CAMERA_V2_DETECT_MAX_HZ=1.0
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS=800
export CAMERA_V2_POSE_MAX_PROJECTION_S=0.45
export CAMERA_V2_POSE_PROJECTION_GAIN=0.82
export CAMERA_V2_EMPTY_CONFIRM_MISSES=3

# Sticky local NvDCF. Global ID is deliberately OFF until this path is green.
export CAMERA_V2_TRACKER_WIDTH=512
export CAMERA_V2_TRACKER_HEIGHT=288
export CAMERA_V2_MIN_DISPLAY_TRACK_CONF=0.08
export CAMERA_V2_TRACK_BOX_SIDE_MARGIN=0.00
export CAMERA_V2_TRACK_BOX_TOP_MARGIN=0.00
export CAMERA_V2_TRACK_BOX_BOTTOM_MARGIN=0.00
export CAMERA_V2_DEDUP_IOU=0.82
export CAMERA_V2_DEDUP_CONTAINMENT=0.94

export QWEN_REID_ENABLED=0
unset QWEN_REID_URL QWEN_REID_MODEL QWEN_REID_TIMEOUT_SEC || true
unset CAMERA_V2_TRT86_ENGINE CAMERA_V2_TRT86_PYTHON CAMERA_V2_TRT86_WORKER CAMERA_V2_TRT86_SHM_WORKER || true

fail() {
  printf 'CAM01_POSE_PREFLIGHT ERROR: %s\n' "$*" >&2
  exit 1
}

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
import torch
import ultralytics  # noqa: F401
import services.camera_v2.person_tracking_pose_sticky  # noqa: F401
assert torch.cuda.is_available()
PY
  then
    MAIN_PYTHON="$candidate"
    break
  fi
done

[[ -n "$MAIN_PYTHON" ]] || fail "no Python can import GStreamer + CUDA PyTorch + Ultralytics pose runtime"

"$MAIN_PYTHON" - <<'PY'
import torch
import ultralytics
print(
    f"CAM01_POSE_PREFLIGHT python={__import__('sys').executable} "
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"gpu={torch.cuda.get_device_name(0)} ultralytics={ultralytics.__version__}",
    flush=True,
)
PY

printf '%s\n' \
  "CAM01_POSE_PROFILE detector=YOLO26s-pose imgsz=${CAMERA_V2_POSE_IMGSZ} capture=${CAMERA_V2_DETECT_WIDTH}x${CAMERA_V2_DETECT_HEIGHT} conf=${CAMERA_V2_POSE_CONF} iou=${CAMERA_V2_POSE_IOU} active=CAM-01" \
  "CAM01_POSE_PROFILE tracker=NvDCF@${CAMERA_V2_TRACKER_WIDTH}x${CAMERA_V2_TRACKER_HEIGHT} target=${CAMERA_V2_DETECT_TARGET_HZ}Hz max_result_age=${CAMERA_V2_MAX_DETECT_RESULT_AGE_MS}ms empty_confirm=${CAMERA_V2_EMPTY_CONFIRM_MISSES} probation=0 shadow=50" \
  "CAM01_POSE_PIPELINE capture=jit-one-shot no_prefetch=1 pose-validates-person nvdcf-per-frame=1 global-id=off rtsp=${CAMERA_V2_RTSP_TRANSPORT}/${CAMERA_V2_RTSP_LATENCY_MS}ms"

exec "$MAIN_PYTHON" -u -m services.camera_v2.person_tracking_pose_sticky
