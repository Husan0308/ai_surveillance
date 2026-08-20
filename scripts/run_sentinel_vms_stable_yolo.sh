#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HEAD_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
echo "SENTINEL_STABLE_YOLO_BUILD branch=${BRANCH} head=${HEAD_SHA}"

if [[ -n "${DISPLAY:-}" && -z "${QT_QPA_PLATFORM:-}" ]]; then
  export QT_QPA_PLATFORM=xcb
fi

# Same proven camera/display path as production.
export CAMERA_V2_RTSP_TRANSPORT=tcp
export CAMERA_V2_RTSP_LATENCY_MS=250
export CAMERA_V2_FRAME_WIDTH=2560
export CAMERA_V2_FRAME_HEIGHT=1440
export CAMERA_V2_PASCAL_SAFE=1
export CAMERA_V2_HEATMAP=0
export CAMERA_V2_DISPLAY_BACKEND=egl
export CAMERA_V2_EGL_FAILOVER_SEC=8.0

# Important: do NOT install the RF-DETR/flow monkey patches. CameraDetectionV2
# keeps the built-in YOLO worker which already uses classes=[0] at inference.
export CAMERA_V2_DETECT_BACKEND=stable-yolo26m
export CAMERA_V2_YOLO_MODEL="${CAMERA_V2_YOLO_MODEL:-yolo26m.pt}"

# Exact detector-side values from agent/stable-detection-ui-baseline.
export CAMERA_V2_DETECT_WIDTH=704
export CAMERA_V2_DETECT_HEIGHT=448
export CAMERA_V2_DETECT_CONF=0.10
export CAMERA_V2_DETECT_IOU=0.50
export CAMERA_V2_MAX_DET=50
export CAMERA_V2_MICRO_BATCH=2

# Keep inference subordinate to the camera wall on the GTX 1050 Ti. These only
# control scheduling; they do not alter YOLO's person decision or thresholds.
export CAMERA_V2_DETECT_GPU_DUTY=0.24
export CAMERA_V2_DETECT_GPU_DUTY_MIN=0.12
export CAMERA_V2_DETECT_GPU_DUTY_MAX=0.30

# Baseline box presentation only. No optical flow, no NvDCF, no ReID.
export CAMERA_V2_BOX_SIDE_MARGIN=0.08
export CAMERA_V2_BOX_TOP_MARGIN=0.04
export CAMERA_V2_BOX_BOTTOM_MARGIN=0.10
export CAMERA_V2_BOX_MAX_AGE=1.6
export CAMERA_V2_BOX_MAX_PREDICT=0.55

echo "SENTINEL_PROFILE detector=YOLO26m person_only=classes[0] input=704x448 conf=0.10 iou=0.50 batch=2 tracker=baseline-motion flow=0 nvtracker=0 camera_path=pascal-safe-analysis-tiler display=egl"

python scripts/preflight_pascal_safe.py
python scripts/preflight_sentinel_ui.py
python scripts/preflight_camera_v2_core.py

exec python -m services.camera_v2.monitor_ui
