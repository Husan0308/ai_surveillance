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

# Restore the old stable model semantics while keeping the new Pascal-safe wall.
export CAMERA_V2_DETECT_BACKEND=stable-yolo26m
export CAMERA_V2_YOLO_MODEL="${CAMERA_V2_YOLO_MODEL:-yolo26m.pt}"

# Capture stays approximately 16:9 so the CCTV image is not distorted before
# Ultralytics. The model itself owns the old stable 448x704 letterbox target.
export CAMERA_V2_DETECT_WIDTH=704
export CAMERA_V2_DETECT_HEIGHT=400
export CAMERA_V2_YOLO_IMGSZ_WIDTH=704
export CAMERA_V2_YOLO_IMGSZ_HEIGHT=448
export CAMERA_V2_DETECT_CONF=0.10
export CAMERA_V2_DETECT_IOU=0.50
export CAMERA_V2_MAX_DET=50
export CAMERA_V2_MICRO_BATCH=2

# The old stable visual tracker needs sufficiently frequent detector truth. Keep
# the wall dominant, but allow more detector duty than the RF-DETR experiment.
export CAMERA_V2_DETECT_GPU_DUTY=0.34
export CAMERA_V2_DETECT_GPU_DUTY_MIN=0.22
export CAMERA_V2_DETECT_GPU_DUTY_MAX=0.42

# Exact old adaptive-Kalman/Byte semantics plus measured 20-FPS LK motion between
# sparse corrections. Flow can move an existing confirmed track but cannot birth
# or indefinitely refresh one.
export CAMERA_V2_STABLE_HOLD_MS=2800
export CAMERA_V2_STABLE_MEMORY_MS=4800
export CAMERA_V2_STABLE_PREDICTION_MS=420
export CAMERA_V2_STABLE_START_CONF=0.24
export CAMERA_V2_STABLE_NEW_TRACK_CONF=0.18
export CAMERA_V2_STABLE_STRONG_HITS=2
export CAMERA_V2_STABLE_WEAK_HITS=3
export CAMERA_V2_STABLE_BYTE_HIGH=0.24
export CAMERA_V2_STABLE_BYTE_LOW=0.08
export CAMERA_V2_FLOW_MIN_QUALITY=0.30
export CAMERA_V2_FLOW_HARD_AGE_SEC=4.20
export CAMERA_V2_FLOW_GAIN=0.90

# Display envelope only; never feed padding back into detector/tracker truth.
export CAMERA_V2_BOX_SIDE_MARGIN=0.08
export CAMERA_V2_BOX_TOP_MARGIN=0.04
export CAMERA_V2_BOX_BOTTOM_MARGIN=0.10

# Keep enough telemetry to distinguish detector misses from tracker/render misses.
export CAMERA_V2_STABLE_YOLO_LOG_BUDGET=60

echo "SENTINEL_PROFILE detector=YOLO26m person_only=classes[0] capture=704x400 model_imgsz=704x448 conf=0.10 iou=0.50 batch=2 tracker=old-stable-adaptive-kalman-byte flow=continuous-lk camera_path=pascal-safe-analysis-tiler display=egl"

python scripts/preflight_pascal_safe.py
python scripts/preflight_sentinel_ui.py
python scripts/preflight_camera_v2_core.py

exec python -m services.camera_v2.monitor_ui
