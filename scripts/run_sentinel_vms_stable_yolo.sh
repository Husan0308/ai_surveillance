#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HEAD_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
echo "SENTINEL_STABLE_YOLO_BUILD branch=${BRANCH} head=${HEAD_SHA}"

if [[ -n "${DISPLAY:-}" && -z "${QT_QPA_PLATFORM:-}" ]]; then
  export QT_QPA_PLATFORM=xcb
fi

# Keep the already-proven six-camera Pascal-safe display/RTSP path unchanged.
export CAMERA_V2_RTSP_TRANSPORT=tcp
export CAMERA_V2_RTSP_LATENCY_MS=250
export CAMERA_V2_FRAME_WIDTH=2560
export CAMERA_V2_FRAME_HEIGHT=1440
export CAMERA_V2_PASCAL_SAFE=1
export CAMERA_V2_HEATMAP=0
export CAMERA_V2_DISPLAY_BACKEND=egl
export CAMERA_V2_EGL_FAILOVER_SEC=8.0

# DETECTION TRUTH MODE: YOLO only. No tracker, optical flow, ReID or identity.
export CAMERA_V2_DETECT_BACKEND=stable-yolo26m
export CAMERA_V2_YOLO_MODEL="${CAMERA_V2_YOLO_MODEL:-yolo26m.pt}"

# Match the earlier stable person detector semantics. Capture stays close to 16:9;
# Ultralytics owns the model letterbox/preprocess and returns xyxy in capture space.
export CAMERA_V2_DETECT_WIDTH=704
export CAMERA_V2_DETECT_HEIGHT=400
export CAMERA_V2_YOLO_IMGSZ_WIDTH=704
export CAMERA_V2_YOLO_IMGSZ_HEIGHT=448
export CAMERA_V2_DETECT_CONF=0.06
export CAMERA_V2_DETECT_IOU=0.50
export CAMERA_V2_MAX_DET=50
export CAMERA_V2_MICRO_BATCH=2

# Detection is the only subsystem under test. Give it enough cadence to make raw
# boxes continuously observable while the display queue remains leaky/nonblocking.
export CAMERA_V2_DETECT_GPU_DUTY=0.46
export CAMERA_V2_DETECT_GPU_DUTY_MIN=0.38
export CAMERA_V2_DETECT_GPU_DUTY_MAX=0.58

# Latest raw detector truth is shown directly. This hold only bridges round-robin
# inference gaps; it does not predict or move a box.
export CAMERA_V2_RAW_BOX_MAX_AGE=3.00
export CAMERA_V2_RAW_BOX_SIDE_MARGIN=0.05
export CAMERA_V2_RAW_BOX_TOP_MARGIN=0.03
export CAMERA_V2_RAW_BOX_BOTTOM_MARGIN=0.07

# Detailed detector/meta telemetry remains enabled until bbox is visibly proven.
export CAMERA_V2_STABLE_YOLO_LOG_BUDGET=120

echo "SENTINEL_PROFILE detector=YOLO26m person_only=classes[0] mode=one-to-many-nms capture=704x400 model_imgsz=704x448 conf=0.06 iou=0.50 batch=2 tracker=OFF flow=OFF raw_meta=detector-result camera_path=pascal-safe-analysis-tiler display=egl"

python scripts/preflight_pascal_safe.py
python scripts/preflight_sentinel_ui.py
python scripts/preflight_camera_v2_core.py

exec python -m services.camera_v2.monitor_ui
