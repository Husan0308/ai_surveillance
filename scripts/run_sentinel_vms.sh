#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HEAD_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
echo "SENTINEL_BUILD branch=${BRANCH} head=${HEAD_SHA} expected_ui=2026.08.20-r19-analysis-tiler"

if [[ -n "${DISPLAY:-}" && -z "${QT_QPA_PLATFORM:-}" ]]; then
  export QT_QPA_PLATFORM=xcb
fi
echo "SENTINEL_DISPLAY session=${XDG_SESSION_TYPE:-unknown} qt_platform=${QT_QPA_PLATFORM:-auto} display=${DISPLAY:-unset}"

INHERITED_DETECT="${CAMERA_V2_DETECT_WIDTH:-unset}x${CAMERA_V2_DETECT_HEIGHT:-unset}"
INHERITED_MUX="${CAMERA_V2_FRAME_WIDTH:-unset}x${CAMERA_V2_FRAME_HEIGHT:-unset}"

# Keep the proven six-camera DeepStream/NVDEC display path.
export CAMERA_V2_RTSP_TRANSPORT=tcp
export CAMERA_V2_RTSP_LATENCY_MS=250
export CAMERA_V2_FRAME_WIDTH=2560
export CAMERA_V2_FRAME_HEIGHT=1440

# Exact detector profile from the old Apsidal UI snapshot:
# ui-aspect-ratio-final @ 865bfedf (2026-08-13).
export CAMERA_V2_DETECT_BACKEND=stable-yolo26m
export CAMERA_V2_YOLO_MODEL=yolo26m.pt
export CAMERA_V2_DETECT_WIDTH=704
export CAMERA_V2_DETECT_HEIGHT=448
export CAMERA_V2_YOLO_IMGSZ_WIDTH=704
export CAMERA_V2_YOLO_IMGSZ_HEIGHT=448
export CAMERA_V2_DETECT_CONF=0.06
export CAMERA_V2_DETECT_IOU=0.50
export CAMERA_V2_MAX_DET=50
export CAMERA_V2_MICRO_BATCH=2
export CAMERA_V2_DETECT_STARTUP_DELAY=2.0

# The old UI scheduler was latest-only with one in-flight batch and did not use
# the later adaptive RF-DETR duty governor.  The backend replaces the scheduler
# and uses 300 ms submit / 900 ms result freshness limits.
export CAMERA_V2_DETECT_GPU_DUTY=1.0
export CAMERA_V2_DETECT_GPU_DUTY_MIN=1.0
export CAMERA_V2_DETECT_GPU_DUTY_MAX=1.0

export CAMERA_V2_PASCAL_SAFE=1
export CAMERA_V2_HEATMAP=0
export CAMERA_V2_DISPLAY_BACKEND=egl
export CAMERA_V2_EGL_FAILOVER_SEC=8.0

echo "SENTINEL_PROFILE inherited_mux=${INHERITED_MUX} inherited_detector=${INHERITED_DETECT} effective_mux=2560x1440 rtsp=tcp latency=250ms detector=YOLO26m-old-ui source=ui-aspect-ratio-final@865bfedf capture=704x448 imgsz=704x448 conf=0.06 iou=0.50 max_det=50 batch=2 freshness=300ms-submit/900ms-result roi=CAM05-verify+CAM06-augment tracker=exact-old-ui-kalman-byte flow=OFF reid=OFF nvtracker=OFF overlay=post-tiler-wall-space display=egl->x11-on-zero-render pascal_safe=1 ui=camera-only-2x3-click-fullscreen"

python scripts/preflight_old_ui_detection.py

exec python -m services.camera_v2.monitor_ui
