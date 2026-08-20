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

# Proven six-camera RTSP/NVDEC path.
export CAMERA_V2_RTSP_TRANSPORT=tcp
export CAMERA_V2_RTSP_LATENCY_MS=250
export CAMERA_V2_FRAME_WIDTH=2560
export CAMERA_V2_FRAME_HEIGHT=1440

# RF-DETR-S ONLY. Keep the capture tile 16:9 so CCTV geometry is preserved,
# then let RF-DETR resize internally to the published Small 512x512 operating
# point. Returned boxes are in the original 672x384 capture coordinates and
# CameraDetectionV2 scales those to the 2560x1440 mux frame before metadata.
export CAMERA_V2_DETECT_BACKEND=rfdetr-s
export CAMERA_V2_DETECT_WIDTH=672
export CAMERA_V2_DETECT_HEIGHT=384
export CAMERA_V2_RFDETR_MODEL_WIDTH=512
export CAMERA_V2_RFDETR_MODEL_HEIGHT=512
export CAMERA_V2_DETECT_CONF=0.18
export CAMERA_V2_MAX_DET=40
export CAMERA_V2_MICRO_BATCH=1

# Detector-only bring-up: frequent enough to refresh each of six cameras while
# the existing adaptive wall-p95 governor still protects display throughput.
export CAMERA_V2_DETECT_GPU_DUTY=0.28
export CAMERA_V2_DETECT_GPU_DUTY_MIN=0.20
export CAMERA_V2_DETECT_GPU_DUTY_MAX=0.34

# No tracker, no prediction and no optical flow. Repeating the latest real
# RF-DETR result for a short bounded interval only prevents the box blinking
# between sparse round-robin detector calls.
export CAMERA_V2_RFDETR_RAW_HOLD_SEC=2.80
export CAMERA_V2_RFDETR_BOX_SIDE_MARGIN=0.04
export CAMERA_V2_RFDETR_BOX_TOP_MARGIN=0.03
export CAMERA_V2_RFDETR_BOX_BOTTOM_MARGIN=0.06
export CAMERA_V2_RFDETR_TRUTH_LOG_BUDGET=96

export CAMERA_V2_PASCAL_SAFE=1
export CAMERA_V2_HEATMAP=0
export CAMERA_V2_DISPLAY_BACKEND=egl
export CAMERA_V2_EGL_FAILOVER_SEC=8.0

echo "SENTINEL_PROFILE inherited_mux=${INHERITED_MUX} inherited_detector=${INHERITED_DETECT} effective_mux=2560x1440 rtsp=tcp latency=250ms detector=RF-DETR-S capture=672x384 model=512x512 threshold=0.18 batch=1 scheduler=adaptive-duty:20-34% detector_path=analysis-tiler demux=disabled mux_batch_retention=bounded raw_truth=1 tracker=OFF flow=OFF reid=OFF display=egl->x11-on-zero-render pascal_safe=1 ui=camera-only-2x3-click-fullscreen"

python scripts/preflight_rfdetr_core.py
python scripts/preflight_pascal_safe.py
python scripts/preflight_sentinel_ui.py
python scripts/preflight_camera_v2_core.py

exec python -m services.camera_v2.monitor_ui
