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

# RF-DETR-S model, but restore the old Core-v1 detection policy around it.
# RF-DETR >=1.6.2 supports non-square predict(shape=(h,w)); match the model
# resize to the 16:9 CCTV analysis tile so people are not squeezed to square.
export CAMERA_V2_DETECT_BACKEND=rfdetr-s
export CAMERA_V2_DETECT_WIDTH=672
export CAMERA_V2_DETECT_HEIGHT=384
export CAMERA_V2_RFDETR_MODEL_WIDTH=672
export CAMERA_V2_RFDETR_MODEL_HEIGHT=384
export CAMERA_V2_DETECT_CONF=0.12
export CAMERA_V2_MAX_DET=40
export CAMERA_V2_MICRO_BATCH=1

# Old Core-v1 conditional second pass.  CAM-05 verifies its difficult desk ROI;
# CAM-06 augments the sofa/desk ROI while the known TV/static zone is excluded.
export CAMERA_V2_RFDETR_ROI_WIDTH=640
export CAMERA_V2_RFDETR_ROI_HEIGHT=512
export CAMERA_V2_RFDETR_ROI_CONF=0.06
export CAMERA_V2_RFDETR_TRUTH_LOG_BUDGET=120

# RF-DETR-S is heavier than the old YOLO worker.  Keep enough cadence for the
# old 2-hit Byte/Kalman birth policy (~<1 s six-camera correction cycle) while
# retaining the existing wall-p95 adaptive governor.
export CAMERA_V2_DETECT_GPU_DUTY=0.68
export CAMERA_V2_DETECT_GPU_DUTY_MIN=0.55
export CAMERA_V2_DETECT_GPU_DUTY_MAX=0.72

# Restored old-good Core-v1 presentation policy: short hold, bounded prediction,
# 3 s reacquire memory and stale detector result rejection.  No optical flow,
# ReID or NvDCF is allowed in this bring-up.
export CAMERA_V2_OLDGOOD_HOLD_MS=850
export CAMERA_V2_OLDGOOD_MEMORY_MS=3000
export CAMERA_V2_OLDGOOD_PREDICTION_MS=420
export CAMERA_V2_OLDGOOD_MAX_RESULT_AGE_SEC=0.95

export CAMERA_V2_PASCAL_SAFE=1
export CAMERA_V2_HEATMAP=0
export CAMERA_V2_DISPLAY_BACKEND=egl
export CAMERA_V2_EGL_FAILOVER_SEC=8.0

echo "SENTINEL_PROFILE inherited_mux=${INHERITED_MUX} inherited_detector=${INHERITED_DETECT} effective_mux=2560x1440 rtsp=tcp latency=250ms detector=RF-DETR-S capture=672x384 model=672x384 threshold=0.12 batch=1 scheduler=adaptive-duty:55-72% detector_path=analysis-tiler demux=disabled mux_batch_retention=bounded logic=old-good-core-v1 roi=CAM05-verify+CAM06-augment tracker=kalman-byte flow=OFF reid=OFF nvtracker=OFF overlay=post-tiler-wall-space display=egl->x11-on-zero-render pascal_safe=1 ui=camera-only-2x3-click-fullscreen"

python scripts/preflight_rfdetr_core.py
python scripts/preflight_pascal_safe.py
python scripts/preflight_sentinel_ui.py
python scripts/preflight_camera_v2_core.py

exec python -m services.camera_v2.monitor_ui
