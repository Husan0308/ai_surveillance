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

# Keep RTSP deterministic on the target NVR. The display path stays independent
# from detector scheduling, so a slow AI pass must never build a video backlog.
export CAMERA_V2_RTSP_TRANSPORT=tcp
export CAMERA_V2_RTSP_LATENCY_MS=250
export CAMERA_V2_MUX_TIMEOUT_US=50000

export CAMERA_V2_FRAME_WIDTH=2560
export CAMERA_V2_FRAME_HEIGHT=1440
export CAMERA_V2_WALL_WIDTH=2880
export CAMERA_V2_WALL_HEIGHT=1080

# CAM-01 production detector: restore the pose/keypoint-based person validation
# that gave the best occlusion/seated-person results, but execute the exported
# fixed-shape model through ONNX Runtime on CPU. GPU remains reserved for
# NVDEC/display/ReID instead of PyTorch detector kernels.
export CAMERA_V2_DETECTOR_BACKEND=onnx-cpu
export CAMERA_V2_DETECT_TASK=pose
export CAMERA_V2_YOLO_MODEL=yolo26s-pose.onnx
export CAMERA_V2_DETECT_WIDTH=832
export CAMERA_V2_DETECT_HEIGHT=480
export CAMERA_V2_DETECT_CONF=0.10
export CAMERA_V2_DETECT_IOU=0.80
export CAMERA_V2_MAX_DET=50
export CAMERA_V2_MICRO_BATCH=1
export CAMERA_V2_DETECT_ACTIVE_CAMERAS=CAM-01
export CAMERA_V2_DETECT_STARTUP_DELAY=0.5

# Match the old good pose profile: preserve a 1280x720 CAM-01 source crop and
# let the ONNX model perform the single resize to its fixed 832x480 tensor.
# Feeding a pre-shrunk 672x384 tile loses small/occluded-person detail.
export CAMERA_V2_ANALYSIS_TILE_WIDTH=1280
export CAMERA_V2_ANALYSIS_TILE_HEIGHT=720
export CAMERA_V2_ANALYSIS_INTERPOLATION=1
export CAMERA_V2_SINGLE_SOURCE_ANALYSIS=1
export CAMERA_V2_POSE_INPUT_WIDTH=832
export CAMERA_V2_POSE_INPUT_HEIGHT=480
export CAMERA_V2_POSE_CONF=0.10
export CAMERA_V2_POSE_IOU=0.80

# ~80-105 ms CPU pose inference at 22-32% duty gives roughly 2.2-3.5 fresh
# detector updates/sec while the motion tracker fills the gaps. Keep this
# independent of camera-wall GPU occupancy.
export CAMERA_V2_DETECT_GPU_DUTY=0.28
export CAMERA_V2_DETECT_GPU_DUTY_MIN=0.22
export CAMERA_V2_DETECT_GPU_DUTY_MAX=0.32

export CAMERA_V2_PASCAL_SAFE=1
export CAMERA_V2_HEATMAP=0

# Start GPU-native EGL; use X11 only when buffers reach the sink but EGL proves
# zero-render for the bounded watchdog interval.
export CAMERA_V2_DISPLAY_BACKEND=egl
export CAMERA_V2_EGL_FAILOVER_SEC=8.0

export CAMERA_V2_BOX_SIDE_MARGIN=0.08
export CAMERA_V2_BOX_TOP_MARGIN=0.04
export CAMERA_V2_BOX_BOTTOM_MARGIN=0.10
export CAMERA_V2_BOX_MAX_AGE=2.0
export CAMERA_V2_BOX_MAX_PREDICT=0.40
export CAMERA_V2_BOX_RENDER_AGE=0.35

# Avoid stale shell overrides from TensorRT/latency instrumentation experiments.
unset CAMERA_V2_POSE_MODEL CAMERA_V2_POSE_IMGSZ || true
unset CAMERA_V2_TRT86_ENGINE CAMERA_V2_TRT86_PYTHON CAMERA_V2_TRT86_WORKER CAMERA_V2_TRT86_JPEG_QUALITY || true
unset NVDS_ENABLE_LATENCY_MEASUREMENT NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT || true

export QWEN_REID_ENABLED=0

echo "SENTINEL_PROFILE inherited_mux=${INHERITED_MUX} inherited_detector=${INHERITED_DETECT} effective_mux=2560x1440 rtsp=tcp latency=250ms detector=YOLO26s-POSE-ONNX-CPU@832x480 source=1280x720 threshold=0.10 batch=1 scheduler=adaptive-duty:22-32% detector_path=analysis-tiler(single-source-fastpath) demux=disabled tracker=motion-predictor nvtracker=disabled display=egl->x11-on-zero-render pascal_safe=1 ui=camera-only-2x3-click-fullscreen"

python scripts/preflight_pascal_safe.py
python scripts/preflight_sentinel_ui.py
python scripts/preflight_camera_v2_core.py

exec python -m services.camera_v2.monitor_ui
