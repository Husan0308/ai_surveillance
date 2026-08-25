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

# Production person detector profile for the GTX 1050 Ti machine:
# keep NVDEC/display on the GPU, but run CAM-01 detection through ONNX Runtime
# on the CPU. Local benchmark on this host is ~56-74 ms wall time, avoiding the
# GPU contention that pushed the CUDA detector above 250 ms in the live wall.
export CAMERA_V2_DETECTOR_BACKEND=onnx-cpu
export CAMERA_V2_YOLO_MODEL=yolo26s.onnx
export CAMERA_V2_DETECT_WIDTH=672
export CAMERA_V2_DETECT_HEIGHT=384
export CAMERA_V2_DETECT_CONF=0.08
export CAMERA_V2_DETECT_IOU=0.70
export CAMERA_V2_MAX_DET=40
export CAMERA_V2_MICRO_BATCH=1
export CAMERA_V2_DETECT_ACTIVE_CAMERAS=CAM-01
export CAMERA_V2_DETECT_STARTUP_DELAY=0.5

# The analysis branch is capture-only. Do not build/download a 2560x2160 2x3
# analysis wall just to infer CAM-01. The runtime enables a single-source tiler
# fast path when exactly one detector camera is selected.
export CAMERA_V2_ANALYSIS_TILE_WIDTH=672
export CAMERA_V2_ANALYSIS_TILE_HEIGHT=384
export CAMERA_V2_ANALYSIS_INTERPOLATION=1
export CAMERA_V2_SINGLE_SOURCE_ANALYSIS=1

# This now controls detector cadence rather than GPU occupancy. At ~65 ms CPU
# inference, 18% duty lands near 2.5-3 Hz while leaving plenty of CPU headroom
# for Qt, GStreamer callbacks and ReID bookkeeping.
export CAMERA_V2_DETECT_GPU_DUTY=0.18
export CAMERA_V2_DETECT_GPU_DUTY_MIN=0.12
export CAMERA_V2_DETECT_GPU_DUTY_MAX=0.24

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

# Avoid stale shell overrides from earlier pose/TensorRT experiments.
unset CAMERA_V2_POSE_MODEL CAMERA_V2_POSE_IMGSZ CAMERA_V2_POSE_CONF CAMERA_V2_POSE_IOU || true
unset CAMERA_V2_TRT86_ENGINE CAMERA_V2_TRT86_PYTHON CAMERA_V2_TRT86_WORKER CAMERA_V2_TRT86_JPEG_QUALITY || true
unset NVDS_ENABLE_LATENCY_MEASUREMENT NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT || true

export QWEN_REID_ENABLED=0

echo "SENTINEL_PROFILE inherited_mux=${INHERITED_MUX} inherited_detector=${INHERITED_DETECT} effective_mux=2560x1440 rtsp=tcp latency=250ms detector=YOLO26s-ONNX-CPU@672x384 threshold=0.08 batch=1 scheduler=adaptive-duty:12-24% detector_path=analysis-tiler(single-source-fastpath) demux=disabled tracker=motion-predictor nvtracker=disabled display=egl->x11-on-zero-render pascal_safe=1 ui=camera-only-2x3-click-fullscreen"

python scripts/preflight_pascal_safe.py
python scripts/preflight_sentinel_ui.py
python scripts/preflight_camera_v2_core.py

exec python -m services.camera_v2.monitor_ui
