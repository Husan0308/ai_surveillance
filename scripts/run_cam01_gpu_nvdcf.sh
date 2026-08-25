#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Exact CAM-01 tuning profile: all six RTSP feeds stay visible, but AI is only
# scheduled for CAM-01.  YOLO26s-pose refreshes NvDCF; NvDCF owns the visible
# bbox every video frame and survives brief detector misses via shadow tracking.
export CAMERA_V2_RTSP_TRANSPORT=tcp
export CAMERA_V2_RTSP_LATENCY_MS=250
export CAMERA_V2_MUX_TIMEOUT_US=50000

export CAMERA_V2_FRAME_WIDTH=2560
export CAMERA_V2_FRAME_HEIGHT=1440
export CAMERA_V2_WALL_WIDTH=1920
export CAMERA_V2_WALL_HEIGHT=720

export CAMERA_V2_DETECTOR_BACKEND=pose-gpu
export CAMERA_V2_POSE_MODEL="$PWD/yolo26s-pose.pt"
export CAMERA_V2_POSE_IMGSZ=832
export CAMERA_V2_POSE_CONF=0.10
export CAMERA_V2_POSE_IOU=0.80

# Preserve more CAM-01 detail before Ultralytics performs its 832px pose resize.
# Only CAM-01 reaches this gated branch, so the other five cameras do not pay the
# detector copy/resize cost.
export CAMERA_V2_DETECT_WIDTH=1280
export CAMERA_V2_DETECT_HEIGHT=720
export CAMERA_V2_DETECT_CONF=0.10
export CAMERA_V2_DETECT_IOU=0.80
export CAMERA_V2_MAX_DET=50
export CAMERA_V2_MICRO_BATCH=1
export CAMERA_V2_DETECT_ACTIVE_CAMERAS=CAM-01
export CAMERA_V2_DETECT_STARTUP_DELAY=0.5

# Sparse detector + per-frame NvDCF.  The tracker is deliberately small enough
# for Pascal while keeping roughly two seconds of shadow tracking at ~20 fps.
export CAMERA_V2_DETECT_TARGET_HZ=3.0
export CAMERA_V2_DETECT_MIN_HZ=2.2
export CAMERA_V2_DETECT_MAX_HZ=3.6
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS=280
export CAMERA_V2_TRACKER_WIDTH=480
export CAMERA_V2_TRACKER_HEIGHT=288
export CAMERA_V2_MIN_DISPLAY_TRACK_CONF=0.12
export CAMERA_V2_DEDUP_IOU=0.82
export CAMERA_V2_DEDUP_CONTAINMENT=0.94

# No Qwen during the CAM-01 tracking baseline.
export QWEN_REID_ENABLED=0
unset QWEN_REID_URL QWEN_REID_MODEL QWEN_REID_TIMEOUT_SEC || true

# Do not inherit today's CPU-ONNX / TensorRT experiments.
unset CAMERA_V2_SINGLE_SOURCE_ANALYSIS || true
unset CAMERA_V2_ANALYSIS_TILE_WIDTH CAMERA_V2_ANALYSIS_TILE_HEIGHT || true
unset CAMERA_V2_TRT86_ENGINE CAMERA_V2_TRT86_PYTHON CAMERA_V2_TRT86_WORKER || true
unset NVDS_ENABLE_LATENCY_MEASUREMENT NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT || true

printf '%s\n' \
  "CAM01_GPU_PROFILE detector=YOLO26s-pose.pt device=cuda:0 imgsz=832 source=1280x720" \
  "CAM01_GPU_PROFILE active=CAM-01 tracker=NvDCF@480x288 target=3.0Hz shadow_hold=enabled qwen=off"

exec python -u -m services.camera_v2.person_tracking_reid_gpu
