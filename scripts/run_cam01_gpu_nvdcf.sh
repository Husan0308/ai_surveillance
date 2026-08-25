#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# CAM-01 GPU pose + NvDCF baseline. All six RTSP feeds remain visible, while
# detector work is scheduled only for CAM-01. The detector only refreshes NvDCF;
# NvDCF owns the visible bbox on every video frame and bridges brief misses.
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

# IMPORTANT: restore the proven yesterday capture geometry. The direct
# per-source BGRx appsink path was stable at 672x384. 1280x720 caused the gated
# inference branch to stop delivering mailbox frames (calls stayed at zero while
# capture_timeouts increased), even though the six-camera display kept running.
export CAMERA_V2_DETECT_WIDTH=672
export CAMERA_V2_DETECT_HEIGHT=384
export CAMERA_V2_DETECT_CONF=0.10
export CAMERA_V2_DETECT_IOU=0.80
export CAMERA_V2_MAX_DET=50
export CAMERA_V2_MICRO_BATCH=1
export CAMERA_V2_DETECT_ACTIVE_CAMERAS=CAM-01
export CAMERA_V2_DETECT_STARTUP_DELAY=0.5

# Sparse detector + per-frame NvDCF. GPU pose on Pascal can take ~0.4-0.5s in
# the current driver state, so do not throw away a valid refresh just because it
# exceeds the newer 280ms freshness gate. NvDCF keeps moving the live box while
# the detector is in flight; the GPU runtime also projects the refresh forward.
export CAMERA_V2_DETECT_TARGET_HZ=2.4
export CAMERA_V2_DETECT_MIN_HZ=1.8
export CAMERA_V2_DETECT_MAX_HZ=3.0
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS=800
export CAMERA_V2_TRACKER_WIDTH=512
export CAMERA_V2_TRACKER_HEIGHT=288
export CAMERA_V2_MIN_DISPLAY_TRACK_CONF=0.12
export CAMERA_V2_DEDUP_IOU=0.82
export CAMERA_V2_DEDUP_CONTAINMENT=0.94

# No Qwen during the CAM-01 tracking baseline.
export QWEN_REID_ENABLED=0
unset QWEN_REID_URL QWEN_REID_MODEL QWEN_REID_TIMEOUT_SEC || true

# Do not inherit CPU-ONNX / TensorRT experiments.
unset CAMERA_V2_SINGLE_SOURCE_ANALYSIS || true
unset CAMERA_V2_ANALYSIS_TILE_WIDTH CAMERA_V2_ANALYSIS_TILE_HEIGHT || true
unset CAMERA_V2_TRT86_ENGINE CAMERA_V2_TRT86_PYTHON CAMERA_V2_TRT86_WORKER || true
unset NVDS_ENABLE_LATENCY_MEASUREMENT NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT || true

printf '%s\n' \
  "CAM01_GPU_PROFILE detector=YOLO26s-pose.pt device=cuda:0 imgsz=832 source=672x384" \
  "CAM01_GPU_PROFILE active=CAM-01 tracker=NvDCF@512x288 target=2.4Hz max_result_age=800ms shadow_hold=enabled qwen=off"

exec python -u -m services.camera_v2.person_tracking_reid_gpu
