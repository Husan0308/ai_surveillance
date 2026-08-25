#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CAMERA_V2_RTSP_TRANSPORT=tcp
export CAMERA_V2_RTSP_LATENCY_MS=50

export CAMERA_V2_DETECT_WIDTH=672
export CAMERA_V2_DETECT_HEIGHT=384
export CAMERA_V2_MICRO_BATCH=1
export CAMERA_V2_DETECT_ACTIVE_CAMERAS=CAM-01
export CAMERA_V2_DETECT_CONF=0.05
export CAMERA_V2_DETECT_IOU=0.70
export CAMERA_V2_MAX_DET=40

export CAMERA_V2_DETECT_TARGET_HZ=2.0
export CAMERA_V2_DETECT_MIN_HZ=1.8
export CAMERA_V2_DETECT_MAX_HZ=2.3

# Correctness floor: the measured TRT86 round-trip is ~154-190 ms, so 160 ms
# guarantees valid results are frequently discarded before NvDCF sees them.
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS=350

export CAMERA_V2_TRACKER_WIDTH=512
export CAMERA_V2_TRACKER_HEIGHT=288
export CAMERA_V2_MIN_DISPLAY_TRACK_CONF=0.12

export CAMERA_V2_TRT86_PYTHON="${CAMERA_V2_TRT86_PYTHON:-$PWD/.venv-trt86/bin/python}"
export CAMERA_V2_TRT86_ENGINE="${CAMERA_V2_TRT86_ENGINE:-$PWD/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
export CAMERA_V2_TRT86_SHM_WORKER="${CAMERA_V2_TRT86_SHM_WORKER:-$PWD/scripts/yolo26_trt86_shm_worker.py}"

export QWEN_REID_ENABLED=0
unset QWEN_REID_URL QWEN_REID_MODEL QWEN_REID_TIMEOUT_SEC || true

printf '%s\n' \
  "CAM01_TRT86_PROFILE engine=$(basename "$CAMERA_V2_TRT86_ENGINE") input=672x384/b1/fp32 active=CAM-01 target=2.0Hz max_result_age>=350ms tracker=512x288 qwen=0" \
  "CAM01_TRT86_PIPELINE backend=trt86-sidecar-shm-bgr base64=0 jpeg=0 queue_depth=1 nvdcf=per-frame"

exec python -u -m services.camera_v2.person_tracking_trt86
