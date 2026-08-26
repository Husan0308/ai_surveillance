#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

export CAMERA_V2_RTSP_TRANSPORT="${CAMERA_V2_RTSP_TRANSPORT:-tcp}"
export CAMERA_V2_RTSP_LATENCY_MS="${CAMERA_V2_RTSP_LATENCY_MS:-100}"
export CAMERA_V2_DETECT_WIDTH=672
export CAMERA_V2_DETECT_HEIGHT=384
export CAMERA_V2_MICRO_BATCH=1
export CAMERA_V2_DETECT_CONF="${CAMERA_V2_DETECT_CONF:-0.08}"
export CAMERA_V2_DETECT_IOU=0.70
export CAMERA_V2_MAX_DET=40
export CAMERA_V2_DETECT_ACTIVE_CAMERAS="CAM-01,CAM-02,CAM-03,CAM-04,CAM-05,CAM-06"
export CAMERA_V2_DETECT_TARGET_HZ="${CAMERA_V2_DETECT_TARGET_HZ:-0.30}"
export CAMERA_V2_DETECT_MIN_HZ="${CAMERA_V2_DETECT_MIN_HZ:-0.20}"
export CAMERA_V2_DETECT_MAX_HZ="${CAMERA_V2_DETECT_MAX_HZ:-0.45}"
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS="${CAMERA_V2_MAX_DETECT_RESULT_AGE_MS:-350}"
export CAMERA_V2_TRACKER_WIDTH=512
export CAMERA_V2_TRACKER_HEIGHT=288
export CAMERA_V2_MIN_DISPLAY_TRACK_CONF="${CAMERA_V2_MIN_DISPLAY_TRACK_CONF:-0.10}"
export CAMERA_V2_TRT86_PYTHON="${CAMERA_V2_TRT86_PYTHON:-$ROOT/.venv-trt86/bin/python}"
export CAMERA_V2_TRT86_ENGINE="${CAMERA_V2_TRT86_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
export CAMERA_V2_TRT86_SHM_WORKER="${CAMERA_V2_TRT86_SHM_WORKER:-$ROOT/scripts/yolo26_trt86_shm_worker.py}"
export QWEN_REID_ENABLED=0

[[ -x "$CAMERA_V2_TRT86_PYTHON" ]] || { echo "STABLE_GLOBAL_ID ERROR missing TRT86 python: $CAMERA_V2_TRT86_PYTHON" >&2; exit 1; }
[[ -f "$CAMERA_V2_TRT86_ENGINE" ]] || { echo "STABLE_GLOBAL_ID ERROR missing engine: $CAMERA_V2_TRT86_ENGINE" >&2; exit 1; }
[[ -f "$CAMERA_V2_TRT86_SHM_WORKER" ]] || { echo "STABLE_GLOBAL_ID ERROR missing worker: $CAMERA_V2_TRT86_SHM_WORKER" >&2; exit 1; }

PYTHON="${CAMERA_V2_MAIN_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"

printf '%s\n' \
  "RESTORE_GLOBAL_ID_PROFILE detector=YOLO26s/TRT8.6/B1 local=NvDCF global=async-ReID" \
  "RESTORE_GLOBAL_ID_POLICY bbox_owner=NvDCF smoother=native global_id=label-only qwen=off" \
  "RESTORE_GLOBAL_ID_DISPLAY mux_scale=bilinear osd=GPU/NV12-direct rgba_convert=0" \
  "RESTORE_GLOBAL_ID_LABEL confirmed=G### tentative=G###? conflict=G###!"

exec "$PYTHON" -u -m services.camera_v2.person_tracking_trt86_reid_restore_stable
