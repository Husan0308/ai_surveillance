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
export CAMERA_V2_DETECT_TARGET_HZ="${CAMERA_V2_DETECT_TARGET_HZ:-0.55}"
export CAMERA_V2_DETECT_MIN_HZ="${CAMERA_V2_DETECT_MIN_HZ:-0.35}"
export CAMERA_V2_DETECT_MAX_HZ="${CAMERA_V2_DETECT_MAX_HZ:-0.75}"
export CAMERA_V2_MAX_DETECT_RESULT_AGE_MS="${CAMERA_V2_MAX_DETECT_RESULT_AGE_MS:-350}"
export CAMERA_V2_TRACKER_WIDTH=512
export CAMERA_V2_TRACKER_HEIGHT=288
export CAMERA_V2_MIN_DISPLAY_TRACK_CONF="${CAMERA_V2_MIN_DISPLAY_TRACK_CONF:-0.10}"
export CAMERA_V2_TRT86_PYTHON="${CAMERA_V2_TRT86_PYTHON:-$ROOT/.venv-trt86/bin/python}"
export CAMERA_V2_TRT86_ENGINE="${CAMERA_V2_TRT86_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
export CAMERA_V2_TRT86_SHM_WORKER="${CAMERA_V2_TRT86_SHM_WORKER:-$ROOT/scripts/yolo26_trt86_shm_worker.py}"

[[ -x "$CAMERA_V2_TRT86_PYTHON" ]] || { echo "STABLE_TRT ERROR missing TRT86 python: $CAMERA_V2_TRT86_PYTHON" >&2; exit 1; }
[[ -f "$CAMERA_V2_TRT86_ENGINE" ]] || { echo "STABLE_TRT ERROR missing engine: $CAMERA_V2_TRT86_ENGINE" >&2; exit 1; }
[[ -f "$CAMERA_V2_TRT86_SHM_WORKER" ]] || { echo "STABLE_TRT ERROR missing worker: $CAMERA_V2_TRT86_SHM_WORKER" >&2; exit 1; }
"$CAMERA_V2_TRT86_PYTHON" - <<'PY'
import tensorrt as trt
assert str(trt.__version__).startswith('8.6.1'), trt.__version__
print(f"RESTORE_STABLE_TRT_PREFLIGHT tensorrt={trt.__version__}")
PY

PYTHON="${CAMERA_V2_MAIN_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"

printf '%s\n' \
  "RESTORE_STABLE_TRACKING detector=YOLO26s/TRT8.6/B1/672x384 conf=$CAMERA_V2_DETECT_CONF" \
  "RESTORE_STABLE_TRACKING capture=JIT/no-prefetch active=all6 target=${CAMERA_V2_DETECT_TARGET_HZ}Hz/cam" \
  "RESTORE_STABLE_TRACKING bbox_owner=NvDCF tracker=512x288 smoother=native global_id=off external_nms=off"

exec "$PYTHON" -u -m services.camera_v2.person_tracking_trt86_restore_stable
