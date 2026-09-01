#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
BRANCH_EXPECTED="rebuild/service-architecture-v11-deepstream-yolo-cam01-cam02-cam03-cam04-cam05-v1-20260901"
LOG="${V11_DS_YOLO_LOG:-/tmp/CAMERA_V11_DS_YOLO_CAM01_CAM02_CAM03_CAM04_CAM05.log}"
APP_PY="${V11_DS_YOLO_PYTHON:-$HOME/ai_surveillance/.venv/bin/python}"
[[ -x "$APP_PY" ]] || APP_PY="$(command -v python3)"

fail() {
  printf 'V11_DS_YOLO_MULTI_PREFLIGHT RESULT=FAIL reason=%s\n' "$*" >&2
  exit 1
}

[[ -n "${DISPLAY:-}" ]] || fail "DISPLAY_empty"
for plugin in nvurisrcbin nvv4l2decoder queue tee nvstreammux nvvideoconvert capsfilter appsink nvdsosd nveglglessink rtspsrc; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing_plugin=$plugin"
done

TRT_HOME="${V11_TRT86_HOME:-$HOME/ai_surveillance}"
TRT_PY="${V11_STEP2_TRT86_PYTHON:-$TRT_HOME/.venv-trt86/bin/python}"
ENGINE="${V11_STEP2_ENGINE:-$TRT_HOME/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
WORKER="${V11_STEP2_TRT86_WORKER:-$ROOT/scripts/yolo26_trt86_step2_worker.py}"
ENV_FILE="${V11_DS_YOLO_ENV_FILE:-$TRT_HOME/.env}"
DETECTOR_ENABLED="${V11_DS_YOLO_ENABLED:-1}"
CAMERAS="${V11_DS_YOLO_CAMERAS:-CAM-01,CAM-02,CAM-03,CAM-04,CAM-05}"

CURRENT_BRANCH="$(git branch --show-current 2>/dev/null || true)"
[[ "$CURRENT_BRANCH" == "$BRANCH_EXPECTED" ]] || fail "wrong_branch=${CURRENT_BRANCH:-detached} expected=$BRANCH_EXPECTED"
[[ -x "$APP_PY" ]] || fail "app_python_missing=$APP_PY"
[[ -r "$ENV_FILE" ]] || fail "env_file_missing=$ENV_FILE"
[[ "$CAMERAS" == "CAM-01,CAM-02,CAM-03,CAM-04,CAM-05" ]] || fail "camera_set=$CAMERAS expected=CAM-01,CAM-02,CAM-03,CAM-04,CAM-05"

TRT_VERSION="disabled"
if [[ "$DETECTOR_ENABLED" != "0" ]]; then
  [[ -x "$TRT_PY" ]] || fail "trt86_python_missing=$TRT_PY"
  [[ -s "$ENGINE" ]] || fail "trt86_engine_missing=$ENGINE"
  [[ -s "$WORKER" ]] || fail "trt86_worker_missing=$WORKER"
  TRT_VERSION="$($TRT_PY -I -c 'import ctypes, ctypes.util, tensorrt as trt; path=ctypes.util.find_library("cudart"); assert path; ctypes.CDLL(path); print(trt.__version__)' 2>/dev/null || true)"
  [[ "$TRT_VERSION" == 8.6.1* ]] || fail "trt86_or_cudart_preflight_failed=${TRT_VERSION:-missing}"
fi

CONFLICT_PATTERN='services\.camera_v11\.(step1_|step2_|step3_|deepstream_yolo_cam01_v1|deepstream_trt86_cam01_v2|deepstream_trt86_multi_v1)|yolo26_trt86_step2_worker\.py'
conflicts="$(pgrep -af "$CONFLICT_PATTERN" || true)"
[[ -z "$conflicts" ]] || fail $'conflicting_camera_process:\n'"$conflicts"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export V11_ENV_FILE="$ENV_FILE"
export V11_DS_YOLO_ENABLED="$DETECTOR_ENABLED"
export V11_DS_YOLO_CAMERAS="$CAMERAS"
export V11_STEP2_TRT86_PYTHON="$TRT_PY"
export V11_STEP2_TRT86_WORKER="$WORKER"
export V11_STEP2_ENGINE="$ENGINE"
export V11_DS_YOLO_HZ="${V11_DS_YOLO_HZ:-2.0}"
export V11_DS_YOLO_CONF="${V11_DS_YOLO_CONF:-0.18}"
export V11_DS_YOLO_MAX_DET="${V11_DS_YOLO_MAX_DET:-20}"
export V11_DS_YOLO_BOX_STALE_SEC="${V11_DS_YOLO_BOX_STALE_SEC:-0.80}"
export V11_RTSP_LATENCY_MS="${V11_RTSP_LATENCY_MS:-100}"
export V11_EXTRA_SURFACES="${V11_EXTRA_SURFACES:-4}"
export V11_RECONNECT_SEC="${V11_RECONNECT_SEC:-5}"

"$APP_PY" -c 'import services.camera_v11.deepstream_trt86_multi_v1' || fail "runtime_import_failed"
"$APP_PY" - <<'PY' || fail "camera_credentials_unresolved"
from services.ml_service.app.config import load_settings
required = {"CAM-01", "CAM-02", "CAM-03", "CAM-04", "CAM-05"}
rows = {c.camera_id: c for c in load_settings().cameras}
assert required <= rows.keys()
for cid in required:
    assert rows[cid].username and rows[cid].password
PY
"$APP_PY" -c 'from services.camera_v2.native_bridge import NativeMetaBridge; NativeMetaBridge()' || fail "native_metadata_bridge_unavailable"

printf 'V11_DS_YOLO_MULTI_PREFLIGHT RESULT=PASS branch=%s cameras=%s architecture=deepstream-per-camera-single-rtsp+shared-trt86 trt=%s engine=%s hz=%s conf=%s\n' \
  "$BRANCH_EXPECTED" "$CAMERAS" "$TRT_VERSION" "$ENGINE" "$V11_DS_YOLO_HZ" "$V11_DS_YOLO_CONF"
printf 'V11_DS_YOLO_MULTI_GPU_POLICY cameras=5 rtsp_sources=5 rtsp_per_camera=1 decode=deepstream-nvdec detector=shared-trt86-sidecar detector_workers=1 detector_rtsp=0 detector_queue=latest1-per-camera scheduler=round-robin gst_nvinfer=0 trt10=0 second_rtsp=0 opencv=0 ffmpeg=0 model_pt_required=0 log=%s\n' "$LOG"

: >"$LOG"
exec > >(trap '' INT TERM; tee "$LOG") 2>&1
exec "$APP_PY" -u -m services.camera_v11.deepstream_trt86_multi_v1
