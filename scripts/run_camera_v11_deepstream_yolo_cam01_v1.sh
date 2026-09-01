#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
BRANCH_EXPECTED="rebuild/service-architecture-v11-deepstream-yolo-cam01-v1-20260901"
LOG="${V11_DS_YOLO_LOG:-/tmp/CAMERA_V11_DS_YOLO_CAM01.log}"
APP_PY="${V11_DS_YOLO_PYTHON:-$HOME/ai_surveillance/.venv/bin/python}"
[[ -x "$APP_PY" ]] || APP_PY="$(command -v python3)"

fail() {
  printf 'V11_DS_YOLO_CAM01_PREFLIGHT RESULT=FAIL reason=%s\n' "$*" >&2
  exit 1
}

# DeepStream 7.1 ships TensorRT 10.3, but Pascal/SM6.1 is outside TensorRT 10.x
# support. Therefore CAM-01 keeps DeepStream as the sole RTSP/decode/display owner
# and runs YOLO through the already validated isolated TensorRT 8.6 worker.
# The detector branch is a tee from the same decoded stream: no second RTSP session.

[[ -n "${DISPLAY:-}" ]] || fail "DISPLAY_empty"
for plugin in nvurisrcbin nvv4l2decoder queue tee nvstreammux nvvideoconvert capsfilter appsink nvdsosd nveglglessink rtspsrc; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing_plugin=$plugin"
done

# Prefer the canonical original checkout because untracked TRT artifacts/venvs are
# not copied into git worktrees. All locations remain overridable.
TRT_HOME="${V11_TRT86_HOME:-$HOME/ai_surveillance}"
TRT_PY="${V11_STEP2_TRT86_PYTHON:-$TRT_HOME/.venv-trt86/bin/python}"
ENGINE="${V11_STEP2_ENGINE:-$TRT_HOME/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
WORKER="${V11_STEP2_TRT86_WORKER:-$ROOT/scripts/yolo26_trt86_step2_worker.py}"
ENV_FILE="${V11_DS_YOLO_ENV_FILE:-$TRT_HOME/.env}"
DETECTOR_ENABLED="${V11_DS_YOLO_ENABLED:-1}"

CURRENT_BRANCH="$(git branch --show-current 2>/dev/null || true)"
[[ "$CURRENT_BRANCH" == "$BRANCH_EXPECTED" ]] || fail "wrong_branch=${CURRENT_BRANCH:-detached} expected=$BRANCH_EXPECTED"
[[ -x "$APP_PY" ]] || fail "app_python_missing=$APP_PY"
[[ -r "$ENV_FILE" ]] || fail "env_file_missing=$ENV_FILE"

TRT_VERSION="disabled"
if [[ "$DETECTOR_ENABLED" != "0" ]]; then
  [[ -x "$TRT_PY" ]] || fail "trt86_python_missing=$TRT_PY"
  [[ -s "$ENGINE" ]] || fail "trt86_engine_missing=$ENGINE"
  [[ -s "$WORKER" ]] || fail "trt86_worker_missing=$WORKER"
  TRT_VERSION="$($TRT_PY -I -c 'import ctypes, ctypes.util, tensorrt as trt; path=ctypes.util.find_library("cudart"); assert path; ctypes.CDLL(path); print(trt.__version__)' 2>/dev/null || true)"
  [[ "$TRT_VERSION" == 8.6.1* ]] || fail "trt86_or_cudart_preflight_failed=${TRT_VERSION:-missing}"
fi

# Catch old camera/detector processes before opening CAM-01. The current process
# must be the only owner of the camera RTSP session.
CONFLICT_PATTERN='services\.camera_v11\.(step1_|step2_|step3_|deepstream_yolo_cam01_v1|deepstream_trt86_cam01_v2)|yolo26_trt86_step2_worker\.py'
conflicts="$(pgrep -af "$CONFLICT_PATTERN" || true)"
[[ -z "$conflicts" ]] || fail $'conflicting_camera_process:\n'"$conflicts"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export V11_ENV_FILE="$ENV_FILE"
export V11_DS_YOLO_ENABLED="$DETECTOR_ENABLED"
export V11_DS_YOLO_CAMERA="${V11_DS_YOLO_CAMERA:-CAM-01}"
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

"$APP_PY" -c 'import services.camera_v11.deepstream_trt86_cam01_v2' \
  || fail "runtime_import_failed"
"$APP_PY" -c 'from services.ml_service.app.config import load_settings; rows=[c for c in load_settings().cameras if c.camera_id=="CAM-01"]; assert len(rows)==1 and rows[0].username and rows[0].password' \
  || fail "cam01_credentials_unresolved"
"$APP_PY" -c 'from services.camera_v2.native_bridge import NativeMetaBridge; NativeMetaBridge()' \
  || fail "native_metadata_bridge_unavailable"

printf 'V11_DS_YOLO_CAM01_PREFLIGHT RESULT=PASS branch=%s camera=%s architecture=deepstream-single-rtsp+trt86-sidecar trt=%s engine=%s hz=%s conf=%s\n' \
  "$BRANCH_EXPECTED" "$V11_DS_YOLO_CAMERA" "$TRT_VERSION" "$ENGINE" "$V11_DS_YOLO_HZ" "$V11_DS_YOLO_CONF"
printf 'V11_DS_YOLO_CAM01_GPU_POLICY rtsp_sources=1 decode=deepstream-nvdec detector=trt86-sidecar detector_rtsp=0 detector_queue=latest1 gst_nvinfer=0 trt10=0 trt86=%s second_rtsp=0 opencv=0 ffmpeg=0 model_pt_required=0 parser_build=0 log=%s\n' "$DETECTOR_ENABLED" "$LOG"

# Become the runtime so terminal signals reach its GLib shutdown handler. The
# logger ignores terminal signals and exits on EOF after the runtime flushes.
: >"$LOG"
exec > >(trap '' INT TERM; tee "$LOG") 2>&1
exec "$APP_PY" -u -m services.camera_v11.deepstream_trt86_cam01_v2
