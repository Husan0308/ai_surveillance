#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
BRANCH_EXPECTED="rebuild/service-architecture-v11-ui-sentinel-cam01-v1-20260901"
LOG="${V11_DS_YOLO_LOG:-/tmp/CAMERA_V11_DS_YOLO_UI_CAM01.log}"
APP_PY="${V11_DS_YOLO_PYTHON:-$HOME/ai_surveillance/.venv/bin/python}"
[[ -x "$APP_PY" ]] || APP_PY="$(command -v python3)"
fail(){ printf 'V11_UI_CAM01_PIPELINE_PREFLIGHT RESULT=FAIL reason=%s\n' "$*" >&2; exit 1; }
[[ -n "${DISPLAY:-}" ]] || fail "DISPLAY_empty"
for plugin in nvurisrcbin nvv4l2decoder queue tee nvstreammux nvvideoconvert capsfilter appsink nvdsosd nveglglessink rtspsrc; do gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing_plugin=$plugin"; done
TRT_HOME="${V11_TRT86_HOME:-$HOME/ai_surveillance}"
TRT_PY="${V11_STEP2_TRT86_PYTHON:-$TRT_HOME/.venv-trt86/bin/python}"
ENGINE="${V11_STEP2_ENGINE:-$TRT_HOME/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
WORKER="${V11_STEP2_TRT86_WORKER:-$ROOT/scripts/yolo26_trt86_step2_worker.py}"
ENV_FILE="${V11_DS_YOLO_ENV_FILE:-$TRT_HOME/.env}"
CURRENT_BRANCH="$(git branch --show-current 2>/dev/null || true)"
[[ "$CURRENT_BRANCH" == "$BRANCH_EXPECTED" ]] || fail "wrong_branch=${CURRENT_BRANCH:-detached} expected=$BRANCH_EXPECTED"
[[ -x "$APP_PY" ]] || fail "app_python_missing=$APP_PY"; [[ -r "$ENV_FILE" ]] || fail "env_file_missing=$ENV_FILE"
[[ -x "$TRT_PY" ]] || fail "trt86_python_missing=$TRT_PY"; [[ -s "$ENGINE" ]] || fail "trt86_engine_missing=$ENGINE"; [[ -s "$WORKER" ]] || fail "trt86_worker_missing=$WORKER"
TRT_VERSION="$($TRT_PY -I -c 'import ctypes, ctypes.util, tensorrt as trt; path=ctypes.util.find_library("cudart"); assert path; ctypes.CDLL(path); print(trt.__version__)' 2>/dev/null || true)"
[[ "$TRT_VERSION" == 8.6.1* ]] || fail "trt86_or_cudart_preflight_failed=${TRT_VERSION:-missing}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" V11_ENV_FILE="$ENV_FILE"
export V11_DS_YOLO_CAMERAS="CAM-01,CAM-02,CAM-03,CAM-04,CAM-05,CAM-06" V11_DS_YOLO_ENABLED="${V11_DS_YOLO_ENABLED:-1}"
export V11_STEP2_TRT86_PYTHON="$TRT_PY" V11_STEP2_TRT86_WORKER="$WORKER" V11_STEP2_ENGINE="$ENGINE"
export V11_DS_YOLO_HZ="${V11_DS_YOLO_HZ:-2.0}" V11_DS_YOLO_CONF="${V11_DS_YOLO_CONF:-0.18}" V11_DS_YOLO_MAX_DET="${V11_DS_YOLO_MAX_DET:-20}" V11_DS_YOLO_BOX_STALE_SEC="${V11_DS_YOLO_BOX_STALE_SEC:-0.80}"
export V11_RTSP_LATENCY_MS="${V11_RTSP_LATENCY_MS:-100}" V11_EXTRA_SURFACES="${V11_EXTRA_SURFACES:-4}" V11_RECONNECT_SEC="${V11_RECONNECT_SEC:-5}"
export V11_UI_PREVIEW_CAMERA="CAM-01" V11_UI_PREVIEW_HZ="${V11_UI_PREVIEW_HZ:-15.0}" V11_UI_PREVIEW_PATH="${V11_UI_PREVIEW_PATH:-/dev/shm/v11_ui_preview_cam01_v1.bin}"
"$APP_PY" -c 'import services.camera_v11.deepstream_trt86_multi_ui_cam01_v1' || fail "runtime_import_failed"
printf 'V11_UI_CAM01_PIPELINE_PREFLIGHT RESULT=PASS branch=%s cameras=6 rtsp_sources=6 rtsp_extra=0 detector_workers=1 ui_camera=CAM-01 ui_hz=%s trt=%s\n' "$BRANCH_EXPECTED" "$V11_UI_PREVIEW_HZ" "$TRT_VERSION"
: >"$LOG"
exec > >(trap '' INT TERM; tee "$LOG") 2>&1
exec "$APP_PY" -u -m services.camera_v11.deepstream_trt86_multi_ui_cam01_v1
