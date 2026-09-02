#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
BRANCH_EXPECTED="rebuild/service-architecture-v11-bbox-nvdcf-cam01-cam02-cam03-v1-20260902"
LOG="${V11_BBOX_NVDCF_LOG:-/tmp/CAMERA_V11_BBOX_NVDCF_CAM01_CAM02_CAM03_CAM04_CAM05_CAM06_SHADOW.log}"
APP_PY="${V11_DS_YOLO_PYTHON:-$HOME/ai_surveillance/.venv/bin/python}"
[[ -x "$APP_PY" ]] || APP_PY="$(command -v python3)"
fail(){ printf 'V11_BBOX_NVDCF_CAM01_CAM02_CAM03_CAM04_CAM05_CAM06_SHADOW_PREFLIGHT RESULT=FAIL reason=%s\n' "$*" >&2; exit 1; }
[[ -n "${DISPLAY:-}" ]] || fail "DISPLAY_empty"
CURRENT_BRANCH="$(git branch --show-current 2>/dev/null || true)"
[[ "$CURRENT_BRANCH" == "$BRANCH_EXPECTED" ]] || fail "wrong_branch=${CURRENT_BRANCH:-detached} expected=$BRANCH_EXPECTED"
for plugin in nvurisrcbin nvv4l2decoder queue tee nvstreammux nvvideoconvert capsfilter appsink nvtracker nvdsosd nveglglessink rtspsrc; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing_plugin=$plugin"
done
TRT_HOME="${V11_TRT86_HOME:-$HOME/ai_surveillance}"
TRT_PY="${V11_STEP2_TRT86_PYTHON:-$TRT_HOME/.venv-trt86/bin/python}"
ENGINE="${V11_STEP2_ENGINE:-$TRT_HOME/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
WORKER="${V11_STEP2_TRT86_WORKER:-$ROOT/scripts/yolo26_trt86_step2_worker.py}"
ENV_FILE="${V11_DS_YOLO_ENV_FILE:-$TRT_HOME/.env}"
TRACKER_LIB="${V11_BBOX_TRACKER_LL_LIB:-/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so}"
TRACKER_CFG="${V11_BBOX_TRACKER_CONFIG:-$ROOT/config/camera_v11_bbox_nvdcf_cam01_v1.yml}"
[[ -x "$APP_PY" ]] || fail "app_python_missing=$APP_PY"
[[ -r "$ENV_FILE" ]] || fail "env_file_missing=$ENV_FILE"
[[ -x "$TRT_PY" ]] || fail "trt86_python_missing=$TRT_PY"
[[ -s "$ENGINE" ]] || fail "trt86_engine_missing=$ENGINE"
[[ -s "$WORKER" ]] || fail "trt86_worker_missing=$WORKER"
[[ -s "$TRACKER_LIB" ]] || fail "nvdcf_library_missing=$TRACKER_LIB"
[[ -s "$TRACKER_CFG" ]] || fail "nvdcf_config_missing=$TRACKER_CFG"
TRT_VERSION="$($TRT_PY -I -c 'import ctypes, ctypes.util, tensorrt as trt; path=ctypes.util.find_library("cudart"); assert path; ctypes.CDLL(path); print(trt.__version__)' 2>/dev/null || true)"
[[ "$TRT_VERSION" == 8.6.1* ]] || fail "trt86_or_cudart_preflight_failed=${TRT_VERSION:-missing}"
CONFLICT_PATTERN='services\\.camera_v11\\.(deepstream_trt86_multi|deepstream_trt86_nvdcf)|yolo26_trt86_step2_worker\\.py'
conflicts="$(pgrep -af "$CONFLICT_PATTERN" || true)"
[[ -z "$conflicts" ]] || fail $'conflicting_camera_process:\n'"$conflicts"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" V11_ENV_FILE="$ENV_FILE"
export V11_DS_YOLO_CAMERAS="CAM-01,CAM-02,CAM-03,CAM-04,CAM-05,CAM-06"
export V11_UI_PREVIEW_CAMERAS="CAM-01,CAM-02,CAM-03,CAM-04,CAM-05,CAM-06"
export V11_UI_PREVIEW_HZ="${V11_UI_PREVIEW_HZ:-15.0}"
export V11_UI_PREVIEW_PATH_CAM01="${V11_UI_PREVIEW_PATH_CAM01:-/dev/shm/v11_ui_preview_cam01_v1.bin}"
export V11_UI_PREVIEW_PATH_CAM02="${V11_UI_PREVIEW_PATH_CAM02:-/dev/shm/v11_ui_preview_cam02_v1.bin}"
export V11_UI_PREVIEW_PATH_CAM03="${V11_UI_PREVIEW_PATH_CAM03:-/dev/shm/v11_ui_preview_cam03_v1.bin}"
export V11_UI_PREVIEW_PATH_CAM04="${V11_UI_PREVIEW_PATH_CAM04:-/dev/shm/v11_ui_preview_cam04_v1.bin}"
export V11_UI_PREVIEW_PATH_CAM05="${V11_UI_PREVIEW_PATH_CAM05:-/dev/shm/v11_ui_preview_cam05_v1.bin}"
export V11_UI_PREVIEW_PATH_CAM06="${V11_UI_PREVIEW_PATH_CAM06:-/dev/shm/v11_ui_preview_cam06_v1.bin}"
export V11_BBOX_TRACK_CAMERAS="CAM-01,CAM-02,CAM-03,CAM-04,CAM-05,CAM-06"

# Keep the accepted detector/NvDCF profile frozen. CAM-01..04 retain the
# six-frame continuity layer. CAM-05 keeps the proven live-shadow display path,
# and CAM-06 is introduced with the same live-shadow display behavior so the
# sixth camera can be validated without changing association/tracker tuning.
export V11_BBOX_TRACKER_WIDTH="${V11_BBOX_TRACKER_WIDTH:-640}"
export V11_BBOX_TRACKER_HEIGHT="${V11_BBOX_TRACKER_HEIGHT:-384}"
export V11_BBOX_TRACKER_LL_LIB="$TRACKER_LIB"
export V11_BBOX_TRACKER_CONFIG="$TRACKER_CFG"
export CAMERA_V2_TRACK_BOX_SIDE_MARGIN="0.0"
export CAMERA_V2_TRACK_BOX_TOP_MARGIN="0.0"
export CAMERA_V2_TRACK_BOX_BOTTOM_MARGIN="0.0"
export CAMERA_V2_MIN_DISPLAY_TRACK_CONF="${CAMERA_V2_MIN_DISPLAY_TRACK_CONF:-0.22}"

# Existing continuity remains untouched for CAM-01..04.
export V11_BBOX_DISPLAY_HOLD_FRAMES="${V11_BBOX_DISPLAY_HOLD_FRAMES:-6}"
export V11_BBOX_DISPLAY_HOLD_SUPPRESS_IOU="${V11_BBOX_DISPLAY_HOLD_SUPPRESS_IOU:-0.50}"

# CAM-05/CAM-06: use current NvDCF INACTIVE/shadow tBbox for at most 10 frames
# (~0.5 s at 20 FPS) after an already-active ID becomes INACTIVE. Shadow
# confidence can legitimately dip below the active-display threshold while NvDCF
# still keeps the target alive, so use a lower display-only floor here. The
# recently-active + 10-frame bound + overlap guard remain the ghost protections.
export V11_BBOX_SHADOW_DIAG_CAMERAS="${V11_BBOX_SHADOW_DIAG_CAMERAS:-CAM-05,CAM-06}"
export V11_BBOX_SHADOW_DISPLAY_CAMERAS="${V11_BBOX_SHADOW_DISPLAY_CAMERAS:-CAM-05,CAM-06}"
export V11_BBOX_SHADOW_DISPLAY_FRAMES="${V11_BBOX_SHADOW_DISPLAY_FRAMES:-10}"
export V11_BBOX_SHADOW_DISPLAY_MIN_CONF="${V11_BBOX_SHADOW_DISPLAY_MIN_CONF:-0.15}"
export V11_BBOX_SHADOW_DISPLAY_SUPPRESS_IOU="${V11_BBOX_SHADOW_DISPLAY_SUPPRESS_IOU:-0.50}"

export V11_DS_YOLO_ENABLED="${V11_DS_YOLO_ENABLED:-1}"
export V11_STEP2_TRT86_PYTHON="$TRT_PY" V11_STEP2_TRT86_WORKER="$WORKER" V11_STEP2_ENGINE="$ENGINE"
export V11_DS_YOLO_HZ="${V11_DS_YOLO_HZ:-2.0}" V11_DS_YOLO_CONF="${V11_DS_YOLO_CONF:-0.18}"
export V11_DS_YOLO_MAX_DET="${V11_DS_YOLO_MAX_DET:-20}" V11_DS_YOLO_BOX_STALE_SEC="${V11_DS_YOLO_BOX_STALE_SEC:-0.80}"
export V11_RTSP_LATENCY_MS="${V11_RTSP_LATENCY_MS:-100}" V11_EXTRA_SURFACES="${V11_EXTRA_SURFACES:-4}" V11_RECONNECT_SEC="${V11_RECONNECT_SEC:-5}"

"$APP_PY" -c 'import services.camera_v11.deepstream_trt86_nvdcf_bbox_shadow_display_v1' || fail "runtime_import_failed"
"$APP_PY" -c 'from services.ml_service.app.config import load_settings; rows=load_settings().cameras; wanted={"CAM-01","CAM-02","CAM-03","CAM-04","CAM-05","CAM-06"}; found={c.camera_id for c in rows if c.username and c.password}; assert wanted <= found' || fail "camera_credentials_unresolved"
printf 'V11_BBOX_NVDCF_CAM01_CAM02_CAM03_CAM04_CAM05_CAM06_SHADOW_PREFLIGHT RESULT=PASS branch=%s cameras=CAM-01,CAM-02,CAM-03,CAM-04,CAM-05,CAM-06 rtsp_sources=6 detector_workers=1 detector_hz=%s raw_conf=%s tracker=nvdcf tracker_size=%sx%s reid=0 display_track_conf=%s cam01_04_hold_frames=%s cam05_06_shadow_frames=%s cam05_06_shadow_min_conf=%s cam05_06_shadow_iou=%s config=%s trt=%s log=%s\n' "$BRANCH_EXPECTED" "$V11_DS_YOLO_HZ" "$V11_DS_YOLO_CONF" "$V11_BBOX_TRACKER_WIDTH" "$V11_BBOX_TRACKER_HEIGHT" "$CAMERA_V2_MIN_DISPLAY_TRACK_CONF" "$V11_BBOX_DISPLAY_HOLD_FRAMES" "$V11_BBOX_SHADOW_DISPLAY_FRAMES" "$V11_BBOX_SHADOW_DISPLAY_MIN_CONF" "$V11_BBOX_SHADOW_DISPLAY_SUPPRESS_IOU" "$TRACKER_CFG" "$TRT_VERSION" "$LOG"
: >"$LOG"
exec > >(trap '' INT TERM; tee "$LOG") 2>&1
exec "$APP_PY" -u -m services.camera_v11.deepstream_trt86_nvdcf_bbox_shadow_display_v1
