#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
FROZEN_STEP2_SHA="2f83fb3ef5c2bb4e4cba7dc9c923c918fe3847a1"
BBOX_LOCK="/tmp/ai_surveillance_camera_v11_bbox_overlay_v1.lock"
STEP3_LOCK="/tmp/ai_surveillance_camera_v11_step3_tracker_v2.lock"
STEP2_LOCK="/tmp/ai_surveillance_camera_v11_step2_production_v25.lock"
STEP1_LOCK="/tmp/ai_surveillance_camera_v11_step1_v7.lock"
DISPLAY_LOG="${V11_BBOX_DISPLAY_LOG:-/tmp/CAMERA_V11_BBOX_DISPLAY.log}"
TRACKER_LOG="${V11_BBOX_TRACKER_LOG:-/tmp/CAMERA_V11_BBOX_TRACKER.log}"
STATE_PATH="${V11_BBOX_STATE_PATH:-/dev/shm/ai_surveillance/v11_bbox_overlay_v1.json}"
ENGINE="$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine"
TRT_PY="$ROOT/.venv-trt86/bin/python"
APP_PY="$ROOT/.venv/bin/python"
PRIME_SCRIPT="$ROOT/scripts/benchmark_yolo26_trt86_step2_worker_v22.py"

fail() {
  printf 'CAMERA_V11_BBOX_PREFLIGHT result=FAIL reason=%s\n' "$*" >&2
  exit 1
}

command -v flock >/dev/null 2>&1 || fail "flock_missing"
exec 9>"$BBOX_LOCK"; flock -n 9 || fail "another_bbox_launcher_holds=$BBOX_LOCK"
exec 8>"$STEP3_LOCK"; flock -n 8 || fail "another_step3_launcher_holds=$STEP3_LOCK"
exec 7>"$STEP2_LOCK"; flock -n 7 || fail "step2_or_other_step3_holds=$STEP2_LOCK"
exec 6>"$STEP1_LOCK"; flock -n 6 || fail "another_step1_display_holds=$STEP1_LOCK"

[[ -n "${DISPLAY:-}" ]] || fail "DISPLAY_empty"
[[ -s "$ENGINE" ]] || fail "fp32_engine_missing"
[[ -x "$TRT_PY" ]] || fail "trt86_python_missing"
[[ -x "$APP_PY" ]] || fail "app_python_missing"
[[ -f "$PRIME_SCRIPT" ]] || fail "trt_prime_script_missing"

for plugin in nvurisrcbin nvv4l2decoder queue nvstreammux nvvideoconvert capsfilter nvdsosd nveglglessink rtspsrc; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing_plugin=$plugin"
done

# V7 uses decoder low-latency-mode on CAM-02 only.
if ! gst-inspect-1.0 nvv4l2decoder 2>/dev/null | grep 'low-latency-mode' >/dev/null; then
  fail "nvv4l2decoder_low_latency_property_missing"
fi

git cat-file -e "${FROZEN_STEP2_SHA}^{commit}" 2>/dev/null || fail "frozen_step2_sha_missing_locally"
git merge-base --is-ancestor "$FROZEN_STEP2_SHA" HEAD || fail "branch_not_based_on_frozen_step2"

git diff --quiet "$FROZEN_STEP2_SHA" -- \
  services/camera_v11/step1_cam02_lowlat_v7.py \
  services/camera_v11/step1_independent_egl_v4.py \
  services/camera_v11/step2_production_fp32.py \
  services/camera_v11/step2_production_fp32_v12.py \
  services/camera_v11/step2_production_fp32_v13.py \
  services/camera_v11/step2_production_fp32_v18.py \
  services/camera_v11/step2_trt86.py \
  services/camera_v11/step3_tracker_v2.py \
  services/camera_v11/step3_tracking_v2.py \
  scripts/yolo26_trt86_step2_worker.py \
  scripts/run_camera_v11_step1_v7.sh \
  scripts/camera_v11_powermizer_keeper_v25.sh \
  || fail "frozen_camera_or_detector_files_changed"

# Build/import checks before any camera is opened.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$APP_PY" -m py_compile \
  services/camera_v11/bbox_overlay_ipc_v1.py \
  services/camera_v11/step3_bbox_publish_v1.py \
  scripts/test_camera_v11_bbox_overlay_v1.py \
  || fail "python_compile_failed"
"$APP_PY" scripts/test_camera_v11_bbox_overlay_v1.py >/tmp/CAMERA_V11_BBOX_UNIT.log 2>&1 \
  || { cat /tmp/CAMERA_V11_BBOX_UNIT.log >&2; fail "bbox_unit_tests_failed"; }

"$TRT_PY" - <<'PY' || fail "display_import_or_native_bridge_failed"
from services.camera_v2.native_bridge import NativeMetaBridge
from services.camera_v11.step1_bbox_overlay_v1 import V11Step1BboxOverlayV1
bridge = NativeMetaBridge()
print(f"CAMERA_V11_BBOX_IMPORT_OK bridge={bridge.path} runtime={V11Step1BboxOverlayV1.__name__}")
PY

CONFLICT_PATTERN='services\.camera_v11\.(step1_cam02_lowlat_v7|step1_bbox_overlay_v1|step2_production_fp32(_v[0-9]+)?|step3_tracking_v[0-9]+|step3_bbox_publish_v1)|yolo26_trt86_step2_worker\.py'
conflicts="$(pgrep -af "$CONFLICT_PATTERN" || true)"
[[ -z "$conflicts" ]] || fail $'conflicting_camera_or_trt_process:\n'"$conflicts"

# shellcheck source=/dev/null
source "$ROOT/scripts/camera_v11_powermizer_keeper_v25.sh"

display_pid=""
tracker_pid=""
cleaned=0
cleanup() {
  (( cleaned == 1 )) && return 0
  cleaned=1
  for pid in "$tracker_pid" "$display_pid"; do
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "$tracker_pid" "$display_pid"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
  v11_powermizer_stop || true
  rm -f "$STATE_PATH" 2>/dev/null || true
}
on_signal() { cleanup; exit 130; }
trap cleanup EXIT
trap on_signal INT TERM

mkdir -p "$(dirname "$STATE_PATH")"
rm -f "$STATE_PATH"
: >"$DISPLAY_LOG"
: >"$TRACKER_LOG"

# Preserve the exact V25 PowerMizer priming sequence used by accepted Step3.
printf 'CAMERA_V11_BBOX_POWER_PRIME phase=baseline start=1\n'
"$TRT_PY" "$PRIME_SCRIPT" --engine "$ENGINE" --warmup 30 --iterations 100 \
  >>"${V11_POWERMIZER_KEEPER_LOG:-/tmp/CAMERA_V11_POWERMIZER_KEEPER.log}" 2>&1 \
  || fail "power_prime_baseline_failed"
printf 'CAMERA_V11_BBOX_POWER_PRIME phase=baseline result=PASS\n'

v11_powermizer_start || fail "powermizer_keeper_start"
sleep 1
printf 'CAMERA_V11_BBOX_POWER_PRIME phase=gui_held start=1\n'
"$TRT_PY" "$PRIME_SCRIPT" --engine "$ENGINE" --warmup 30 --iterations 100 \
  >>"$V11_POWERMIZER_KEEPER_LOG" 2>&1 \
  || fail "power_prime_gui_held_failed"

clock="$(v11_powermizer_mem_clock_mhz || true)"
minimum_mhz="${V11_POWERMIZER_MIN_MEMORY_MHZ:-3000}"
if [[ ! "$clock" =~ ^[0-9]+$ ]] || (( clock < minimum_mhz )); then
  fail "vram_startup_prime_gate memory_mhz=${clock:-unknown} minimum_mhz=$minimum_mhz"
fi
printf 'CAMERA_V11_BBOX_POWER_PRIME phase=gui_held result=PASS memory_mhz=%s\n' "$clock"

# Preserve V7 camera policy exactly. Only the display tail gains batch1 metadata + GPU OSD.
export V11_RTSP_TRANSPORT=tcp
export V11_RTSP_LATENCY_MS="${V11_RTSP_LATENCY_MS:-100}"
export V11_DROP_ON_LATENCY="${V11_DROP_ON_LATENCY:-1}"
export V11_EXTRA_SURFACES="${V11_EXTRA_SURFACES:-4}"
export V11_UDP_BUFFER_SIZE="${V11_UDP_BUFFER_SIZE:-8388608}"
export V11_RECONNECT_SEC="${V11_RECONNECT_SEC:-5}"
export V11_STARTUP_STAGGER_SEC="${V11_STARTUP_STAGGER_SEC:-0.40}"
export V11_STATS_INTERVAL_SEC="${V11_STATS_INTERVAL_SEC:-5}"
export V11_TILE_WIDTH="${V11_TILE_WIDTH:-640}"
export V11_TILE_HEIGHT="${V11_TILE_HEIGHT:-360}"
export V11_SCALE_INTERPOLATION="${V11_SCALE_INTERPOLATION:-4}"
export V11_LOWLAT_CAMERAS="${V11_LOWLAT_CAMERAS:-CAM-02}"
export V11_BBOX_STATE_PATH="$STATE_PATH"
export V11_BBOX_STALE_SEC="${V11_BBOX_STALE_SEC:-1.10}"

printf 'CAMERA_V11_BBOX_PREFLIGHT result=PASS frozen_step2_sha=%s display_log=%s tracker_log=%s state=%s\n' \
  "$FROZEN_STEP2_SHA" "$DISPLAY_LOG" "$TRACKER_LOG" "$STATE_PATH"

"$TRT_PY" -u -m services.camera_v11.step1_bbox_overlay_v1 >"$DISPLAY_LOG" 2>&1 &
display_pid=$!

display_ready=0
for _ in $(seq 1 150); do
  if grep -q '^CAMERA_V11_BBOX_DISPLAY_ARCH ' "$DISPLAY_LOG"; then
    display_ready=1
    break
  fi
  kill -0 "$display_pid" 2>/dev/null || break
  sleep 0.1
done
(( display_ready == 1 )) || { tail -n 100 "$DISPLAY_LOG" >&2 || true; fail "bbox_display_not_ready"; }

export V11_STEP2_MODE=full
export V11_STEP2_HZ="${V11_STEP3_HZ:-2.0}"
export V11_STEP2_CONF="${V11_STEP3_CONF:-0.18}"
export V11_STEP2_ENGINE="$ENGINE"
export V11_STEP2_TRT86_PYTHON="$TRT_PY"
export V11_STEP2_TRT86_WORKER="$ROOT/scripts/yolo26_trt86_step2_worker.py"

"$APP_PY" -u -m services.camera_v11.step3_bbox_publish_v1 >"$TRACKER_LOG" 2>&1 &
tracker_pid=$!

live_ready=0
for _ in $(seq 1 "${V11_BBOX_LIVE_READY_ATTEMPTS:-600}"); do
  if grep -q '^CAMERA_V11_STEP3_V2_TRACKER ' "$TRACKER_LOG" && \
     grep -q '^CAMERA_V11_BBOX_PUBLISHER ' "$TRACKER_LOG"; then
    live_ready=1
    break
  fi
  kill -0 "$tracker_pid" 2>/dev/null || break
  sleep 0.1
done
(( live_ready == 1 )) || {
  tail -n 120 "$TRACKER_LOG" >&2 || true
  tail -n 80 "$DISPLAY_LOG" >&2 || true
  fail "live_bbox_metadata_not_ready"
}

printf 'CAMERA_V11_BBOX_RUNNING display_pid=%s tracker_pid=%s state=%s labels=0 reid=0 global_id=0\n' \
  "$display_pid" "$tracker_pid" "$STATE_PATH"
printf 'CAMERA_V11_BBOX_HINT log_command="tail -f %s %s"\n' "$DISPLAY_LOG" "$TRACKER_LOG"

while kill -0 "$display_pid" 2>/dev/null && kill -0 "$tracker_pid" 2>/dev/null; do
  sleep 1
done

kill -0 "$display_pid" 2>/dev/null || wait "$display_pid"
wait "$tracker_pid"
