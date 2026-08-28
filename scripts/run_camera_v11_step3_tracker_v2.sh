#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
FROZEN_STEP2_SHA="2f83fb3ef5c2bb4e4cba7dc9c923c918fe3847a1"
STEP3_LOCK="/tmp/ai_surveillance_camera_v11_step3_tracker_v2.lock"
STEP2_LOCK="/tmp/ai_surveillance_camera_v11_step2_production_v25.lock"
DISPLAY_LOG="${V11_STEP3_DISPLAY_LOG:-/tmp/CAMERA_V11_STEP3_DISPLAY.log}"
TRACKER_LOG="${V11_STEP3_TRACKER_LOG:-/tmp/CAMERA_V11_STEP3_TRACKER.log}"
ENGINE="$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine"

fail() {
  printf 'CAMERA_V11_STEP3_V2_PREFLIGHT result=FAIL reason=%s\n' "$*" >&2
  exit 1
}

command -v flock >/dev/null 2>&1 || fail "flock_missing"
exec 8>"$STEP3_LOCK"
flock -n 8 || fail "another_step3_launcher_holds=$STEP3_LOCK"
# Also hold the frozen Step2 production lock for the whole run so Step2 and Step3
# cannot accidentally double-open camera/GPU resources.
exec 7>"$STEP2_LOCK"
flock -n 7 || fail "step2_or_other_step3_holds=$STEP2_LOCK"
[[ -n "${DISPLAY:-}" ]] || fail "DISPLAY_empty"
[[ -s "$ENGINE" ]] || fail "fp32_engine_missing"

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
  scripts/yolo26_trt86_step2_worker.py \
  scripts/run_camera_v11_step1_v7.sh \
  scripts/check_camera_v11_step1_v7_log.py \
  scripts/camera_v11_powermizer_keeper_v25.sh \
  scripts/run_camera_v11_step2_production_fp32_v25.sh \
  scripts/check_camera_v11_step1_v25_aggregate_log.py \
  scripts/check_camera_v11_step2_production_log_v25.py \
  || fail "frozen_step2_files_changed"

CONFLICT_PATTERN='services\.camera_v11\.(step1_cam02_lowlat_v7|step2_production_fp32(_v[0-9]+)?|step3_tracking_v[0-9]+)|yolo26_trt86_step2_worker\.py|build_yolo26s_b1_.*trt86\.py'
conflicts="$(pgrep -af "$CONFLICT_PATTERN" || true)"
[[ -z "$conflicts" ]] || fail $'conflicting_camera_or_trt_process:\n'"$conflicts"

# shellcheck source=/dev/null
source "$ROOT/scripts/camera_v11_powermizer_keeper_v25.sh"

# NVIDIA 580 can report a successful GPUPowerMizerMode CLI assignment without
# actually applying it. With the nvidia-settings GUI kept alive, re-issuing the
# assignment while CUDA work is active reliably exposes whether the policy took
# effect. This is bounded and fail-closed: no clock locking or overclocking.
v11_step3_ensure_vram_boost() {
  local minimum_mhz="${V11_POWERMIZER_MIN_MEMORY_MHZ:-3000}"
  local attempts="${V11_STEP3_POWERMIZER_REAPPLY_ATTEMPTS:-20}"
  local delay="${V11_STEP3_POWERMIZER_REAPPLY_DELAY_SEC:-0.25}"
  local clock=""
  local attempt=0

  [[ "$minimum_mhz" =~ ^[0-9]+$ ]] || return 1
  [[ "$attempts" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ -n "${V11_POWERMIZER_KEEPER_PID:-}" ]] && kill -0 "$V11_POWERMIZER_KEEPER_PID" 2>/dev/null || return 1

  for attempt in $(seq 1 "$attempts"); do
    clock="$(v11_powermizer_mem_clock_mhz || true)"
    if [[ "$clock" =~ ^[0-9]+$ ]] && (( clock >= minimum_mhz )); then
      printf 'CAMERA_V11_POWERMIZER_KEEPER result=BOOST_OK memory_mhz=%s minimum_mhz=%s pid=%s reapply_attempt=%s\n' \
        "$clock" "$minimum_mhz" "$V11_POWERMIZER_KEEPER_PID" "$((attempt - 1))"
      return 0
    fi

    DISPLAY="$DISPLAY" nvidia-settings -a '[gpu:0]/GPUPowerMizerMode=1' \
      >>"$V11_POWERMIZER_KEEPER_LOG" 2>&1 || true
    sleep "$delay"
  done

  clock="$(v11_powermizer_mem_clock_mhz || true)"
  if [[ "$clock" =~ ^[0-9]+$ ]] && (( clock >= minimum_mhz )); then
    printf 'CAMERA_V11_POWERMIZER_KEEPER result=BOOST_OK memory_mhz=%s minimum_mhz=%s pid=%s reapply_attempt=%s\n' \
      "$clock" "$minimum_mhz" "$V11_POWERMIZER_KEEPER_PID" "$attempts"
    return 0
  fi

  printf 'CAMERA_V11_POWERMIZER_KEEPER result=FAIL reason=memory_clock_not_boosted_after_reapply memory_mhz=%s minimum_mhz=%s attempts=%s\n' \
    "${clock:-unknown}" "$minimum_mhz" "$attempts" >&2
  return 1
}

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
}
on_signal() {
  cleanup
  exit 130
}
trap cleanup EXIT
trap on_signal INT TERM

: >"$DISPLAY_LOG"
: >"$TRACKER_LOG"
v11_powermizer_start || fail "powermizer_keeper_start"

printf 'CAMERA_V11_STEP3_V2_PREFLIGHT result=PASS frozen_step2_sha=%s power_keeper=1 display_log=%s tracker_log=%s\n' \
  "$FROZEN_STEP2_SHA" "$DISPLAY_LOG" "$TRACKER_LOG"

bash "$ROOT/scripts/run_camera_v11_step1_v7.sh" >"$DISPLAY_LOG" 2>&1 &
display_pid=$!
sleep "${V11_STEP3_DISPLAY_WARMUP_SEC:-8}"
kill -0 "$display_pid" 2>/dev/null || fail "display_exited_during_warmup"

export V11_STEP2_MODE=full
export V11_STEP2_HZ="${V11_STEP3_HZ:-2.0}"
export V11_STEP2_CONF="${V11_STEP3_CONF:-0.18}"
export V11_STEP2_ENGINE="$ENGINE"
export V11_STEP2_TRT86_PYTHON="$ROOT/.venv-trt86/bin/python"
export V11_STEP2_TRT86_WORKER="$ROOT/scripts/yolo26_trt86_step2_worker.py"

"$ROOT/.venv/bin/python" -u -m services.camera_v11.step3_tracking_v2 >"$TRACKER_LOG" 2>&1 &
tracker_pid=$!

ready=0
for _ in $(seq 1 300); do
  if grep -q 'CAMERA_V11_STEP2_WARMUP iterations=10 status=OK' "$TRACKER_LOG"; then
    ready=1
    break
  fi
  kill -0 "$tracker_pid" 2>/dev/null || break
  sleep 0.1
done
(( ready == 1 )) || fail "tracker_detector_warmup_failed"

v11_step3_ensure_vram_boost || fail "vram_boost_gate"

printf 'CAMERA_V11_STEP3_V2_RUNNING display_pid=%s tracker_pid=%s keeper_pid=%s\n' \
  "$display_pid" "$tracker_pid" "$V11_POWERMIZER_KEEPER_PID"

while kill -0 "$display_pid" 2>/dev/null && kill -0 "$tracker_pid" 2>/dev/null; do
  sleep 1
done

kill -0 "$display_pid" 2>/dev/null || wait "$display_pid"
wait "$tracker_pid"
