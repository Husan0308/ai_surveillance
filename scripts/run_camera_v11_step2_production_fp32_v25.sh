#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
FROZEN_STEP1_SHA="dfb88d9850e8d73b14b06e78b3f884d8c01a5788"
LOCK_FILE="/tmp/ai_surveillance_camera_v11_step2_production_v25.lock"
DISPLAY_LOG="${V11_STEP2_DISPLAY_LOG:-/tmp/CAMERA_V11_STEP2_DISPLAY.log}"
DETECTOR_LOG="${V11_STEP2_DETECTOR_LOG:-/tmp/CAMERA_V11_STEP2_DETECTOR.log}"
ENGINE="$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine"

fail() {
  printf 'CAMERA_V11_STEP2_V25_PREFLIGHT result=FAIL reason=%s\n' "$*" >&2
  exit 1
}

command -v flock >/dev/null 2>&1 || fail "flock_missing"
exec 8>"$LOCK_FILE"
flock -n 8 || fail "another_launcher_holds=$LOCK_FILE"
[[ -n "${DISPLAY:-}" ]] || fail "DISPLAY_empty"
[[ -s "$ENGINE" ]] || fail "fp32_engine_missing"

# Frozen Step1 remains byte-for-byte authoritative.
git diff --quiet "$FROZEN_STEP1_SHA" -- \
  services/camera_v11/step1_cam02_lowlat_v7.py \
  services/camera_v11/step1_independent_egl_v4.py \
  scripts/run_camera_v11_step1_v7.sh \
  scripts/check_camera_v11_step1_v7_log.py \
  || fail "frozen_step1_differs"

# shellcheck source=/dev/null
source "$ROOT/scripts/camera_v11_powermizer_keeper_v25.sh"

display_pid=""
detector_pid=""
cleanup() {
  trap - EXIT INT TERM
  for pid in "$detector_pid" "$display_pid"; do
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "$detector_pid" "$display_pid"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
  v11_powermizer_stop || true
}
trap cleanup EXIT INT TERM

: >"$DISPLAY_LOG"
: >"$DETECTOR_LOG"
v11_powermizer_start || fail "powermizer_keeper_start"

printf 'CAMERA_V11_STEP2_V25_PREFLIGHT result=PASS frozen_step1_sha=%s precision=fp32 batch=1 power_keeper=1 display_log=%s detector_log=%s\n' \
  "$FROZEN_STEP1_SHA" "$DISPLAY_LOG" "$DETECTOR_LOG"

bash "$ROOT/scripts/run_camera_v11_step1_v7.sh" >"$DISPLAY_LOG" 2>&1 &
display_pid=$!
sleep "${V11_STEP2_DISPLAY_WARMUP_SEC:-8}"
kill -0 "$display_pid" 2>/dev/null || fail "display_exited_during_warmup"

"$ROOT/scripts/run_camera_v11_step2_stage_v18.sh" full >"$DETECTOR_LOG" 2>&1 &
detector_pid=$!

# TensorRT warmup supplies the load that should put VRAM at its performance clock.
ready=0
for _ in $(seq 1 300); do
  if grep -q 'CAMERA_V11_STEP2_WARMUP iterations=10 status=OK' "$DETECTOR_LOG"; then
    ready=1
    break
  fi
  kill -0 "$detector_pid" 2>/dev/null || break
  sleep 0.1
done
(( ready == 1 )) || fail "detector_warmup_failed"
v11_powermizer_verify_boost || fail "vram_boost_gate"

printf 'CAMERA_V11_STEP2_V25_RUNNING display_pid=%s detector_pid=%s keeper_pid=%s\n' \
  "$display_pid" "$detector_pid" "$V11_POWERMIZER_KEEPER_PID"

while kill -0 "$display_pid" 2>/dev/null && kill -0 "$detector_pid" 2>/dev/null; do
  sleep 1
done

kill -0 "$display_pid" 2>/dev/null || wait "$display_pid"
wait "$detector_pid"
