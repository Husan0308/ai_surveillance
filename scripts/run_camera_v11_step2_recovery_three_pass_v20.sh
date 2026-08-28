#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
BASE="rebuild/service-architecture-v11-clean-step1-cam02-lowlat-v7-20260827"
OUT="${V11_STEP2_RECOVERY_OUT:-/tmp/camera_v11_step2_recovery_three_pass_v20}"
DURATION="${V11_STEP2_RECOVERY_DURATION_SEC:-60}"
WARMUP="${V11_STEP2_RECOVERY_DISPLAY_WARMUP_SEC:-8}"
mkdir -p "$OUT"

display_pid=""
detector_pid=""
dmon_pid=""
pidstat_pid=""
cleanup() {
  for pid in "$pidstat_pid" "$dmon_pid" "$detector_pid" "$display_pid"; do
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "$pidstat_pid" "$dmon_pid" "$detector_pid" "$display_pid"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
  display_pid=""
  detector_pid=""
  dmon_pid=""
  pidstat_pid=""
}
trap cleanup EXIT INT TERM

git diff --quiet "$BASE" -- services/camera_v11/step1_cam02_lowlat_v7.py \
  services/camera_v11/step1_independent_egl_v4.py scripts/run_camera_v11_step1_v7.sh \
  scripts/check_camera_v11_step1_v7_log.py || exit 1

run_display_only() {
  local prefix="$OUT/display_only"
  local display_log="$prefix.display.log"
  local dmon_log="$prefix.dmon.log"
  local pidstat_log="$prefix.pidstat.log"
  local check_log="$prefix.check.log"
  cleanup
  "$ROOT/scripts/prime_camera_v11_trt86_memory_clock_v20.sh" || return 1
  : >"$display_log"
  : >"$dmon_log"
  : >"$pidstat_log"
  : >"$check_log"
  printf 'CAMERA_V11_STEP2_RECOVERY_DISPLAY_ONLY_START duration=%ss\n' "$DURATION"
  bash "$ROOT/scripts/run_camera_v11_step1_burst_backpressure_v20.sh" >"$display_log" 2>&1 &
  display_pid=$!
  sleep "$WARMUP"
  kill -0 "$display_pid" 2>/dev/null || return 1
  nvidia-smi dmon -s pucvmet -d 1 >"$dmon_log" 2>&1 &
  dmon_pid=$!
  pidstat -h -u -r -p "$display_pid" 1 >"$pidstat_log" 2>&1 &
  pidstat_pid=$!
  sleep "$DURATION"
  cleanup
  "$ROOT/.venv/bin/python" "$ROOT/scripts/check_camera_v11_step1_burst_backpressure_v20_log.py" \
    "$display_log" | tee "$check_log"
  local check_status=${PIPESTATUS[0]}
  "$ROOT/.venv/bin/python" "$ROOT/scripts/summarize_camera_v11_step2_resources.py" \
    --dmon "$dmon_log" --pidstat "$pidstat_log" | tee -a "$check_log" || check_status=1
  if (( check_status == 0 )); then
    printf 'CAMERA_V11_STEP2_RECOVERY_DISPLAY_ONLY_RESULT result=PASS\n' | tee -a "$check_log"
  else
    printf 'CAMERA_V11_STEP2_RECOVERY_DISPLAY_ONLY_RESULT result=FAIL\n' | tee -a "$check_log"
  fi
  return "$check_status"
}

run_full() {
  local run_number="$1"
  local prefix="$OUT/full_$run_number"
  local display_log="$prefix.display.log"
  local detector_log="$prefix.detector.log"
  local dmon_log="$prefix.dmon.log"
  local pidstat_log="$prefix.pidstat.log"
  local check_log="$prefix.check.log"
  cleanup
  "$ROOT/scripts/prime_camera_v11_trt86_memory_clock_v20.sh" || return 1
  : >"$display_log"
  : >"$detector_log"
  : >"$dmon_log"
  : >"$pidstat_log"
  : >"$check_log"
  printf 'CAMERA_V11_STEP2_RECOVERY_FULL_START run=%s duration=%ss display_warmup=%ss analytics_nice=%s\n' \
    "$run_number" "$DURATION" "$WARMUP" "${V11_STEP2_NICE:-10}"
  bash "$ROOT/scripts/run_camera_v11_step1_burst_backpressure_v20.sh" >"$display_log" 2>&1 &
  display_pid=$!
  sleep "$WARMUP"
  kill -0 "$display_pid" 2>/dev/null || return 1
  "$ROOT/scripts/run_camera_v11_step2_stage_local_trt_v20.sh" full >"$detector_log" 2>&1 &
  detector_pid=$!
  local ready=0
  for _ in $(seq 1 300); do
    grep -q 'CAMERA_V11_STEP2_WARMUP iterations=10 status=OK' "$detector_log" && {
      ready=1
      break
    }
    kill -0 "$detector_pid" 2>/dev/null || break
    sleep 0.1
  done
  (( ready == 1 )) || return 1
  nvidia-smi dmon -s pucvmet -d 1 >"$dmon_log" 2>&1 &
  dmon_pid=$!
  pidstat -h -u -r -p "$display_pid,$detector_pid" 1 >"$pidstat_log" 2>&1 &
  pidstat_pid=$!
  sleep "$DURATION"
  cleanup

  "$ROOT/.venv/bin/python" "$ROOT/scripts/check_camera_v11_step1_burst_backpressure_v20_log.py" \
    "$display_log" | tee "$check_log"
  local check_status=${PIPESTATUS[0]}
  "$ROOT/.venv/bin/python" "$ROOT/scripts/check_camera_v11_step2_production_log_v15.py" \
    --display-log "$display_log" --detector-log "$detector_log" | tee -a "$check_log"
  (( PIPESTATUS[0] == 0 )) || check_status=1
  "$ROOT/.venv/bin/python" "$ROOT/scripts/summarize_camera_v11_step2_resources.py" \
    --dmon "$dmon_log" --pidstat "$pidstat_log" | tee -a "$check_log" || check_status=1
  grep 'CAMERA_V11_STEP2_PROFILE' "$detector_log" | tail -n 1 | tee -a "$check_log"
  grep 'CAMERA_V11_STEP2_SOURCE' "$detector_log" | tail -n 1 | tee -a "$check_log"
  grep 'CAMERA_V11_STEP2_V12_LATEST' "$detector_log" | tail -n 1 | tee -a "$check_log"
  grep 'CAMERA_V11_STEP2_V13_CREDIT_STATS' "$detector_log" | tail -n 1 | tee -a "$check_log"
  if (( check_status == 0 )); then
    printf 'CAMERA_V11_STEP2_RECOVERY_FULL_RESULT run=%s result=PASS\n' "$run_number" | tee -a "$check_log"
  else
    printf 'CAMERA_V11_STEP2_RECOVERY_FULL_RESULT run=%s result=FAIL\n' "$run_number" | tee -a "$check_log"
  fi
  return "$check_status"
}

failed=0
run_display_only || failed=1
if (( failed == 0 )); then
  for run_number in 1 2 3; do
    run_full "$run_number" || {
      failed=1
      break
    }
  done
fi

if (( failed == 0 )); then
  printf 'CAMERA_V11_STEP2_RECOVERY_FINAL result=PASS display_only=1 consecutive_full=3 duration=%ss\n' "$DURATION"
else
  printf 'CAMERA_V11_STEP2_RECOVERY_FINAL result=FAIL duration=%ss\n' "$DURATION"
fi
exit "$failed"
