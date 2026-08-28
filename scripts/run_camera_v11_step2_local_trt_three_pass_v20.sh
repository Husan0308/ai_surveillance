#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
BASE="rebuild/service-architecture-v11-clean-step1-cam02-lowlat-v7-20260827"
OUT="${V11_STEP2_LOCAL_THREE_PASS_OUT:-/tmp/camera_v11_step2_local_trt_three_pass_v20}"
DURATION="${V11_STEP2_LOCAL_THREE_PASS_DURATION_SEC:-60}"
WARMUP="${V11_STEP2_LOCAL_THREE_PASS_DISPLAY_WARMUP_SEC:-8}"
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

failed=0
for run_number in 1 2 3; do
  cleanup
  "$ROOT/scripts/prime_camera_v11_trt86_memory_clock_v20.sh" || exit 1
  prefix="$OUT/full_$run_number"
  display_log="$prefix.display.log"
  detector_log="$prefix.detector.log"
  dmon_log="$prefix.dmon.log"
  pidstat_log="$prefix.pidstat.log"
  check_log="$prefix.check.log"
  : >"$display_log"
  : >"$detector_log"
  : >"$dmon_log"
  : >"$pidstat_log"
  : >"$check_log"

  printf 'CAMERA_V11_STEP2_LOCAL_THREE_PASS_START run=%s duration=%ss display_warmup=%ss analytics_nice=%s\n' \
    "$run_number" "$DURATION" "$WARMUP" "${V11_STEP2_NICE:-10}"
  bash "$ROOT/scripts/run_camera_v11_step1_v7.sh" >"$display_log" 2>&1 &
  display_pid=$!
  sleep "$WARMUP"
  kill -0 "$display_pid" 2>/dev/null || {
    failed=1
    break
  }
  "$ROOT/scripts/run_camera_v11_step2_stage_local_trt_v20.sh" full >"$detector_log" 2>&1 &
  detector_pid=$!
  ready=0
  for _ in $(seq 1 300); do
    grep -q 'CAMERA_V11_STEP2_WARMUP iterations=10 status=OK' "$detector_log" && {
      ready=1
      break
    }
    kill -0 "$detector_pid" 2>/dev/null || break
    sleep 0.1
  done
  if (( ready != 1 )); then
    failed=1
    break
  fi
  nvidia-smi dmon -s pucvmet -d 1 >"$dmon_log" 2>&1 &
  dmon_pid=$!
  pidstat -h -u -r -p "$display_pid,$detector_pid" 1 >"$pidstat_log" 2>&1 &
  pidstat_pid=$!
  sleep "$DURATION"
  cleanup

  "$ROOT/.venv/bin/python" "$ROOT/scripts/check_camera_v11_step2_production_log_v15.py" \
    --display-log "$display_log" --detector-log "$detector_log" | tee "$check_log"
  check_status=${PIPESTATUS[0]}
  "$ROOT/.venv/bin/python" "$ROOT/scripts/summarize_camera_v11_step2_resources.py" \
    --dmon "$dmon_log" --pidstat "$pidstat_log" | tee -a "$check_log" || check_status=1
  grep 'CAMERA_V11_STEP2_PROFILE' "$detector_log" | tail -n 1 | tee -a "$check_log"
  grep 'CAMERA_V11_STEP2_SOURCE' "$detector_log" | tail -n 1 | tee -a "$check_log"
  grep 'CAMERA_V11_STEP2_V12_LATEST' "$detector_log" | tail -n 1 | tee -a "$check_log"
  grep 'CAMERA_V11_STEP2_V13_CREDIT_STATS' "$detector_log" | tail -n 1 | tee -a "$check_log"
  if (( check_status != 0 )); then
    printf 'CAMERA_V11_STEP2_LOCAL_THREE_PASS_RESULT run=%s result=FAIL\n' "$run_number" | tee -a "$check_log"
    failed=1
    break
  fi
  printf 'CAMERA_V11_STEP2_LOCAL_THREE_PASS_RESULT run=%s result=PASS\n' "$run_number" | tee -a "$check_log"
done

if (( failed == 0 )); then
  printf 'CAMERA_V11_STEP2_LOCAL_THREE_PASS_FINAL result=PASS consecutive=3 duration=%ss analytics_nice=%s\n' \
    "$DURATION" "${V11_STEP2_NICE:-10}"
else
  printf 'CAMERA_V11_STEP2_LOCAL_THREE_PASS_FINAL result=FAIL duration=%ss analytics_nice=%s\n' \
    "$DURATION" "${V11_STEP2_NICE:-10}"
fi
exit "$failed"
