#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="${V11_STEP2_SWEEP_OUT:-/tmp/camera_v11_step2_synthetic_sweep_v20}"
DURATION="${V11_STEP2_SWEEP_DURATION_SEC:-30}"
WARMUP="${V11_STEP2_SWEEP_DISPLAY_WARMUP_SEC:-8}"
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

"$ROOT/scripts/prime_camera_v11_trt86_memory_clock_v20.sh" || exit 1
failed=0
for total_hz in 1 3 6 9 12; do
  cleanup
  per_camera_hz="$(awk -v value="$total_hz" 'BEGIN { printf "%.6f", value / 6.0 }')"
  prefix="$OUT/total_${total_hz}hz"
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

  printf 'CAMERA_V11_STEP2_SWEEP_START total_hz=%s per_camera_hz=%s duration=%ss warmup=%ss\n' \
    "$total_hz" "$per_camera_hz" "$DURATION" "$WARMUP"
  bash "$ROOT/scripts/run_camera_v11_step1_v7.sh" >"$display_log" 2>&1 &
  display_pid=$!
  sleep "$WARMUP"
  if ! kill -0 "$display_pid" 2>/dev/null; then
    printf 'CAMERA_V11_STEP2_SWEEP_RESULT total_hz=%s result=FAIL reason=display_warmup_exit\n' \
      "$total_hz" | tee "$check_log"
    failed=1
    continue
  fi

  V11_STEP2_HZ="$per_camera_hz" \
    "$ROOT/scripts/run_camera_v11_step2_stage_v18.sh" synthetic-trt >"$detector_log" 2>&1 &
  detector_pid=$!
  nvidia-smi dmon -s pucvmet -d 1 >"$dmon_log" 2>&1 &
  dmon_pid=$!
  pidstat -h -u -r -p "$display_pid,$detector_pid" 1 >"$pidstat_log" 2>&1 &
  pidstat_pid=$!
  sleep "$DURATION"
  cleanup

  if "$ROOT/.venv/bin/python" "$ROOT/scripts/check_camera_v11_step1_v7_log.py" \
      "$display_log" | tee "$check_log"; then
    display_status=PASS
  else
    display_status=FAIL
    failed=1
  fi
  "$ROOT/.venv/bin/python" "$ROOT/scripts/summarize_camera_v11_step2_resources.py" \
    --dmon "$dmon_log" --pidstat "$pidstat_log" | tee -a "$check_log" || failed=1
  grep 'CAMERA_V11_STEP2_PROFILE' "$detector_log" | tail -n 1 | tee -a "$check_log"
  grep 'CAMERA_V11_STEP2_SOURCE' "$detector_log" | tail -n 1 | tee -a "$check_log"
  printf 'CAMERA_V11_STEP2_SWEEP_RESULT total_hz=%s per_camera_hz=%s display=%s logs=%s\n' \
    "$total_hz" "$per_camera_hz" "$display_status" "$OUT" | tee -a "$check_log"
done
exit "$failed"
