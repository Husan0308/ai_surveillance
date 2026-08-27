#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="${V11_STEP2_STAGE_OUT:-/tmp/camera_v11_step2_stages_v18}"
DURATION="${V11_STEP2_STAGE_DURATION_SEC:-20}"
WARMUP="${V11_STEP2_STAGE_DISPLAY_WARMUP_SEC:-5}"
mkdir -p "$OUT"
display_pid=""; detector_pid=""; dmon_pid=""; pidstat_pid=""
cleanup() {
  for pid in "$pidstat_pid" "$dmon_pid" "$detector_pid" "$display_pid"; do
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "$pidstat_pid" "$dmon_pid" "$detector_pid" "$display_pid"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
  display_pid=""; detector_pid=""; dmon_pid=""; pidstat_pid=""
}
trap cleanup EXIT INT TERM

failed=0
for stage in display-only extraction preprocessing synthetic-trt full; do
  cleanup
  display_log="$OUT/${stage}.display.log"; detector_log="$OUT/${stage}.detector.log"
  dmon_log="$OUT/${stage}.dmon.log"; pidstat_log="$OUT/${stage}.pidstat.log"
  check_log="$OUT/${stage}.check.log"
  : >"$display_log"; : >"$detector_log"; : >"$dmon_log"; : >"$pidstat_log"; : >"$check_log"
  printf 'CAMERA_V11_STEP2_STAGE_START stage=%s duration=%ss warmup=%ss\n' "$stage" "$DURATION" "$WARMUP"
  bash "$ROOT/scripts/run_camera_v11_step1_v7.sh" >"$display_log" 2>&1 & display_pid=$!
  sleep "$WARMUP"
  if ! kill -0 "$display_pid" 2>/dev/null; then
    printf 'CAMERA_V11_STEP2_STAGE_RESULT stage=%s display=FAIL reason=warmup_exit\n' "$stage"
    failed=1
    continue
  fi
  if [[ "$stage" != "display-only" ]]; then
    "$ROOT/scripts/run_camera_v11_step2_stage_v18.sh" "$stage" >"$detector_log" 2>&1 & detector_pid=$!
  fi
  nvidia-smi dmon -s pucvmet -d 1 >"$dmon_log" 2>&1 & dmon_pid=$!
  pid_list="$display_pid"; [[ -n "$detector_pid" ]] && pid_list="$pid_list,$detector_pid"
  pidstat -h -u -r -p "$pid_list" 1 >"$pidstat_log" 2>&1 & pidstat_pid=$!
  sleep "$DURATION"
  cleanup
  if "$ROOT/.venv/bin/python" "$ROOT/scripts/check_camera_v11_step1_v7_log.py" "$display_log" | tee "$check_log"; then
    display_status=PASS
  else
    display_status=FAIL; failed=1
  fi
  "$ROOT/.venv/bin/python" "$ROOT/scripts/summarize_camera_v11_step2_resources.py" \
    --dmon "$dmon_log" --pidstat "$pidstat_log" | tee -a "$check_log" || failed=1
  if [[ "$stage" == "full" ]]; then
    "$ROOT/.venv/bin/python" "$ROOT/scripts/check_camera_v11_step2_production_log_v15.py" \
      --display-log "$display_log" --detector-log "$detector_log" | tee -a "$check_log" || failed=1
  fi
  printf 'CAMERA_V11_STEP2_STAGE_RESULT stage=%s display=%s logs=%s\n' "$stage" "$display_status" "$OUT"
done
exit "$failed"
