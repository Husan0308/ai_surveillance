#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
BASE="rebuild/service-architecture-v11-clean-step1-cam02-lowlat-v7-20260827"
OUT="${V11_STEP2_ACCEPTANCE_OUT:-/tmp/camera_v11_step2_acceptance_v20}"
DURATION="${V11_STEP2_ACCEPTANCE_DURATION_SEC:-60}"
WARMUP="${V11_STEP2_ACCEPTANCE_DISPLAY_WARMUP_SEC:-8}"
mkdir -p "$OUT"

display_pid=""
detector_pid=""
worker_pid=""
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
  worker_pid=""
  dmon_pid=""
  pidstat_pid=""
}
trap cleanup EXIT INT TERM

git diff --quiet "$BASE" -- services/camera_v11/step1_cam02_lowlat_v7.py \
  services/camera_v11/step1_independent_egl_v4.py scripts/run_camera_v11_step1_v7.sh \
  scripts/check_camera_v11_step1_v7_log.py || {
    printf 'CAMERA_V11_STEP2_ACCEPTANCE result=FAIL reason=frozen_step1_diff\n'
    exit 1
  }

failed=0
for run in display_only full_1 full_2 full_3; do
  cleanup
  "$ROOT/scripts/prime_camera_v11_trt86_memory_clock_v20.sh" || exit 1
  display_log="$OUT/$run.display.log"
  detector_log="$OUT/$run.detector.log"
  dmon_log="$OUT/$run.dmon.log"
  pidstat_log="$OUT/$run.pidstat.log"
  check_log="$OUT/$run.check.log"
  : >"$display_log"
  : >"$detector_log"
  : >"$dmon_log"
  : >"$pidstat_log"
  : >"$check_log"

  printf 'CAMERA_V11_STEP2_ACCEPTANCE_START run=%s duration=%ss display_warmup=%ss\n' \
    "$run" "$DURATION" "$WARMUP"
  bash "$ROOT/scripts/run_camera_v11_step1_v7.sh" >"$display_log" 2>&1 &
  display_pid=$!
  sleep "$WARMUP"
  if ! kill -0 "$display_pid" 2>/dev/null; then
    printf 'CAMERA_V11_STEP2_ACCEPTANCE_RESULT run=%s result=FAIL reason=display_warmup_exit\n' \
      "$run" | tee "$check_log"
    failed=1
    continue
  fi

  pid_list="$display_pid"
  if [[ "$run" != "display_only" ]]; then
    "$ROOT/scripts/run_camera_v11_step2_stage_v18.sh" full >"$detector_log" 2>&1 &
    detector_pid=$!
    ready=0
    for _ in $(seq 1 300); do
      if ! kill -0 "$detector_pid" 2>/dev/null; then
        break
      fi
      if grep -q 'CAMERA_V11_STEP2_WARMUP iterations=10 status=OK' "$detector_log"; then
        ready=1
        break
      fi
      sleep 0.1
    done
    if (( ready != 1 )); then
      printf 'CAMERA_V11_STEP2_ACCEPTANCE_RESULT run=%s result=FAIL reason=detector_warmup\n' \
        "$run" | tee "$check_log"
      failed=1
      continue
    fi
    worker_pid="$(pgrep -P "$detector_pid" -f 'yolo26_trt86_step2_worker.py' | head -n 1 || true)"
    [[ -n "$worker_pid" ]] || {
      printf 'CAMERA_V11_STEP2_ACCEPTANCE_RESULT run=%s result=FAIL reason=worker_pid_missing\n' \
        "$run" | tee "$check_log"
      failed=1
      continue
    }
    pid_list="$display_pid,$detector_pid,$worker_pid"
  fi

  nvidia-smi dmon -s pucvmet -d 1 >"$dmon_log" 2>&1 &
  dmon_pid=$!
  pidstat -h -u -r -p "$pid_list" 1 >"$pidstat_log" 2>&1 &
  pidstat_pid=$!
  sleep "$DURATION"
  cleanup

  if [[ "$run" == "display_only" ]]; then
    "$ROOT/.venv/bin/python" "$ROOT/scripts/check_camera_v11_step1_v7_log.py" \
      "$display_log" | tee "$check_log"
    check_status=${PIPESTATUS[0]}
  else
    "$ROOT/.venv/bin/python" "$ROOT/scripts/check_camera_v11_step2_production_log_v15.py" \
      --display-log "$display_log" --detector-log "$detector_log" | tee "$check_log"
    check_status=${PIPESTATUS[0]}
  fi
  (( check_status == 0 )) || failed=1
  "$ROOT/.venv/bin/python" "$ROOT/scripts/summarize_camera_v11_step2_resources.py" \
    --dmon "$dmon_log" --pidstat "$pidstat_log" | tee -a "$check_log" || failed=1
  grep 'CAMERA_V11_STEP2_PROFILE' "$detector_log" | tail -n 1 | tee -a "$check_log" || true
  grep 'CAMERA_V11_STEP2_SOURCE' "$detector_log" | tail -n 1 | tee -a "$check_log" || true
  grep 'CAMERA_V11_STEP2_V12_LATEST' "$detector_log" | tail -n 1 | tee -a "$check_log" || true
  grep 'CAMERA_V11_STEP2_V13_CREDIT_STATS' "$detector_log" | tail -n 1 | tee -a "$check_log" || true
  if (( check_status == 0 )); then
    result=PASS
  else
    result=FAIL
  fi
  printf 'CAMERA_V11_STEP2_ACCEPTANCE_RESULT run=%s result=%s logs=%s\n' \
    "$run" "$result" "$OUT" | tee -a "$check_log"
done

if (( failed == 0 )); then
  printf 'CAMERA_V11_STEP2_ACCEPTANCE_FINAL result=PASS display_only=1 full_consecutive=3 duration=%ss\n' \
    "$DURATION"
else
  printf 'CAMERA_V11_STEP2_ACCEPTANCE_FINAL result=FAIL display_only=1 full_consecutive=3 duration=%ss\n' \
    "$DURATION"
fi
exit "$failed"
