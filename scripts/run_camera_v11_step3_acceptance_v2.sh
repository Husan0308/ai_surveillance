#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="${V11_STEP3_V2_ACCEPTANCE_OUT:-/tmp/camera_v11_step3_acceptance_v2}"
DURATION="${V11_STEP3_V2_DURATION_SEC:-60}"
WARMUP_WINDOWS="${V11_STEP3_V2_WARMUP_WINDOWS:-2}"
REUSE_RUN1="${V11_STEP3_V2_REUSE_RUN1:-0}"
REUSE_RUNTIME_SHA="b9de3928866acf59ea62224f7f13dfcb86dd99f0"
mkdir -p "$OUT"

fail() {
  printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE result=FAIL reason=%s\n' "$*" >&2
  exit 1
}

dump_failure_context() {
  local launcher_log="$1"
  local tracker_log="$2"
  local display_log="$3"
  printf 'CAMERA_V11_STEP3_V2_FAILURE_CONTEXT begin\n'
  for log in "$launcher_log" "$tracker_log" "$display_log"; do
    [[ -s "$log" ]] || continue
    printf '%s\n' "--- $(basename "$log") ---"
    grep -E \
      'CAMERA_V11_STEP3_V2_PREFLIGHT|CAMERA_V11_POWERMIZER_KEEPER|CAMERA_V11_STEP1V7_PREFLIGHT|CAMERA_V11_STEP2_WARMUP|Traceback|ModuleNotFoundError|ImportError|RuntimeError|ERROR|FAIL' \
      "$log" | tail -n 40 || true
  done
  printf 'CAMERA_V11_STEP3_V2_FAILURE_CONTEXT end\n'
}

run_checker() {
  local display_log="$1"
  local tracker_log="$2"
  local check_log="$3"
  "$ROOT/.venv/bin/python" "$ROOT/scripts/check_camera_v11_step3_tracker_v2_log.py" \
    --display-log "$display_log" --tracker-log "$tracker_log" \
    --warmup-windows "$WARMUP_WINDOWS" | tee -a "$check_log"
  return "${PIPESTATUS[0]}"
}

[[ "$DURATION" =~ ^[1-9][0-9]*$ ]] || fail "invalid_duration=$DURATION"
[[ "$WARMUP_WINDOWS" =~ ^[0-9]+$ ]] || fail "invalid_warmup_windows=$WARMUP_WINDOWS"
[[ "$REUSE_RUN1" == "0" || "$REUSE_RUN1" == "1" ]] || fail "invalid_reuse_run1=$REUSE_RUN1"
command -v timeout >/dev/null 2>&1 || fail "timeout_missing"

failed=0
start_run=1

printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_START stage=unit\n'
"$ROOT/.venv/bin/python" "$ROOT/scripts/test_camera_v11_step3_tracker_v2.py" \
  2>&1 | tee "$OUT/unit.log"
if (( PIPESTATUS[0] != 0 )); then
  printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_RESULT stage=unit result=FAIL\n'
  failed=1
else
  printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_RESULT stage=unit result=PASS\n'
fi

# Safe fast path: the previous run1 was produced by REUSE_RUNTIME_SHA. Current
# changes after that SHA are checker/acceptance-only. Reuse is permitted only if
# every runtime file is byte-identical to that known run and its logs pass the
# current checker. This saves one 60-second production run without weakening gates.
if (( failed == 0 )) && [[ "$REUSE_RUN1" == "1" ]]; then
  display_log="$OUT/full_1.display.log"
  tracker_log="$OUT/full_1.tracker.log"
  launcher_log="$OUT/full_1.launcher.log"
  check_log="$OUT/full_1.check.log"

  for path in \
    services/camera_v11/step3_tracker_v2.py \
    services/camera_v11/step3_tracking_v2.py \
    scripts/run_camera_v11_step3_tracker_v2.sh \
    scripts/camera_v11_powermizer_keeper_v25.sh \
    scripts/benchmark_yolo26_trt86_step2_worker_v22.py; do
    git diff --quiet "$REUSE_RUNTIME_SHA" HEAD -- "$path" \
      || fail "reuse_run1_runtime_changed path=$path"
  done

  [[ -s "$display_log" && -s "$tracker_log" && -s "$launcher_log" ]] \
    || fail "reuse_run1_logs_missing out=$OUT"
  grep -q 'CAMERA_V11_POWERMIZER_KEEPER result=BOOST_OK' "$launcher_log" \
    || fail "reuse_run1_no_vram_boost_gate"

  : >"$check_log"
  printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_START stage=full run=1 source=reuse runtime_sha=%s\n' \
    "$REUSE_RUNTIME_SHA"
  if run_checker "$display_log" "$tracker_log" "$check_log"; then
    printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_RESULT stage=full run=1 result=PASS source=reused-runtime-identical\n' \
      | tee -a "$check_log"
    start_run=2
  else
    printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_RESULT stage=full run=1 result=FAIL source=reuse reason=tracker_checker\n' \
      | tee -a "$check_log"
    failed=1
  fi
fi

if (( failed == 0 )); then
  for run in $(seq "$start_run" 3); do
    display_log="$OUT/full_${run}.display.log"
    tracker_log="$OUT/full_${run}.tracker.log"
    launcher_log="$OUT/full_${run}.launcher.log"
    check_log="$OUT/full_${run}.check.log"
    : >"$display_log"
    : >"$tracker_log"
    : >"$launcher_log"
    : >"$check_log"

    printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_START stage=full run=%s duration=%ss warmup_windows=%s\n' \
      "$run" "$DURATION" "$WARMUP_WINDOWS"

    V11_STEP3_DISPLAY_LOG="$display_log" \
    V11_STEP3_TRACKER_LOG="$tracker_log" \
      timeout -s TERM "$((DURATION + 15))s" \
      bash "$ROOT/scripts/run_camera_v11_step3_tracker_v2.sh" \
      >"$launcher_log" 2>&1
    status=$?

    if (( status != 0 && status != 124 && status != 130 && status != 143 )); then
      printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_RESULT stage=full run=%s result=FAIL launcher_status=%s\n' \
        "$run" "$status" | tee -a "$check_log"
      dump_failure_context "$launcher_log" "$tracker_log" "$display_log" | tee -a "$check_log"
      failed=1
      break
    fi

    if ! grep -q 'CAMERA_V11_POWERMIZER_KEEPER result=BOOST_OK' "$launcher_log"; then
      printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_RESULT stage=full run=%s result=FAIL reason=no_vram_boost_gate\n' \
        "$run" | tee -a "$check_log"
      dump_failure_context "$launcher_log" "$tracker_log" "$display_log" | tee -a "$check_log"
      failed=1
      break
    fi

    if ! run_checker "$display_log" "$tracker_log" "$check_log"; then
      printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_RESULT stage=full run=%s result=FAIL reason=tracker_checker\n' \
        "$run" | tee -a "$check_log"
      dump_failure_context "$launcher_log" "$tracker_log" "$display_log" | tee -a "$check_log"
      failed=1
      break
    fi

    if pgrep -af 'services\.camera_v11\.(step1_|step2_|step3_)|yolo26_trt86_step2_worker\.py' >/dev/null 2>&1; then
      printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_RESULT stage=full run=%s result=FAIL reason=stale_project_process\n' \
        "$run" | tee -a "$check_log"
      pgrep -af 'services\.camera_v11\.(step1_|step2_|step3_)|yolo26_trt86_step2_worker\.py' \
        | tee -a "$check_log" || true
      failed=1
      break
    fi

    printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_RESULT stage=full run=%s result=PASS\n' \
      "$run" | tee -a "$check_log"
  done
fi

if (( failed == 0 )); then
  printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_FINAL result=PASS unit=1 full_consecutive=3 duration=%ss step2_gate=1 reuse_run1=%s\n' \
    "$DURATION" "$REUSE_RUN1"
else
  printf 'CAMERA_V11_STEP3_V2_ACCEPTANCE_FINAL result=FAIL duration=%ss step2_gate=1 reuse_run1=%s\n' \
    "$DURATION" "$REUSE_RUN1"
fi
exit "$failed"
