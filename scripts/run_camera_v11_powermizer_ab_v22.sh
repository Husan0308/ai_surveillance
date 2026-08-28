#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
ENGINE="${V11_TRT86_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
OUT="${V11_POWERMIZER_AB_OUT:-/tmp/camera_v11_powermizer_ab_v22}"
PY="$ROOT/.venv-trt86/bin/python"
mkdir -p "$OUT"

fail() {
  printf 'V11_POWERMIZER_AB result=FAIL reason=%s\n' "$*" >&2
  exit 1
}

[[ -n "${DISPLAY:-}" ]] || fail "DISPLAY_empty"
[[ -x "$PY" ]] || fail "missing_trt86_python=$PY"
[[ -s "$ENGINE" ]] || fail "missing_engine=$ENGINE"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia_smi_missing"
command -v nvidia-settings >/dev/null 2>&1 || fail "nvidia_settings_missing"

CONFLICT_PATTERN='services\.camera_v11\.(step1_|step2_|step3_)|yolo26_trt86_step2_worker\.py|probe_camera_v11_trt86_b1_precision\.py|build_yolo26s_b1_.*trt86\.py'
conflicts="$(pgrep -af "$CONFLICT_PATTERN" || true)"
[[ -z "$conflicts" ]] || fail $'project_gpu_process_alive:\n'"$conflicts"

query_mode() {
  nvidia-settings -q '[gpu:0]/GPUPowerMizerMode' -t 2>/dev/null | tail -n 1 | tr -d '[:space:]'
}

original_mode="$(query_mode || true)"
[[ "$original_mode" =~ ^[012]$ ]] || fail "cannot_query_GPUPowerMizerMode value=${original_mode:-none}"

restore_mode() {
  if [[ -n "${original_mode:-}" ]]; then
    nvidia-settings -a "[gpu:0]/GPUPowerMizerMode=$original_mode" >/dev/null 2>&1 || true
  fi
}
trap restore_mode EXIT INT TERM

printf 'V11_POWERMIZER_AB_START original_mode=%s target_mode=1 engine=%s\n' "$original_mode" "$ENGINE"
printf '%s\n' '=== GPU PERF MODES ==='
nvidia-settings -q '[gpu:0]/GPUPerfModes' -t 2>&1 | tee "$OUT/gpu_perf_modes.log" || true
printf '%s\n' '=== BEFORE ==='
nvidia-smi --query-gpu=pstate,clocks.current.sm,clocks.current.memory,clocks.max.sm,clocks.max.memory,utilization.gpu,utilization.memory --format=csv | tee "$OUT/before.log"

telemetry_pid=""
start_telemetry() {
  local file="$1"
  : >"$file"
  nvidia-smi \
    --query-gpu=timestamp,pstate,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
    --format=csv,noheader,nounits -lms 200 >"$file" 2>&1 &
  telemetry_pid=$!
}
stop_telemetry() {
  if [[ -n "$telemetry_pid" ]] && kill -0 "$telemetry_pid" 2>/dev/null; then
    kill -TERM "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  telemetry_pid=""
}
run_case() {
  local label="$1"
  local telemetry="$OUT/$label.telemetry.csv"
  local log="$OUT/$label.log"
  start_telemetry "$telemetry"
  "$PY" "$ROOT/scripts/benchmark_yolo26_trt86_step2_worker_v22.py" \
    --engine "$ENGINE" --warmup 30 --iterations 100 2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  stop_telemetry
  "$ROOT/.venv/bin/python" "$ROOT/scripts/summarize_camera_v11_gpu_telemetry_v22.py" \
    "$telemetry" --label "$label" | tee "$OUT/$label.telemetry.summary.log"
  return "$status"
}

printf 'V11_POWERMIZER_AB_CASE label=original mode=%s\n' "$original_mode"
run_case original

printf 'V11_POWERMIZER_AB_SET requested=1\n'
nvidia-settings -a '[gpu:0]/GPUPowerMizerMode=1' 2>&1 | tee "$OUT/set_max_performance.log"
sleep 1
max_mode="$(query_mode || true)"
printf 'V11_POWERMIZER_AB_MODE requested=1 effective=%s\n' "${max_mode:-none}"
[[ "$max_mode" == "1" ]] || fail "GPUPowerMizerMode_not_effective effective=${max_mode:-none}"

printf '%s\n' '=== AFTER MODE=1 ==='
nvidia-smi --query-gpu=pstate,clocks.current.sm,clocks.current.memory,clocks.max.sm,clocks.max.memory,utilization.gpu,utilization.memory --format=csv | tee "$OUT/after_mode1.log"

printf 'V11_POWERMIZER_AB_CASE label=max_performance mode=1\n'
run_case max_performance

restore_mode
restored="$(query_mode || true)"
printf 'V11_POWERMIZER_AB_RESTORE requested=%s effective=%s\n' "$original_mode" "${restored:-none}"

printf '%s\n' '--- key results ---'
grep -hE 'V11_TRT86_ASYNC_WORKER_RESULT|V11_GPU_TELEMETRY_RESULT' \
  "$OUT"/*.log 2>/dev/null || true
printf 'V11_POWERMIZER_AB_RESULT status=COMPLETE original_mode=%s restored_mode=%s clock_lock=0 overclock=0\n' \
  "$original_mode" "${restored:-none}"
