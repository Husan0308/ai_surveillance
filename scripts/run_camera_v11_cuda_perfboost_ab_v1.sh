#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
ENGINE="$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine"
TRT_PY="$ROOT/.venv-trt86/bin/python"
BENCH="$ROOT/scripts/benchmark_yolo26_trt86_step2_worker_v22.py"

fail() {
  printf 'V11_CUDA_PERFBOOST_AB RESULT=FAIL reason=%s\n' "$*" >&2
  exit 1
}

command -v nvidia-smi >/dev/null 2>&1 || fail nvidia_smi_missing
[[ -x "$TRT_PY" ]] || fail trt86_python_missing
[[ -s "$ENGINE" ]] || fail detector_engine_missing
[[ -f "$BENCH" ]] || fail benchmark_missing

if pgrep -x nvidia-settings >/dev/null 2>&1; then
  fail nvidia_settings_running
fi

sample_once() {
  nvidia-smi --query-gpu=pstate,clocks.current.sm,clocks.current.memory,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true
}

run_case() {
  local label="$1"
  local perfboost="$2"
  local log="/tmp/CAMERA_V11_CUDA_PERFBOOST_AB_${label}.log"
  local pid=""
  local peak_mem=0
  local peak_sm=0
  local peak_util=0
  local busy_mem_peak=0
  local busy_mem_min=999999
  local samples=0
  local busy_samples=0
  local best_pstate=99

  printf 'V11_CUDA_PERFBOOST_AB_CASE label=%s CUDA_DISABLE_PERF_BOOST=%s before=%q\n' \
    "$label" "$perfboost" "$(sample_once)"

  if [[ "$perfboost" == "unset" ]]; then
    env -u CUDA_DISABLE_PERF_BOOST \
      "$TRT_PY" "$BENCH" --engine "$ENGINE" --warmup 30 --iterations 140 >"$log" 2>&1 &
  else
    CUDA_DISABLE_PERF_BOOST="$perfboost" \
      "$TRT_PY" "$BENCH" --engine "$ENGINE" --warmup 30 --iterations 140 >"$log" 2>&1 &
  fi
  pid=$!

  while kill -0 "$pid" 2>/dev/null; do
    local sample=""
    sample="$(sample_once)"
    if [[ -n "$sample" ]]; then
      local pstate sm mem util
      IFS=',' read -r pstate sm mem util <<<"$sample"
      pstate="$(printf '%s' "$pstate" | tr -d '[:space:]P')"
      sm="$(printf '%s' "$sm" | tr -d '[:space:]')"
      mem="$(printf '%s' "$mem" | tr -d '[:space:]')"
      util="$(printf '%s' "$util" | tr -d '[:space:]')"
      [[ "$pstate" =~ ^[0-9]+$ ]] && (( pstate < best_pstate )) && best_pstate=$pstate || true
      [[ "$sm" =~ ^[0-9]+$ ]] && (( sm > peak_sm )) && peak_sm=$sm || true
      [[ "$mem" =~ ^[0-9]+$ ]] && (( mem > peak_mem )) && peak_mem=$mem || true
      [[ "$util" =~ ^[0-9]+$ ]] && (( util > peak_util )) && peak_util=$util || true
      if [[ "$util" =~ ^[0-9]+$ ]] && (( util >= 50 )) && [[ "$mem" =~ ^[0-9]+$ ]]; then
        (( mem > busy_mem_peak )) && busy_mem_peak=$mem || true
        (( mem < busy_mem_min )) && busy_mem_min=$mem || true
        busy_samples=$((busy_samples + 1))
      fi
      samples=$((samples + 1))
    fi
    sleep 0.02
  done

  if ! wait "$pid"; then
    tail -n 80 "$log" >&2 || true
    fail "${label}_benchmark_failed"
  fi
  (( best_pstate == 99 )) && best_pstate=-1
  (( busy_mem_min == 999999 )) && busy_mem_min=0
  printf 'V11_CUDA_PERFBOOST_AB_METRICS label=%s CUDA_DISABLE_PERF_BOOST=%s peak_memory_mhz=%s busy_memory_min_mhz=%s busy_memory_peak_mhz=%s peak_sm_mhz=%s peak_gpu_util=%s best_pstate=P%s samples=%s busy_samples=%s after=%q log=%s\n' \
    "$label" "$perfboost" "$peak_mem" "$busy_mem_min" "$busy_mem_peak" "$peak_sm" "$peak_util" "$best_pstate" "$samples" "$busy_samples" "$(sample_once)" "$log"
  grep 'V11_TRT86_ASYNC_WORKER_RESULT' "$log" || true
}

run_case default unset
sleep 2
run_case perfboost_disabled 1

printf 'V11_CUDA_PERFBOOST_AB RESULT=COMPLETE decision=compare_busy_memory_and_inference_latency production_changed=0\n'
