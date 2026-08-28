#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="${V11_STEP4_REID_BENCH_OUT:-/tmp/camera_v11_step4_reid_bench_v1}"
ENGINE="${V11_STEP4_REID_ENGINE:-$ROOT/artifacts/reid/resnet50_market1501_aicity156_b1-8_fp32_trt86.engine}"
mkdir -p "$OUT"

fail() {
  printf 'V11_STEP4_REID_BENCH_WRAPPER RESULT=FAIL reason=%s\n' "$*" >&2
  exit 1
}

[[ -n "${DISPLAY:-}" ]] || fail "DISPLAY_empty"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia_smi_missing"

CONFLICT_PATTERN='services\.camera_v11\.(step1_|step2_|step3_|step4_)|yolo26_trt86_step2_worker\.py|reid_trt86_worker_v11\.py'
conflicts="$(pgrep -af "$CONFLICT_PATTERN" || true)"
[[ -z "$conflicts" ]] || fail $'project_gpu_process_alive:\n'"$conflicts"

# Validate the exact isolated Python runtime used by reid_trt86_worker_v11.py.
# This runs before PowerMizer/telemetry so a broken Python package cannot waste a
# GPU benchmark run. It repairs only .venv-trt86's NumPy wheel, never /usr/lib.
bash "$ROOT/scripts/ensure_camera_v11_trt86_runtime_v1.sh" \
  || fail "trt86_runtime_prepare_failed"

# shellcheck source=/dev/null
source "$ROOT/scripts/camera_v11_powermizer_keeper_v25.sh"
telemetry_pid=""
cleanup() {
  if [[ -n "$telemetry_pid" ]] && kill -0 "$telemetry_pid" 2>/dev/null; then
    kill -TERM "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  telemetry_pid=""
  v11_powermizer_stop || true
}
trap cleanup EXIT INT TERM

v11_powermizer_start || fail "powermizer_keeper_start"

if [[ ! -s "$ENGINE" ]]; then
  printf 'V11_STEP4_REID_BENCH_WRAPPER ENGINE_MISSING action=prepare path=%s\n' "$ENGINE"
  bash "$ROOT/scripts/prepare_camera_v11_step4_reid_v1.sh" \
    || fail "engine_prepare_failed"
fi
[[ -s "$ENGINE" ]] || fail "engine_missing_after_prepare path=$ENGINE"

telemetry="$OUT/gpu.csv"
: >"$telemetry"
nvidia-smi \
  --query-gpu=timestamp,pstate,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
  --format=csv,noheader,nounits -lms 200 >"$telemetry" 2>&1 &
telemetry_pid=$!

set +e
"$ROOT/.venv/bin/python" "$ROOT/scripts/benchmark_camera_v11_step4_reid_v1.py" \
  --engine "$ENGINE" --warmup "${V11_STEP4_REID_WARMUP:-20}" \
  --iterations "${V11_STEP4_REID_ITERATIONS:-50}" \
  2>&1 | tee "$OUT/benchmark.log"
status=${PIPESTATUS[0]}
set -e

if [[ -n "$telemetry_pid" ]] && kill -0 "$telemetry_pid" 2>/dev/null; then
  kill -TERM "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
fi
telemetry_pid=""

"$ROOT/.venv/bin/python" "$ROOT/scripts/summarize_camera_v11_gpu_telemetry_v22.py" \
  "$telemetry" --label step4-reid | tee "$OUT/telemetry.summary.log" || true

(( status == 0 )) || fail "benchmark_status=$status"
printf 'V11_STEP4_REID_BENCH_WRAPPER RESULT=PASS engine=%s keeper=1 telemetry=%s\n' "$ENGINE" "$telemetry"
