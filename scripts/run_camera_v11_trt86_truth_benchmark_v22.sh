#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
ENGINE="${V11_TRT86_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
OUT="${V11_TRT86_TRUTH_OUT:-/tmp/camera_v11_trt86_truth_v22}"
PY="$ROOT/.venv-trt86/bin/python"
mkdir -p "$OUT"

fail() {
  printf 'V11_TRT86_TRUTH result=FAIL reason=%s\n' "$*" >&2
  exit 1
}

[[ -x "$PY" ]] || fail "missing_trt86_python=$PY"
[[ -s "$ENGINE" ]] || fail "missing_engine=$ENGINE"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia_smi_missing"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum_missing"

# This diagnostic must start from an idle project state. Do not kill anything
# automatically: showing the exact conflict is safer than terminating unrelated work.
CONFLICT_PATTERN='services\\.camera_v11\\.(step1_|step2_|step3_)|yolo26_trt86_step2_worker\\.py|probe_camera_v11_trt86_b1_precision\\.py|build_yolo26s_b1_.*trt86\\.py'
conflicts="$(pgrep -af "$CONFLICT_PATTERN" || true)"
[[ -z "$conflicts" ]] || fail $'project_gpu_process_alive:\n'"$conflicts"

engine_sha="$(sha256sum "$ENGINE" | awk '{print $1}')"
printf 'V11_TRT86_TRUTH_START engine=%s sha256=%s out=%s\n' "$ENGINE" "$engine_sha" "$OUT"
{
  printf '%s\n' '=== NVIDIA SNAPSHOT ==='
  # Keep the core snapshot authoritative. PCIe link fields are diagnostic-only and
  # vary across nvidia-smi releases, so failure to expose them must not abort the benchmark.
  nvidia-smi --query-gpu=name,driver_version,pstate,clocks.current.sm,clocks.current.memory,clocks.max.sm,clocks.max.memory,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,persistence_mode --format=csv
  printf '%s\n' '=== PCIE SNAPSHOT (OPTIONAL) ==='
  if ! nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv; then
    printf '%s\n' 'V11_TRT86_PCIE_SNAPSHOT status=SKIP reason=query_not_supported'
  fi
  printf '%s\n' '=== TRT PYTHON ==='
  "$PY" - <<'PY'
import sys
import tensorrt as trt
print(f"python={sys.version.split()[0]} tensorrt={trt.__version__}")
PY
  printf 'engine_sha256=%s\n' "$engine_sha"
} | tee "$OUT/environment.log"

telemetry_pid=""
start_telemetry() {
  local log="$1"
  : >"$log"
  nvidia-smi \
    --query-gpu=timestamp,pstate,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
    --format=csv,noheader,nounits -lms 200 >"$log" 2>&1 &
  telemetry_pid=$!
}
stop_telemetry() {
  if [[ -n "$telemetry_pid" ]] && kill -0 "$telemetry_pid" 2>/dev/null; then
    kill -TERM "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  telemetry_pid=""
}
trap stop_telemetry EXIT INT TERM

run_with_telemetry() {
  local label="$1"
  shift
  local telemetry="$OUT/$label.telemetry.csv"
  local log="$OUT/$label.log"
  start_telemetry "$telemetry"
  "$@" 2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  stop_telemetry
  "$ROOT/.venv/bin/python" "$ROOT/scripts/summarize_camera_v11_gpu_telemetry_v22.py" \
    "$telemetry" --label "$label" | tee "$OUT/$label.telemetry.summary.log"
  return "$status"
}

printf '%s\n' '=== HARNESS: execute_v2 + cudaDeviceSynchronize ==='
run_with_telemetry sync_harness \
  "$PY" "$ROOT/scripts/probe_camera_v11_trt86_b1_precision.py" \
  --engine "$ENGINE" --warmup 30 --iterations 100

printf '%s\n' '=== HARNESS: production execute_async_v2 worker ==='
run_with_telemetry async_worker \
  "$PY" "$ROOT/scripts/benchmark_yolo26_trt86_step2_worker_v22.py" \
  --engine "$ENGINE" --warmup 30 --iterations 100

TRTEXEC="${V11_TRTEXEC:-}"
if [[ -z "$TRTEXEC" ]]; then
  TRTEXEC="$(command -v trtexec || true)"
fi
if [[ -z "$TRTEXEC" && -x /usr/src/tensorrt/bin/trtexec ]]; then
  TRTEXEC=/usr/src/tensorrt/bin/trtexec
fi
if [[ -n "$TRTEXEC" && -x "$TRTEXEC" ]]; then
  printf '%s\n' '=== HARNESS: TensorRT trtexec ==='
  run_with_telemetry trtexec \
    "$TRTEXEC" --loadEngine="$ENGINE" --warmUp=3000 --duration=10 --iterations=100 --streams=1 --useSpinWait
else
  printf 'V11_TRT86_TRTEXEC status=SKIP reason=not_found\n' | tee "$OUT/trtexec.log"
fi

printf 'V11_TRT86_TRUTH_RESULT status=COMPLETE engine_sha256=%s out=%s clocks_modified=0 clock_prime=0\n' \
  "$engine_sha" "$OUT"
printf '%s\n' '--- key results ---'
grep -hE 'V11_TRT86_B1_PRECISION_RESULT|V11_TRT86_ASYNC_WORKER_RESULT|V11_GPU_TELEMETRY_RESULT|Throughput:|Latency:|GPU Compute Time:|V11_TRT86_TRTEXEC' \
  "$OUT"/*.log 2>/dev/null || true
