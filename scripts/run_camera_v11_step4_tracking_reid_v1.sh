#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
FROZEN_STEP3_SHA="d2c9e62f9ed2b5f80dc9a4d496e0fda94afddc51"
STEP4_LOCK="/tmp/ai_surveillance_camera_v11_step4_tracking_reid_v1.lock"
STEP3_LOCK="/tmp/ai_surveillance_camera_v11_step3_tracker_v2.lock"
STEP2_LOCK="/tmp/ai_surveillance_camera_v11_step2_production_v25.lock"
DISPLAY_LOG="${V11_STEP4_DISPLAY_LOG:-/tmp/CAMERA_V11_STEP4_DISPLAY.log}"
STEP4_LOG="${V11_STEP4_LOG:-/tmp/CAMERA_V11_STEP4_REID.log}"
GPU_LOG="${V11_STEP4_GPU_LOG:-/tmp/CAMERA_V11_STEP4_GPU.csv}"
YOLO_ENGINE="$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine"
REID_ENGINE="$ROOT/artifacts/reid/resnet50_market1501_aicity156_b1-8_fp32_trt86.engine"
TRT_PY="$ROOT/.venv-trt86/bin/python"
PRIME_SCRIPT="$ROOT/scripts/benchmark_yolo26_trt86_step2_worker_v22.py"

fail() {
  printf 'CAMERA_V11_STEP4_PREFLIGHT result=FAIL reason=%s\n' "$*" >&2
  exit 1
}

command -v flock >/dev/null 2>&1 || fail "flock_missing"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia_smi_missing"
exec 9>"$STEP4_LOCK"
flock -n 9 || fail "another_step4_launcher_holds=$STEP4_LOCK"
exec 8>"$STEP3_LOCK"
flock -n 8 || fail "step3_or_step4_holds=$STEP3_LOCK"
exec 7>"$STEP2_LOCK"
flock -n 7 || fail "step2_or_other_step_holds=$STEP2_LOCK"
[[ -n "${DISPLAY:-}" ]] || fail "DISPLAY_empty"
[[ -s "$YOLO_ENGINE" ]] || fail "yolo_fp32_engine_missing"
[[ -x "$TRT_PY" ]] || fail "trt86_python_missing"
[[ -f "$PRIME_SCRIPT" ]] || fail "trt_prime_script_missing"

if [[ ! -s "$REID_ENGINE" ]]; then
  printf 'CAMERA_V11_STEP4_PREFLIGHT reid_engine_missing=1 action=prepare\n'
  bash "$ROOT/scripts/prepare_camera_v11_step4_reid_v1.sh" || fail "reid_engine_prepare_failed"
fi
[[ -s "$REID_ENGINE" ]] || fail "reid_engine_missing_after_prepare"
bash "$ROOT/scripts/ensure_camera_v11_trt86_runtime_v1.sh" || fail "trt86_runtime_invalid"

git cat-file -e "${FROZEN_STEP3_SHA}^{commit}" 2>/dev/null || fail "frozen_step3_sha_missing_locally"
git merge-base --is-ancestor "$FROZEN_STEP3_SHA" HEAD || fail "branch_not_based_on_frozen_step3"
git diff --quiet "$FROZEN_STEP3_SHA" -- \
  services/camera_v11/step1_cam02_lowlat_v7.py \
  services/camera_v11/step1_independent_egl_v4.py \
  services/camera_v11/step2_production_fp32.py \
  services/camera_v11/step2_production_fp32_v12.py \
  services/camera_v11/step2_production_fp32_v13.py \
  services/camera_v11/step2_production_fp32_v18.py \
  services/camera_v11/step2_trt86.py \
  services/camera_v11/step3_tracker_v2.py \
  services/camera_v11/step3_tracking_v2.py \
  scripts/run_camera_v11_step3_tracker_v2.sh \
  scripts/check_camera_v11_step3_tracker_v2_log.py \
  || fail "frozen_step3_files_changed"

CONFLICT_PATTERN='services\.camera_v11\.(step1_cam02_lowlat_v7|step2_production_fp32(_v[0-9]+)?|step3_tracking_v[0-9]+|step4_tracking_reid_v[0-9]+)|yolo26_trt86_step2_worker\.py|reid_trt86_worker_v11\.py'
conflicts="$(pgrep -af "$CONFLICT_PATTERN" || true)"
[[ -z "$conflicts" ]] || fail $'conflicting_camera_or_trt_process:\n'"$conflicts"

# shellcheck source=/dev/null
source "$ROOT/scripts/camera_v11_powermizer_keeper_v25.sh"
display_pid=""
step4_pid=""
telemetry_pid=""
cleaned=0
cleanup() {
  (( cleaned == 1 )) && return 0
  cleaned=1
  for pid in "$step4_pid" "$display_pid" "$telemetry_pid"; do
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "$step4_pid" "$display_pid" "$telemetry_pid"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
  telemetry_pid=""
  v11_powermizer_stop || true
}
on_signal() {
  cleanup
  exit 130
}
trap cleanup EXIT
trap on_signal INT TERM

: >"$DISPLAY_LOG"
: >"$STEP4_LOG"
: >"$GPU_LOG"

printf 'CAMERA_V11_STEP4_POWER_PRIME phase=baseline start=1\n'
"$TRT_PY" "$PRIME_SCRIPT" --engine "$YOLO_ENGINE" --warmup 30 --iterations 100 \
  >>"${V11_POWERMIZER_KEEPER_LOG:-/tmp/CAMERA_V11_POWERMIZER_KEEPER.log}" 2>&1 \
  || fail "power_prime_baseline_failed"
printf 'CAMERA_V11_STEP4_POWER_PRIME phase=baseline result=PASS\n'

v11_powermizer_start || fail "powermizer_keeper_start"
sleep 1
printf 'CAMERA_V11_STEP4_POWER_PRIME phase=gui_held start=1\n'
"$TRT_PY" "$PRIME_SCRIPT" --engine "$YOLO_ENGINE" --warmup 30 --iterations 100 \
  >>"$V11_POWERMIZER_KEEPER_LOG" 2>&1 \
  || fail "power_prime_gui_held_failed"
clock="$(v11_powermizer_mem_clock_mhz || true)"
minimum_mhz="${V11_POWERMIZER_MIN_MEMORY_MHZ:-3000}"
if [[ ! "$clock" =~ ^[0-9]+$ ]] || (( clock < minimum_mhz )); then
  fail "vram_startup_prime_gate memory_mhz=${clock:-unknown} minimum_mhz=$minimum_mhz"
fi
printf 'CAMERA_V11_POWERMIZER_KEEPER result=BOOST_OK memory_mhz=%s minimum_mhz=%s pid=%s source=v24-continuous-prime\n' \
  "$clock" "$minimum_mhz" "$V11_POWERMIZER_KEEPER_PID"

printf 'CAMERA_V11_STEP4_PREFLIGHT result=PASS frozen_step3_sha=%s runtime=trt86 reid_engine=1 display_log=%s step4_log=%s gpu_log=%s\n' \
  "$FROZEN_STEP3_SHA" "$DISPLAY_LOG" "$STEP4_LOG" "$GPU_LOG"

bash "$ROOT/scripts/run_camera_v11_step1_v7.sh" >"$DISPLAY_LOG" 2>&1 &
display_pid=$!
sleep "${V11_STEP4_DISPLAY_WARMUP_SEC:-8}"
kill -0 "$display_pid" 2>/dev/null || fail "display_exited_during_warmup"

export V11_STEP2_MODE=full
export V11_STEP2_HZ="${V11_STEP4_HZ:-2.0}"
export V11_STEP2_CONF="${V11_STEP4_CONF:-0.18}"
export V11_STEP2_ENGINE="$YOLO_ENGINE"
export V11_STEP2_TRT86_PYTHON="$TRT_PY"
export V11_STEP2_TRT86_WORKER="$ROOT/scripts/yolo26_trt86_step2_worker.py"
export V11_STEP4_REID_ENGINE="$REID_ENGINE"
export V11_STEP4_REID_MAX_BATCH="${V11_STEP4_REID_MAX_BATCH:-2}"
export V11_STEP4_REID_MAX_WAIT_MS="${V11_STEP4_REID_MAX_WAIT_MS:-3}"
export V11_STEP4_REID_MAX_PENDING="${V11_STEP4_REID_MAX_PENDING:-12}"
export V11_STEP4_REID_MAX_AGE_MS="${V11_STEP4_REID_MAX_AGE_MS:-300}"
export V11_STEP4_REID_REFRESH_SEC="${V11_STEP4_REID_REFRESH_SEC:-1.0}"
export V11_STEP4_REID_MAX_PER_UPDATE="${V11_STEP4_REID_MAX_PER_UPDATE:-2}"

nvidia-smi \
  --query-gpu=timestamp,pstate,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
  --format=csv,noheader,nounits -lms 500 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!

"$ROOT/.venv/bin/python" -u -m services.camera_v11.step4_tracking_reid_v2 >"$STEP4_LOG" 2>&1 &
step4_pid=$!

live_ready=0
for _ in $(seq 1 "${V11_STEP4_LIVE_READY_ATTEMPTS:-900}"); do
  if grep -q '^CAMERA_V11_STEP4_REID ' "$STEP4_LOG" && \
     grep -q '^CAMERA_V11_STEP3_V2_TRACKER ' "$STEP4_LOG"; then
    live_ready=1
    break
  fi
  kill -0 "$step4_pid" 2>/dev/null || break
  sleep 0.1
done
(( live_ready == 1 )) || fail "live_step4_not_ready"

printf 'CAMERA_V11_STEP4_RUNNING display_pid=%s step4_pid=%s telemetry_pid=%s keeper_pid=%s shadow_merge=0\n' \
  "$display_pid" "$step4_pid" "$telemetry_pid" "$V11_POWERMIZER_KEEPER_PID"

while kill -0 "$display_pid" 2>/dev/null && kill -0 "$step4_pid" 2>/dev/null; do
  sleep 1
done

kill -0 "$display_pid" 2>/dev/null || wait "$display_pid"
wait "$step4_pid"
status=$?

if [[ -n "$telemetry_pid" ]] && kill -0 "$telemetry_pid" 2>/dev/null; then
  kill -TERM "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
fi
telemetry_pid=""
"$ROOT/.venv/bin/python" "$ROOT/scripts/summarize_camera_v11_gpu_telemetry_v22.py" \
  "$GPU_LOG" --label step4-live || true
exit "$status"
