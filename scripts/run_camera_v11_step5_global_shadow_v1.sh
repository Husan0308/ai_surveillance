#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
STEP4_LOCK="/tmp/ai_surveillance_camera_v11_step4_reid_quality_v1.lock"
STEP3_LOCK="/tmp/ai_surveillance_camera_v11_step3_tracker_v2.lock"
STEP2_LOCK="/tmp/ai_surveillance_camera_v11_step2_production_v25.lock"
DISPLAY_LOG="${V11_STEP5_DISPLAY_LOG:-/tmp/CAMERA_V11_STEP5_GLOBAL_DISPLAY.log}"
MATCH_LOG="${V11_STEP5_GLOBAL_LOG:-/tmp/CAMERA_V11_STEP5_GLOBAL.log}"
PAIR_TSV="${V11_STEP4_PAIR_TSV:-$ROOT/artifacts/reid/step4_pair_scores_v1.tsv}"
MATCH_TSV="${V11_STEP4_MATCH_TSV:-$ROOT/artifacts/reid/step4_same_room_matches_v1.tsv}"
GLOBAL_TSV="${V11_STEP5_GLOBAL_TSV:-$ROOT/artifacts/reid/step5_global_shadow_v1.tsv}"
DETECTOR_ENGINE="$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine"
REID_ENGINE="$ROOT/artifacts/reid/resnet50_market1501_aicity156_b1-8_fp32_trt86.engine"
TRT_PY="$ROOT/.venv-trt86/bin/python"
REID_WORKER="$ROOT/scripts/reid_trt86_worker_v11.py"
REID_MANIFEST="$ROOT/artifacts/reid/python_trt86_site/.v11_runtime_paths.json"
PRIME_SCRIPT="$ROOT/scripts/benchmark_yolo26_trt86_step2_worker_v22.py"
POWER_LOG="${V11_STEP5_POWER_LOG:-/tmp/CAMERA_V11_STEP5_POWER_V1.log}"

fail() {
  printf 'CAMERA_V11_STEP5_GLOBAL_SHADOW_PREFLIGHT result=FAIL reason=%s\n' "$*" >&2
  exit 1
}

command -v flock >/dev/null 2>&1 || fail "flock_missing"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia_smi_missing"
exec 9>"$STEP4_LOCK"
flock -n 9 || fail "another_step4_or_step5_launcher_holds=$STEP4_LOCK"
exec 8>"$STEP3_LOCK"
flock -n 8 || fail "step3_or_step4_holds=$STEP3_LOCK"
exec 7>"$STEP2_LOCK"
flock -n 7 || fail "step2_or_step3_holds=$STEP2_LOCK"
[[ -n "${DISPLAY:-}" ]] || fail "DISPLAY_empty"
[[ -s "$DETECTOR_ENGINE" ]] || fail "detector_fp32_engine_missing"
[[ -s "$REID_ENGINE" ]] || fail "reid_fp32_engine_missing"
[[ -x "$TRT_PY" ]] || fail "trt86_python_missing"
[[ -f "$REID_WORKER" ]] || fail "reid_worker_missing"
[[ -f "$REID_MANIFEST" ]] || fail "reid_runtime_manifest_missing"
[[ -f "$PRIME_SCRIPT" ]] || fail "trt_prime_script_missing"

"$ROOT/.venv/bin/python" "$ROOT/scripts/check_camera_v11_frozen_step123_guard.py" \
  || fail "frozen_step123_guard"

CONFLICT_PATTERN='services\.camera_v11\.(step1_cam02_lowlat_v7|step2_production_fp32(_v[0-9]+)?|step3_tracking_v[0-9]+|step4_reid_(quality|gallery|pair|same_room)_runtime_v[0-9]+|step5_global_shadow_runtime_v[0-9]+)|yolo26_trt86_step2_worker\.py|reid_trt86_worker_v11\.py'
conflicts="$(pgrep -af "$CONFLICT_PATTERN" || true)"
[[ -z "$conflicts" ]] || fail $'conflicting_camera_or_trt_process:\n'"$conflicts"
if pgrep -x nvidia-settings >/dev/null 2>&1; then
  fail "nvidia_settings_gui_running_close_it_before_step5"
fi

display_pid=""
match_pid=""
prime_pid=""
cleaned=0
cleanup() {
  (( cleaned == 1 )) && return 0
  cleaned=1
  if [[ -n "$prime_pid" ]] && kill -0 "$prime_pid" 2>/dev/null; then
    kill -TERM "$prime_pid" 2>/dev/null || true
    wait "$prime_pid" 2>/dev/null || true
  fi
  for pid in "$match_pid" "$display_pid"; do
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "$match_pid" "$display_pid"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
}
on_signal() {
  cleanup
  exit 130
}
trap cleanup EXIT
trap on_signal INT TERM

: >"$DISPLAY_LOG"
: >"$MATCH_LOG"
: >"$POWER_LOG"

# Keep the known-good Step4 V3 natural floating-clock hard gate. This catches
# the GTX 1050 Ti / driver memory-clock regression before any camera is opened.
printf 'CAMERA_V11_STEP5_GLOBAL_SHADOW_POWER mode=natural-floating powermizer_write=0 nvidia_settings_keeper=0 start=1\n'
"$TRT_PY" "$PRIME_SCRIPT" --engine "$DETECTOR_ENGINE" --warmup 30 --iterations 120 \
  >>"$POWER_LOG" 2>&1 &
prime_pid=$!
peak_mem=0
peak_sm=0
peak_util=0
best_pstate=99
samples=0
while kill -0 "$prime_pid" 2>/dev/null; do
  sample="$(nvidia-smi --query-gpu=pstate,clocks.current.sm,clocks.current.memory,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
  if [[ -n "$sample" ]]; then
    IFS=',' read -r pstate sm mem util <<<"$sample"
    pstate="$(printf '%s' "$pstate" | tr -d '[:space:]P')"
    sm="$(printf '%s' "$sm" | tr -d '[:space:]')"
    mem="$(printf '%s' "$mem" | tr -d '[:space:]')"
    util="$(printf '%s' "$util" | tr -d '[:space:]')"
    [[ "$pstate" =~ ^[0-9]+$ ]] && (( pstate < best_pstate )) && best_pstate=$pstate || true
    [[ "$sm" =~ ^[0-9]+$ ]] && (( sm > peak_sm )) && peak_sm=$sm || true
    [[ "$mem" =~ ^[0-9]+$ ]] && (( mem > peak_mem )) && peak_mem=$mem || true
    [[ "$util" =~ ^[0-9]+$ ]] && (( util > peak_util )) && peak_util=$util || true
    samples=$((samples + 1))
  fi
  sleep 0.02
done
wait "$prime_pid" || fail "natural_prime_failed"
prime_pid=""
(( best_pstate == 99 )) && best_pstate=-1
minimum_mhz="${V11_STEP5_MIN_ACTIVE_MEMORY_MHZ:-3000}"
minimum_util="${V11_STEP5_MIN_PRIME_GPU_UTIL:-50}"
printf 'CAMERA_V11_STEP5_GLOBAL_SHADOW_NATURAL_PRIME peak_memory_mhz=%s peak_sm_mhz=%s peak_gpu_util=%s best_pstate=P%s samples=%s minimum_memory_mhz=%s minimum_gpu_util=%s\n' \
  "$peak_mem" "$peak_sm" "$peak_util" "$best_pstate" "$samples" "$minimum_mhz" "$minimum_util"
[[ "$minimum_mhz" =~ ^[0-9]+$ ]] || fail "invalid_minimum_memory_mhz"
[[ "$minimum_util" =~ ^[0-9]+$ ]] || fail "invalid_minimum_gpu_util"
(( peak_util >= minimum_util )) || fail "natural_prime_not_busy_peak_gpu_util_${peak_util}_min_${minimum_util}"
(( peak_mem >= minimum_mhz )) || fail "natural_prime_memory_clock_${peak_mem}_min_${minimum_mhz}"
printf 'CAMERA_V11_STEP5_GLOBAL_SHADOW_NATURAL_PRIME result=PASS active_memory_mhz=%s active_sm_mhz=%s peak_gpu_util=%s\n' \
  "$peak_mem" "$peak_sm" "$peak_util"
printf 'CAMERA_V11_STEP5_GLOBAL_SHADOW_PREFLIGHT result=PASS frozen_sha=%s reid_engine=%s precision=fp32 pair_tsv=%s match_tsv=%s global_tsv=%s power_policy=natural-floating active_prime_gate=hard\n' \
  "d2c9e62f9ed2b5f80dc9a4d496e0fda94afddc51" "$REID_ENGINE" "$PAIR_TSV" "$MATCH_TSV" "$GLOBAL_TSV"

bash "$ROOT/scripts/run_camera_v11_step1_v7.sh" >"$DISPLAY_LOG" 2>&1 &
display_pid=$!
sleep "${V11_STEP5_DISPLAY_WARMUP_SEC:-8}"
kill -0 "$display_pid" 2>/dev/null || fail "display_exited_during_warmup"

export V11_STEP2_MODE=full
export V11_STEP2_HZ="${V11_STEP5_HZ:-2.0}"
export V11_STEP2_CONF="${V11_STEP5_DETECTOR_CONF:-0.18}"
export V11_STEP2_ENGINE="$DETECTOR_ENGINE"
export V11_STEP2_TRT86_PYTHON="$TRT_PY"
export V11_STEP2_TRT86_WORKER="$ROOT/scripts/yolo26_trt86_step2_worker.py"
export V11_STEP4_REID_ENGINE="$REID_ENGINE"
export V11_STEP4_REID_PYTHON="$TRT_PY"
export V11_STEP4_REID_WORKER="$REID_WORKER"
export V11_STEP4_PAIR_TSV="$PAIR_TSV"
export V11_STEP4_MATCH_TSV="$MATCH_TSV"
export V11_STEP5_GLOBAL_TSV="$GLOBAL_TSV"
export V11_STEP4_MATCH_MIN_ROBUST_SCORE=off
export V11_STEP4_MATCH_MIN_ROW_MARGIN=off
export V11_STEP4_MATCH_MIN_COLUMN_MARGIN=off
export V11_STEP4_MATCH_RUN_SEC="${V11_STEP5_RUN_SEC:-60}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export V11_STEP4_MATCH_CPU="${V11_STEP5_MATCH_CPU:-11}"

taskset -c "${V11_STEP5_RUNTIME_CPUS:-0-10}" \
  "$ROOT/.venv/bin/python" -u -m services.camera_v11.step5_global_shadow_runtime_v1 \
  >"$MATCH_LOG" 2>&1 &
match_pid=$!

live_ready=0
for _ in $(seq 1 "${V11_STEP5_READY_ATTEMPTS:-900}"); do
  if grep -q '^CAMERA_V11_STEP5_GLOBAL_SHADOW_V1 ' "$MATCH_LOG" && \
     grep -q '^CAMERA_V11_STEP4_REID_SAME_ROOM_MATCHER_V1 ' "$MATCH_LOG" && \
     grep -q '^CAMERA_V11_STEP4_REID_PAIR_SCORER_V1 ' "$MATCH_LOG" && \
     grep -q '^CAMERA_V11_STEP4_REID_GALLERY_V1 ' "$MATCH_LOG" && \
     grep -q '^CAMERA_V11_STEP3_V2_TRACKER ' "$MATCH_LOG"; then
    live_ready=1
    break
  fi
  kill -0 "$match_pid" 2>/dev/null || break
  sleep 0.1
done
(( live_ready == 1 )) || fail "step5_runtime_not_ready"

runtime_sample="$(nvidia-smi --query-gpu=pstate,clocks.current.sm,clocks.current.memory,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
printf 'CAMERA_V11_STEP5_GLOBAL_SHADOW_RUNTIME_CLOCK sample=%q diagnostic_only=1\n' "$runtime_sample"
printf 'CAMERA_V11_STEP5_GLOBAL_SHADOW_RUNNING display_pid=%s runtime_pid=%s camera_queue=0 reid_sync=0 matcher_async=1 state_async=1 production_global_id=0 room_id=0 face=0 handoff=0\n' \
  "$display_pid" "$match_pid"
while kill -0 "$display_pid" 2>/dev/null && kill -0 "$match_pid" 2>/dev/null; do
  sleep 1
done

kill -0 "$display_pid" 2>/dev/null || wait "$display_pid"
wait "$match_pid"
