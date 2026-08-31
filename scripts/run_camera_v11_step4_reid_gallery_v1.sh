#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
STEP4_LOCK="/tmp/ai_surveillance_camera_v11_step4_reid_quality_v1.lock"
STEP3_LOCK="/tmp/ai_surveillance_camera_v11_step3_tracker_v2.lock"
STEP2_LOCK="/tmp/ai_surveillance_camera_v11_step2_production_v25.lock"
DISPLAY_LOG="${V11_STEP4_GALLERY_DISPLAY_LOG:-/tmp/CAMERA_V11_STEP4_GALLERY_DISPLAY.log}"
GALLERY_LOG="${V11_STEP4_GALLERY_LOG:-/tmp/CAMERA_V11_STEP4_REID_GALLERY.log}"
DETECTOR_ENGINE="$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine"
REID_ENGINE="$ROOT/artifacts/reid/resnet50_market1501_aicity156_b1-8_fp32_trt86.engine"
TRT_PY="$ROOT/.venv-trt86/bin/python"
REID_WORKER="$ROOT/scripts/reid_trt86_worker_v11.py"
REID_MANIFEST="$ROOT/artifacts/reid/python_trt86_site/.v11_runtime_paths.json"
PRIME_SCRIPT="$ROOT/scripts/benchmark_yolo26_trt86_step2_worker_v22.py"

fail() {
  printf 'CAMERA_V11_STEP4_REID_GALLERY_PREFLIGHT result=FAIL reason=%s\n' "$*" >&2
  exit 1
}

command -v flock >/dev/null 2>&1 || fail "flock_missing"
exec 9>"$STEP4_LOCK"
flock -n 9 || fail "another_step4_launcher_holds=$STEP4_LOCK"
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

CONFLICT_PATTERN='services\.camera_v11\.(step1_cam02_lowlat_v7|step2_production_fp32(_v[0-9]+)?|step3_tracking_v[0-9]+|step4_reid_(quality|gallery)_runtime_v[0-9]+)|yolo26_trt86_step2_worker\.py|reid_trt86_worker_v11\.py'
conflicts="$(pgrep -af "$CONFLICT_PATTERN" || true)"
[[ -z "$conflicts" ]] || fail $'conflicting_camera_or_trt_process:\n'"$conflicts"

# shellcheck source=/dev/null
source "$ROOT/scripts/camera_v11_powermizer_keeper_v25.sh"
display_pid=""
gallery_pid=""
cleaned=0
cleanup() {
  (( cleaned == 1 )) && return 0
  cleaned=1
  for pid in "$gallery_pid" "$display_pid"; do
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "$gallery_pid" "$display_pid"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
  v11_powermizer_stop || true
}
on_signal() {
  cleanup
  exit 130
}
trap cleanup EXIT
trap on_signal INT TERM

: >"$DISPLAY_LOG"
: >"$GALLERY_LOG"

printf 'CAMERA_V11_STEP4_REID_GALLERY_POWER_PRIME phase=baseline start=1\n'
"$TRT_PY" "$PRIME_SCRIPT" --engine "$DETECTOR_ENGINE" --warmup 30 --iterations 100 \
  >>"${V11_POWERMIZER_KEEPER_LOG:-/tmp/CAMERA_V11_POWERMIZER_KEEPER.log}" 2>&1 \
  || fail "power_prime_baseline_failed"
v11_powermizer_start || fail "powermizer_keeper_start"
sleep 1
printf 'CAMERA_V11_STEP4_REID_GALLERY_POWER_PRIME phase=gui_held start=1\n'
"$TRT_PY" "$PRIME_SCRIPT" --engine "$DETECTOR_ENGINE" --warmup 30 --iterations 100 \
  >>"$V11_POWERMIZER_KEEPER_LOG" 2>&1 \
  || fail "power_prime_gui_held_failed"
clock="$(v11_powermizer_mem_clock_mhz || true)"
minimum_mhz="${V11_POWERMIZER_MIN_MEMORY_MHZ:-3000}"
if [[ ! "$clock" =~ ^[0-9]+$ ]] || (( clock < minimum_mhz )); then
  fail "vram_startup_prime_gate memory_mhz=${clock:-unknown} minimum_mhz=$minimum_mhz"
fi
printf 'CAMERA_V11_POWERMIZER_KEEPER result=BOOST_OK memory_mhz=%s minimum_mhz=%s pid=%s source=v24-continuous-prime\n' \
  "$clock" "$minimum_mhz" "$V11_POWERMIZER_KEEPER_PID"
printf 'CAMERA_V11_STEP4_REID_GALLERY_PREFLIGHT result=PASS frozen_sha=%s reid_engine=%s precision=fp32 display_log=%s gallery_log=%s\n' \
  "d2c9e62f9ed2b5f80dc9a4d496e0fda94afddc51" "$REID_ENGINE" "$DISPLAY_LOG" "$GALLERY_LOG"

bash "$ROOT/scripts/run_camera_v11_step1_v7.sh" >"$DISPLAY_LOG" 2>&1 &
display_pid=$!
sleep "${V11_STEP4_GALLERY_DISPLAY_WARMUP_SEC:-8}"
kill -0 "$display_pid" 2>/dev/null || fail "display_exited_during_warmup"

export V11_STEP2_MODE=full
export V11_STEP2_HZ="${V11_STEP4_GALLERY_HZ:-2.0}"
export V11_STEP2_CONF="${V11_STEP4_GALLERY_DETECTOR_CONF:-0.18}"
export V11_STEP2_ENGINE="$DETECTOR_ENGINE"
export V11_STEP2_TRT86_PYTHON="$TRT_PY"
export V11_STEP2_TRT86_WORKER="$ROOT/scripts/yolo26_trt86_step2_worker.py"
export V11_STEP4_REID_ENGINE="$REID_ENGINE"
export V11_STEP4_REID_PYTHON="$TRT_PY"
export V11_STEP4_REID_WORKER="$REID_WORKER"

"$ROOT/.venv/bin/python" -u -m services.camera_v11.step4_reid_gallery_runtime_v1 \
  >"$GALLERY_LOG" 2>&1 &
gallery_pid=$!

live_ready=0
for _ in $(seq 1 "${V11_STEP4_GALLERY_READY_ATTEMPTS:-900}"); do
  if grep -q '^CAMERA_V11_STEP4_REID_GALLERY_V1 ' "$GALLERY_LOG" && \
     grep -q '^CAMERA_V11_STEP3_V2_TRACKER ' "$GALLERY_LOG"; then
    live_ready=1
    break
  fi
  kill -0 "$gallery_pid" 2>/dev/null || break
  sleep 0.1
done
(( live_ready == 1 )) || fail "gallery_runtime_not_ready"

printf 'CAMERA_V11_STEP4_REID_GALLERY_RUNNING display_pid=%s gallery_pid=%s keeper_pid=%s camera_queue=0 reid_sync=0 reid_workers=1 pending=bounded-keyed-latest\n' \
  "$display_pid" "$gallery_pid" "$V11_POWERMIZER_KEEPER_PID"
while kill -0 "$display_pid" 2>/dev/null && kill -0 "$gallery_pid" 2>/dev/null; do
  sleep 1
done

kill -0 "$display_pid" 2>/dev/null || wait "$display_pid"
wait "$gallery_pid"
