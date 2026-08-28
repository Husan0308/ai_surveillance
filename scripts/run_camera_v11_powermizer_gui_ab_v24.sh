#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
ENGINE="${V11_TRT86_ENGINE:-$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine}"
OUT="${V11_POWERMIZER_GUI_AB_OUT:-/tmp/camera_v11_powermizer_gui_ab_v24}"
PY="$ROOT/.venv-trt86/bin/python"
mkdir -p "$OUT"

fail() {
  printf 'V11_POWERMIZER_GUI_AB result=FAIL reason=%s\n' "$*" >&2
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

telemetry_pid=""
gui_pid=""
cleanup() {
  if [[ -n "$telemetry_pid" ]] && kill -0 "$telemetry_pid" 2>/dev/null; then
    kill -TERM "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  telemetry_pid=""
  if [[ -n "$gui_pid" ]] && kill -0 "$gui_pid" 2>/dev/null; then
    kill -TERM "$gui_pid" 2>/dev/null || true
    wait "$gui_pid" 2>/dev/null || true
  fi
  gui_pid=""
}
trap cleanup EXIT INT TERM

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

printf 'V11_POWERMIZER_GUI_AB_START display=%s engine=%s\n' "$DISPLAY" "$ENGINE"
printf 'V11_POWERMIZER_GUI_AB_CASE label=baseline\n'
run_case baseline

printf 'V11_POWERMIZER_GUI_AB_GUI action=start\n'
DISPLAY="$DISPLAY" nvidia-settings >"$OUT/nvidia-settings-gui.log" 2>&1 &
gui_pid=$!
sleep 2
kill -0 "$gui_pid" 2>/dev/null || fail "nvidia_settings_gui_exited log=$OUT/nvidia-settings-gui.log"

printf 'V11_POWERMIZER_GUI_AB_SET requested=1 gui_pid=%s\n' "$gui_pid"
set +e
DISPLAY="$DISPLAY" nvidia-settings -a '[gpu:0]/GPUPowerMizerMode=1' \
  2>&1 | tee "$OUT/set_mode1.log"
set_status=${PIPESTATUS[0]}
set -e
printf 'V11_POWERMIZER_GUI_AB_SET_RESULT status=%s\n' "$set_status"

sleep 1
printf '%s\n' '=== CLOCK SNAPSHOT WITH GUI HELD ==='
nvidia-smi --query-gpu=pstate,clocks.current.sm,clocks.current.memory,clocks.max.sm,clocks.max.memory,utilization.gpu,utilization.memory --format=csv | tee "$OUT/gui_held_snapshot.log"

printf 'V11_POWERMIZER_GUI_AB_CASE label=gui_held_mode1\n'
run_case gui_held_mode1

printf 'V11_POWERMIZER_GUI_AB_GUI action=stop pid=%s\n' "$gui_pid"
kill -TERM "$gui_pid" 2>/dev/null || true
wait "$gui_pid" 2>/dev/null || true
gui_pid=""
sleep 1

printf '%s\n' '--- key results ---'
grep -hE 'V11_TRT86_ASYNC_WORKER_RESULT|V11_GPU_TELEMETRY_RESULT' \
  "$OUT"/*.log 2>/dev/null || true
printf 'V11_POWERMIZER_GUI_AB_RESULT status=COMPLETE gui_closed=1 clock_lock=0 overclock=0\n'
