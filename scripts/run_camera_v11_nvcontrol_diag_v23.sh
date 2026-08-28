#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="${V11_NVCONTROL_DIAG_OUT:-/tmp/camera_v11_nvcontrol_diag_v23}"
mkdir -p "$OUT"
LOG="$OUT/diag.log"
: >"$LOG"

emit() {
  printf '%s\n' "$*" | tee -a "$LOG"
}

run_capture() {
  local label="$1"
  shift
  emit "=== $label ==="
  set +e
  "$@" 2>&1 | tee -a "$LOG"
  local status=${PIPESTATUS[0]}
  set -e
  emit "V11_NVCONTROL_CMD label=$label status=$status"
  return 0
}

set -e
emit "V11_NVCONTROL_DIAG_START display=${DISPLAY:-none} wayland_display=${WAYLAND_DISPLAY:-none} xdg_session_type=${XDG_SESSION_TYPE:-none} xdg_session_id=${XDG_SESSION_ID:-none}"

if command -v loginctl >/dev/null 2>&1 && [[ -n "${XDG_SESSION_ID:-}" ]]; then
  run_capture session loginctl show-session "$XDG_SESSION_ID" -p Type -p Name -p Class -p Remote -p Display
fi

run_capture smi_basic nvidia-smi --query-gpu=name,driver_version,pstate,display_mode,display_active,persistence_mode,clocks.current.sm,clocks.current.memory,clocks.max.sm,clocks.max.memory --format=csv
run_capture smi_supported_clocks nvidia-smi --query-supported-clocks=mem,gr --format=csv
run_capture smi_clock_policy nvidia-smi -q -d CLOCK

if command -v nvidia-settings >/dev/null 2>&1; then
  run_capture settings_version nvidia-settings --version
  if [[ -n "${DISPLAY:-}" ]]; then
    run_capture settings_gpus_display nvidia-settings -c "$DISPLAY" -q gpus
    run_capture settings_powermizer_target_display nvidia-settings -c "$DISPLAY" -q '[gpu:0]/GPUPowerMizerMode'
    run_capture settings_powermizer_generic_display nvidia-settings -c "$DISPLAY" -q GPUPowerMizerMode
    run_capture settings_perf_modes_display nvidia-settings -c "$DISPLAY" -q '[gpu:0]/GPUPerfModes'
  fi
  run_capture settings_gpus nvidia-settings -q gpus
  run_capture settings_powermizer_target nvidia-settings -q '[gpu:0]/GPUPowerMizerMode'
  run_capture settings_powermizer_generic nvidia-settings -q GPUPowerMizerMode
else
  emit "V11_NVCONTROL_INFO nvidia_settings=missing"
fi

if command -v nvidia-xconfig >/dev/null 2>&1; then
  run_capture xconfig_gpu_info nvidia-xconfig --query-gpu-info
fi
if command -v xrandr >/dev/null 2>&1 && [[ -n "${DISPLAY:-}" ]]; then
  run_capture xrandr_providers xrandr --listproviders
fi

# nvidia-smi documents -lmc, but support is device-specific and root-only.
# Query help/support only; this diagnostic never changes clocks.
run_capture smi_lock_memory_info nvidia-smi -lmi

emit "V11_NVCONTROL_DIAG_RESULT status=COMPLETE clocks_modified=0 out=$OUT"
emit "--- key results ---"
grep -E 'V11_NVCONTROL_DIAG_START|V11_NVCONTROL_CMD|GPUPowerMizerMode|Provider|Name|Display Active|Display Mode|Memory|Graphics|SM|not supported|Not Supported|No targets|Unable to find display|Missing Extension' "$LOG" | tail -n 160 || true
