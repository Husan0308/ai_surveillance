#!/usr/bin/env bash
# Source-only helper for the NVIDIA 580 PowerMizer/VRAM boost regression.
# It deliberately keeps the nvidia-settings GUI process alive only while Step2 runs.

V11_POWERMIZER_KEEPER_PID=""
V11_POWERMIZER_KEEPER_LOG="${V11_POWERMIZER_KEEPER_LOG:-/tmp/CAMERA_V11_POWERMIZER_KEEPER.log}"

v11_powermizer_fail() {
  printf 'CAMERA_V11_POWERMIZER_KEEPER result=FAIL reason=%s\n' "$*" >&2
  return 1
}

v11_powermizer_mem_clock_mhz() {
  nvidia-smi --query-gpu=clocks.current.memory --format=csv,noheader,nounits 2>/dev/null \
    | head -n 1 | tr -d '[:space:]'
}

v11_powermizer_start() {
  [[ -n "${DISPLAY:-}" ]] || v11_powermizer_fail "DISPLAY_empty" || return 1
  command -v nvidia-settings >/dev/null 2>&1 || v11_powermizer_fail "nvidia_settings_missing" || return 1
  command -v nvidia-smi >/dev/null 2>&1 || v11_powermizer_fail "nvidia_smi_missing" || return 1

  : >"$V11_POWERMIZER_KEEPER_LOG"
  DISPLAY="$DISPLAY" nvidia-settings >"$V11_POWERMIZER_KEEPER_LOG" 2>&1 &
  V11_POWERMIZER_KEEPER_PID=$!
  sleep 2
  if ! kill -0 "$V11_POWERMIZER_KEEPER_PID" 2>/dev/null; then
    v11_powermizer_fail "gui_exited log=$V11_POWERMIZER_KEEPER_LOG"
    V11_POWERMIZER_KEEPER_PID=""
    return 1
  fi

  set +e
  DISPLAY="$DISPLAY" nvidia-settings -a '[gpu:0]/GPUPowerMizerMode=1' >>"$V11_POWERMIZER_KEEPER_LOG" 2>&1
  local set_status=$?
  set -e
  if (( set_status != 0 )); then
    v11_powermizer_fail "mode1_assignment_failed status=$set_status log=$V11_POWERMIZER_KEEPER_LOG"
    v11_powermizer_stop
    return 1
  fi

  printf 'CAMERA_V11_POWERMIZER_KEEPER result=STARTED pid=%s display=%s requested_mode=1\n' \
    "$V11_POWERMIZER_KEEPER_PID" "$DISPLAY"
}

v11_powermizer_verify_boost() {
  local minimum_mhz="${V11_POWERMIZER_MIN_MEMORY_MHZ:-3000}"
  local attempts="${V11_POWERMIZER_VERIFY_ATTEMPTS:-30}"
  local clock=""
  [[ "$minimum_mhz" =~ ^[0-9]+$ ]] || v11_powermizer_fail "invalid_min_memory_mhz=$minimum_mhz" || return 1
  [[ "$attempts" =~ ^[1-9][0-9]*$ ]] || v11_powermizer_fail "invalid_verify_attempts=$attempts" || return 1
  [[ -n "$V11_POWERMIZER_KEEPER_PID" ]] && kill -0 "$V11_POWERMIZER_KEEPER_PID" 2>/dev/null \
    || v11_powermizer_fail "keeper_not_alive" || return 1

  for _ in $(seq 1 "$attempts"); do
    clock="$(v11_powermizer_mem_clock_mhz || true)"
    if [[ "$clock" =~ ^[0-9]+$ ]] && (( clock >= minimum_mhz )); then
      printf 'CAMERA_V11_POWERMIZER_KEEPER result=BOOST_OK memory_mhz=%s minimum_mhz=%s pid=%s\n' \
        "$clock" "$minimum_mhz" "$V11_POWERMIZER_KEEPER_PID"
      return 0
    fi
    sleep 0.1
  done

  v11_powermizer_fail "memory_clock_not_boosted memory_mhz=${clock:-unknown} minimum_mhz=$minimum_mhz"
  return 1
}

v11_powermizer_stop() {
  if [[ -n "${V11_POWERMIZER_KEEPER_PID:-}" ]] && kill -0 "$V11_POWERMIZER_KEEPER_PID" 2>/dev/null; then
    kill -TERM "$V11_POWERMIZER_KEEPER_PID" 2>/dev/null || true
    wait "$V11_POWERMIZER_KEEPER_PID" 2>/dev/null || true
    printf 'CAMERA_V11_POWERMIZER_KEEPER result=STOPPED pid=%s\n' "$V11_POWERMIZER_KEEPER_PID"
  fi
  V11_POWERMIZER_KEEPER_PID=""
}
