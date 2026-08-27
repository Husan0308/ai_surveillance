#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

echo "V84_CLOCK_DIAG root=$ROOT"

echo "--- repo camera processes ---"
if pgrep -af 'run_camera_v2_detection_lowlat|run_camera_v2_detection_sticky|services\.camera_v2|yolo26_trt86' ; then
  echo "V84_CLOCK_DIAG FAIL reason=repo_camera_processes_alive"
  echo "V84_CLOCK_DIAG next='bash scripts/cleanup_camera_v2_gpu_workers.sh'"
  exit 2
fi
echo "V84_CLOCK_DIAG repo_camera_processes=0"

echo "--- current GPU clocks/state ---"
nvidia-smi --query-gpu=name,pstate,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,utilization.gpu,temperature.gpu,power.limit \
  --format=csv,noheader,nounits 2>/dev/null || \
  nvidia-smi -q -d PERFORMANCE,CLOCK,TEMPERATURE,POWER

echo "--- supported clocks ---"
nvidia-smi -q -d SUPPORTED_CLOCKS 2>&1 | sed -n '1,220p'

echo "--- memory-clock lock capability ---"
if nvidia-smi -lmi 2>&1; then
  :
else
  echo "V84_CLOCK_DIAG memory_lock_info=unsupported_or_driver_rejected"
fi

echo "--- nvidia-settings PowerMizer (read only) ---"
if command -v nvidia-settings >/dev/null 2>&1; then
  nvidia-settings -q '[gpu:0]/GPUPowerMizerMode' 2>&1 || true
  nvidia-settings -q '[gpu:0]/GPUCurrentClockFreqs' 2>&1 || true
  nvidia-settings -q '[gpu:0]/GPUPerfModes' 2>&1 || true
else
  echo "V84_CLOCK_DIAG nvidia_settings=missing"
fi

echo "V84_CLOCK_DIAG note='Do not lock/overclock anything yet. Send this output first.'"
