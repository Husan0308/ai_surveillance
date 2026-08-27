#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."
ROOT="$PWD"
TRT_PY="${CAMERA_V2_TRT86_PYTHON:-$ROOT/.venv-trt86/bin/python}"

if pgrep -af 'run_camera_v2_detection_lowlat|run_camera_v2_detection_sticky|services\.camera_v2|yolo26_trt86' >/tmp/v84_repo_gpu_ps.$$ 2>/dev/null; then
  cat /tmp/v84_repo_gpu_ps.$$
  rm -f /tmp/v84_repo_gpu_ps.$$
  echo 'V84_BATCH_COMPARE FAIL reason=repo_camera_processes_alive'
  exit 2
fi
rm -f /tmp/v84_repo_gpu_ps.$$

sample_loop() {
  local out="$1"
  : >"$out"
  while true; do
    nvidia-smi --query-gpu=pstate,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu \
      --format=csv,noheader,nounits 2>/dev/null >>"$out" || true
    sleep 0.2
  done
}

run_case() {
  local name="$1"; shift
  local mon="/tmp/V84_${name}_GPU.log"
  sample_loop "$mon" &
  local mon_pid=$!
  "$@"
  local rc=$?
  kill "$mon_pid" 2>/dev/null || true
  wait "$mon_pid" 2>/dev/null || true
  echo "V84_${name}_MONITOR unique_samples_begin"
  sort -u "$mon" | head -n 30
  echo "V84_${name}_MONITOR unique_samples_end"
  return "$rc"
}

echo 'V84_BATCH_COMPARE start=batch1'
run_case B1 "$TRT_PY" scripts/check_trt86_isolation_v73.py --warmup 20 --runs 40
b1_rc=$?

echo 'V84_BATCH_COMPARE start=batch6'
run_case B6 "$TRT_PY" scripts/check_trt86_batch6_cleanroom_v83.py --warmup 20 --runs 40
b6_rc=$?

echo "V84_BATCH_COMPARE done b1_rc=$b1_rc b6_rc=$b6_rc"
echo "V84_BATCH_COMPARE note='Ignore old PASS/FAIL thresholds; compare measured latency, throughput, and memory clock.'"
exit 0
