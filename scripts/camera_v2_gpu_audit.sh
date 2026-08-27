#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

printf 'CAMERA_V73_GPU_AUDIT root=%s\n' "$ROOT"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo 'CAMERA_V73_GPU_AUDIT ERROR nvidia-smi missing' >&2
  exit 2
fi

printf '%s\n' '--- GPU summary ---'
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true

printf '%s\n' '--- Compute applications ---'
mapfile -t GPU_PIDS < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF{print $1}' | sort -nu)
if ((${#GPU_PIDS[@]} == 0)); then
  echo 'CAMERA_V73_GPU_AUDIT compute_apps=0'
else
  for pid in "${GPU_PIDS[@]}"; do
    [[ -r "/proc/$pid/status" ]] || continue
    uid="$(awk '/^Uid:/{print $2; exit}' "/proc/$pid/status" 2>/dev/null || true)"
    user_name="$(getent passwd "$uid" 2>/dev/null | cut -d: -f1 || true)"
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
    ppid="$(awk '/^PPid:/{print $2; exit}' "/proc/$pid/status" 2>/dev/null || true)"
    printf 'CAMERA_V73_GPU_PROCESS pid=%s ppid=%s user=%s cwd=%q cmd=%q\n' \
      "$pid" "${ppid:-?}" "${user_name:-?}" "${cwd:-?}" "${cmd:-?}"
  done
fi

printf '%s\n' '--- Camera runtime processes ---'
pgrep -af 'services\.camera_v2\.runtime_bbox_v7|run_camera_v2_bbox_v7|yolo26_trt86_shm_worker_v4\.py' || true

printf '%s\n' '--- Repo-owned GPU candidates ---'
repo_gpu=0
for pid in "${GPU_PIDS[@]:-}"; do
  [[ -r "/proc/$pid/status" ]] || continue
  uid="$(awk '/^Uid:/{print $2; exit}' "/proc/$pid/status" 2>/dev/null || true)"
  [[ "$uid" == "$(id -u)" ]] || continue
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
  if [[ "$cwd" == "$ROOT" || "$cmd" == *"$ROOT"* || "$cmd" == *"yolo26_trt86_shm_worker_v4.py"* ]]; then
    repo_gpu=$((repo_gpu + 1))
    printf 'CAMERA_V73_GPU_REPO_PROCESS pid=%s cwd=%q cmd=%q\n' "$pid" "${cwd:-?}" "${cmd:-?}"
  fi
done
printf 'CAMERA_V73_GPU_AUDIT repo_gpu_processes=%d\n' "$repo_gpu"
