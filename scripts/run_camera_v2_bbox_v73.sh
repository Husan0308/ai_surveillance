#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
LOCK_FILE="/tmp/ai_surveillance_camera_v2_gpu.lock"

fail() { printf 'CAMERA_BBOX_V73_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }

command -v flock >/dev/null 2>&1 || fail "flock missing (install util-linux)"
exec 9>"$LOCK_FILE"
flock -n 9 || fail "another Camera V2 owner already holds $LOCK_FILE; stop/cleanup it first"

# Refuse to start on top of an already-running repo-owned CUDA sidecar. V7 had 16-20ms
# isolated TRT; later runs at 160-190ms with almost no NvDCF are a system-state signal,
# not a bbox-tuning signal.
mapfile -t GPU_PIDS < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF{print $1}' | sort -nu)
foreign_repo_gpu=()
for pid in "${GPU_PIDS[@]:-}"; do
  [[ -r "/proc/$pid/status" ]] || continue
  uid="$(awk '/^Uid:/{print $2; exit}' "/proc/$pid/status" 2>/dev/null || true)"
  [[ "$uid" == "$(id -u)" ]] || continue
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
  if [[ "$cwd" == "$ROOT" && ( "$cmd" == *"yolo26_trt86_shm_worker_v4.py"* || "$cmd" == *"multiprocessing.spawn"* || "$cmd" == *"services.camera_v2"* ) ]]; then
    foreign_repo_gpu+=("$pid")
  elif [[ "$cmd" == *"$ROOT/scripts/yolo26_trt86_shm_worker_v4.py"* ]]; then
    foreign_repo_gpu+=("$pid")
  fi
done

if ((${#foreign_repo_gpu[@]})); then
  fail "repo-owned CUDA process already active pid(s)=${foreign_repo_gpu[*]}; run: bash scripts/cleanup_camera_v2_gpu_workers.sh"
fi

printf '%s\n' \
  "CAMERA_BBOX_V73_PREFLIGHT status=OK single_owner=1 stale_repo_gpu=0" \
  "CAMERA_BBOX_V73_POLICY algorithm=v7.2-unchanged purpose=gpu-state-isolation"

# Keep the V7.2 algorithm bit-for-bit unchanged; V7.3 only adds process ownership guards.
exec bash "$ROOT/scripts/run_camera_v2_bbox_v72.sh"
