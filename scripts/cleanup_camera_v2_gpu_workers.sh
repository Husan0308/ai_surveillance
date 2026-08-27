#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
ME_UID="$(id -u)"

mapfile -t GPU_PIDS < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF{print $1}' | sort -nu)
CANDIDATES=()

for pid in "${GPU_PIDS[@]:-}"; do
  [[ -r "/proc/$pid/status" ]] || continue
  uid="$(awk '/^Uid:/{print $2; exit}' "/proc/$pid/status" 2>/dev/null || true)"
  [[ "$uid" == "$ME_UID" ]] || continue
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
  # Only touch this repo's own camera/TRT processes. Never kill unrelated CUDA apps.
  if [[ "$cwd" == "$ROOT" && ( "$cmd" == *"yolo26_trt86_shm_worker_v4.py"* || "$cmd" == *"multiprocessing.spawn"* || "$cmd" == *"services.camera_v2"* ) ]]; then
    CANDIDATES+=("$pid")
  elif [[ "$cmd" == *"$ROOT/scripts/yolo26_trt86_shm_worker_v4.py"* ]]; then
    CANDIDATES+=("$pid")
  fi
done

# Also include live camera runtime parents in this repo, even if nvidia-smi only lists
# their CUDA-owning child. This ensures the process tree cannot immediately respawn it.
while read -r pid _rest; do
  [[ -n "${pid:-}" && "$pid" != "$$" ]] || continue
  [[ -r "/proc/$pid/status" ]] || continue
  uid="$(awk '/^Uid:/{print $2; exit}' "/proc/$pid/status" 2>/dev/null || true)"
  [[ "$uid" == "$ME_UID" ]] || continue
  cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
  [[ "$cwd" == "$ROOT" ]] || continue
  CANDIDATES+=("$pid")
done < <(pgrep -af 'services\.camera_v2\.runtime_bbox_v7|run_camera_v2_bbox_v7' || true)

if ((${#CANDIDATES[@]} == 0)); then
  echo 'CAMERA_V73_CLEANUP candidates=0 status=clean'
  exit 0
fi

mapfile -t CANDIDATES < <(printf '%s\n' "${CANDIDATES[@]}" | sort -nu)
printf 'CAMERA_V73_CLEANUP candidates=%s signal=TERM\n' "${CANDIDATES[*]}"
kill -TERM "${CANDIDATES[@]}" 2>/dev/null || true

for _ in 1 2 3 4 5 6 7 8 9 10; do
  alive=()
  for pid in "${CANDIDATES[@]}"; do
    kill -0 "$pid" 2>/dev/null && alive+=("$pid")
  done
  ((${#alive[@]} == 0)) && break
  sleep 0.2
done

alive=()
for pid in "${CANDIDATES[@]}"; do
  kill -0 "$pid" 2>/dev/null && alive+=("$pid")
done
if ((${#alive[@]})); then
  printf 'CAMERA_V73_CLEANUP stubborn=%s signal=KILL\n' "${alive[*]}"
  kill -KILL "${alive[@]}" 2>/dev/null || true
  sleep 0.3
fi

remaining=0
for pid in "${CANDIDATES[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    remaining=$((remaining + 1))
  fi
done
printf 'CAMERA_V73_CLEANUP status=%s remaining=%d\n' "$([[ $remaining -eq 0 ]] && echo OK || echo FAIL)" "$remaining"
[[ $remaining -eq 0 ]]
