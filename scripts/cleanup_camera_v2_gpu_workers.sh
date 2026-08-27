#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
ME_UID="$(id -u)"

CANDIDATES=()

is_repo_owned_pid() {
  local pid="$1"
  [[ -r "/proc/$pid/status" ]] || return 1
  local uid cwd
  uid="$(awk '/^Uid:/{print $2; exit}' "/proc/$pid/status" 2>/dev/null || true)"
  [[ "$uid" == "$ME_UID" ]] || return 1
  cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
  [[ "$cwd" == "$ROOT" ]]
}

add_pid() {
  local pid="$1"
  [[ -n "${pid:-}" && "$pid" != "$$" ]] || return 0
  is_repo_owned_pid "$pid" || return 0
  CANDIDATES+=("$pid")
}

# CUDA-owning children currently visible to nvidia-smi.
while read -r pid; do
  [[ -n "${pid:-}" ]] || continue
  is_repo_owned_pid "$pid" || continue
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  if [[ "$cmd" == *"yolo26_trt86"* || "$cmd" == *"multiprocessing.spawn"* || "$cmd" == *"services.camera_v2"* ]]; then
    add_pid "$pid"
  fi
done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF{print $1}' | sort -nu)

# Repo camera supervisors/runtimes. Include historical launchers because they can keep
# a shell alive and respawn CUDA children even when nvidia-smi is momentarily empty.
LEGACY_RE='run_camera_v2_detection_lowlat\.sh|run_camera_v2_detection_sticky\.sh'
CURRENT_RE='run_camera_v2_|services\.camera_v2|camera-v8-trt86|yolo26_trt86_batch6_worker|yolo26_trt86_shm_worker'
while read -r pid _rest; do
  add_pid "$pid"
done < <(pgrep -af "${LEGACY_RE}|${CURRENT_RE}" || true)

# Recursively include descendants of every matched repo-owned parent. This prevents a
# killed launcher from leaving an already-running Python/TRT child behind.
queue=("${CANDIDATES[@]:-}")
seen=" "
while ((${#queue[@]})); do
  parent="${queue[0]}"
  queue=("${queue[@]:1}")
  [[ "$seen" == *" $parent "* ]] && continue
  seen+="$parent "
  while read -r child; do
    [[ -n "${child:-}" ]] || continue
    if is_repo_owned_pid "$child"; then
      CANDIDATES+=("$child")
      queue+=("$child")
    fi
  done < <(pgrep -P "$parent" 2>/dev/null || true)
done

if ((${#CANDIDATES[@]} == 0)); then
  echo 'CAMERA_V83_CLEANUP candidates=0 status=clean'
  exit 0
fi

mapfile -t CANDIDATES < <(printf '%s\n' "${CANDIDATES[@]}" | awk 'NF' | sort -nu)
printf 'CAMERA_V83_CLEANUP candidates=%s signal=TERM\n' "${CANDIDATES[*]}"

# Send TERM to the complete tree at once so supervisor and children cannot race by
# respawning each other during shutdown.
kill -TERM "${CANDIDATES[@]}" 2>/dev/null || true

for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
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
  printf 'CAMERA_V83_CLEANUP stubborn=%s signal=KILL\n' "${alive[*]}"
  kill -KILL "${alive[@]}" 2>/dev/null || true
  sleep 0.4
fi

remaining=()
for pid in "${CANDIDATES[@]}"; do
  kill -0 "$pid" 2>/dev/null && remaining+=("$pid")
done

# Also verify no matching repo-owned launcher survived under a different PID.
while read -r pid _rest; do
  [[ -n "${pid:-}" && "$pid" != "$$" ]] || continue
  if is_repo_owned_pid "$pid"; then
    remaining+=("$pid")
  fi
done < <(pgrep -af "${LEGACY_RE}|${CURRENT_RE}" || true)

if ((${#remaining[@]})); then
  mapfile -t remaining < <(printf '%s\n' "${remaining[@]}" | sort -nu)
  printf 'CAMERA_V83_CLEANUP status=FAIL remaining=%s\n' "${remaining[*]}"
  exit 1
fi

echo 'CAMERA_V83_CLEANUP status=OK remaining=0'
