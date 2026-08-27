#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
uid_now="$(id -u)"

mapfile -t pids < <(
  for p in /proc/[0-9]*; do
    pid="${p##*/}"
    [[ -r "$p/status" && -r "$p/cmdline" ]] || continue
    uid="$(awk '/^Uid:/{print $2; exit}' "$p/status" 2>/dev/null || true)"
    [[ "$uid" == "$uid_now" ]] || continue
    cmd="$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null || true)"
    cwd="$(readlink -f "$p/cwd" 2>/dev/null || true)"
    if [[ "$cwd" == "$ROOT" && ( \
      "$cmd" == *"services.camera_v2.runtime_v8_pascal"* || \
      "$cmd" == *"camera-v8-trt86-batch6-bridge"* || \
      "$cmd" == *"yolo26_trt86_batch6_worker_v8.py"* ) ]]; then
      echo "$pid"
    elif [[ "$cmd" == *"$ROOT/scripts/yolo26_trt86_batch6_worker_v8.py"* ]]; then
      echo "$pid"
    fi
  done | sort -nu
)

if ((${#pids[@]} == 0)); then
  echo "CAMERA_V8_CLEANUP status=CLEAN processes=0"
  exit 0
fi

echo "CAMERA_V8_CLEANUP terminate=${pids[*]}"
kill -TERM "${pids[@]}" 2>/dev/null || true
for _ in {1..20}; do
  alive=()
  for pid in "${pids[@]}"; do
    kill -0 "$pid" 2>/dev/null && alive+=("$pid")
  done
  ((${#alive[@]} == 0)) && { echo "CAMERA_V8_CLEANUP status=CLEAN"; exit 0; }
  sleep 0.1
done
alive=()
for pid in "${pids[@]}"; do
  kill -0 "$pid" 2>/dev/null && alive+=("$pid")
done
if ((${#alive[@]})); then
  echo "CAMERA_V8_CLEANUP kill=${alive[*]}" >&2
  kill -KILL "${alive[@]}" 2>/dev/null || true
fi
echo "CAMERA_V8_CLEANUP status=CLEAN"
