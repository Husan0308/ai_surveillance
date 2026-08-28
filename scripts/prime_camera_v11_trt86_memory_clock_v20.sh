#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
ENGINE="$ROOT/artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine"
EXPECTED_SHA="632245a122f5e85fa99572aeca5fd352adab96580725afe979d6e31f9cdd4c6a"
LOG="${V11_TRT86_CLOCK_PRIME_LOG:-/tmp/CAMERA_V11_TRT86_CLOCK_PRIME.log}"
MAX_ATTEMPTS="${V11_TRT86_CLOCK_PRIME_ATTEMPTS:-12}"

fail() {
  printf 'CAMERA_V11_TRT86_CLOCK_PRIME result=FAIL reason=%s\n' "$*" >&2
  exit 1
}

[[ -s "$ENGINE" ]] || fail "engine_missing"
actual_sha="$(sha256sum "$ENGINE" | awk '{print $1}')"
[[ "$actual_sha" == "$EXPECTED_SHA" ]] || fail "engine_sha256_mismatch actual=$actual_sha"

read_clocks() {
  nvidia-smi \
    --query-gpu=clocks.current.memory,clocks.max.memory \
    --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

[[ "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] || fail "invalid_attempts=$MAX_ATTEMPTS"

IFS=, read -r current_mhz max_mhz <<<"$(read_clocks)"
[[ "$current_mhz" =~ ^[0-9]+$ && "$max_mhz" =~ ^[0-9]+$ ]] || \
  fail "memory_clock_query_invalid current=$current_mhz max=$max_mhz"
minimum_mhz=$((max_mhz * 85 / 100))

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  (( current_mhz >= minimum_mhz )) && break
  printf 'CAMERA_V11_TRT86_CLOCK_PRIME action=disposable-context attempt=%s/%s current=%sMHz max=%sMHz minimum=%sMHz\n' \
    "$attempt" "$MAX_ATTEMPTS" "$current_mhz" "$max_mhz" "$minimum_mhz"
  "$ROOT/.venv-trt86/bin/python" -I \
    "$ROOT/scripts/benchmark_yolo26_trt86_step2_worker.py" \
    --engine "$ENGINE" --warmup 10 --iterations 1 >"$LOG" 2>&1 || \
    fail "disposable_worker_failed attempt=$attempt log=$LOG"

  for _ in $(seq 1 30); do
    IFS=, read -r current_mhz max_mhz <<<"$(read_clocks)"
    (( current_mhz >= minimum_mhz )) && break
    sleep 0.1
  done
done

(( current_mhz >= minimum_mhz )) || \
  fail "memory_clock_not_boosted attempts=$MAX_ATTEMPTS current=${current_mhz}MHz minimum=${minimum_mhz}MHz log=$LOG"

printf 'CAMERA_V11_TRT86_CLOCK_PRIME result=PASS current=%sMHz max=%sMHz minimum=%sMHz engine_sha256=%s\n' \
  "$current_mhz" "$max_mhz" "$minimum_mhz" "$actual_sha"
