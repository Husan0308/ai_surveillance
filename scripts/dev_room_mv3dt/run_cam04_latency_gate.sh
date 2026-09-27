#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

[[ "$(git branch --show-current)" == "wip/cam04-latency-decouple" ]] || {
  echo "CAM04_LATENCY_GATE status=FAIL reason=wrong_branch current=$(git branch --show-current)" >&2
  exit 1
}
[[ -n "${DISPLAY:-}" ]] || {
  echo "CAM04_LATENCY_GATE status=FAIL reason=DISPLAY_empty" >&2
  exit 1
}

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/.runtime/cam04-latency-gate-$STAMP"
mkdir -p "$OUT"

bash scripts/dev_room_mv3dt/build_cam04_latency_diagnostic.sh | tee "$OUT/build.log"

BUILD_JSON="$ROOT/.runtime/mv3dt/diagnostic-bin/cam04-latency/build.json"
[[ -f "$BUILD_JSON" ]] || {
  echo "CAM04_LATENCY_GATE status=FAIL reason=missing_build_json" >&2
  exit 1
}

readarray -t BUILD_VALUES < <(python3 - "$BUILD_JSON" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
print(d["binary"])
print(d["sha256"])
print(d["source_sha256"])
PY
)
BIN="${BUILD_VALUES[0]}"
HASH="${BUILD_VALUES[1]}"
SOURCE_HASH="${BUILD_VALUES[2]}"

TIMING_LOG="$OUT/preview-latency.jsonl"
RUN_LOG="$OUT/room-pair.log"
UI_LOG="$OUT/frontend.log"

SHM_CLEAN_IMAGE="${MV3DT_BUILD_IMAGE:-nvcr.io/nvidia/deepstream:9.1-triton-multiarch}"
echo "CAM04_LATENCY_GATE cleaning_stale_shm=true image=$SHM_CLEAN_IMAGE"
docker run --rm --pull=never --ipc=host --entrypoint sh "$SHM_CLEAN_IMAGE" -c '
  rm -f /dev/shm/v11_ui_preview_cam01_v1.bin /dev/shm/v11_ui_preview_cam04_v1.bin
'
for path in /dev/shm/v11_ui_preview_cam01_v1.bin /dev/shm/v11_ui_preview_cam04_v1.bin; do
  if [[ -e "$path" ]]; then
    echo "CAM04_LATENCY_GATE status=FAIL reason=stale_shm_cleanup_failed path=$path" >&2
    ls -l "$path" >&2 || true
    exit 1
  fi
done

echo "CAM04_LATENCY_GATE binary=$BIN"
echo "CAM04_LATENCY_GATE sha256=$HASH"
echo "CAM04_LATENCY_GATE source_sha256=$SOURCE_HASH"
echo "CAM04_LATENCY_GATE queue_buffers=${MV3DT_TEST_CAM04_ANALYTICS_QUEUE_BUFFERS:-16}"

MV3DT_BINARY="$BIN" MV3DT_DIAGNOSTIC_BINARY_SHA256="$HASH" MV3DT_DIAGNOSTIC_SOURCE_SHA256="$SOURCE_HASH" MV3DT_FRAME_AUDIT_LOG=1 MV3DT_SOURCE_HEALTH_DIR=1 MV3DT_TEST_CAM04_ANALYTICS_QUEUE_BUFFERS="${MV3DT_TEST_CAM04_ANALYTICS_QUEUE_BUFFERS:-16}" python3 scripts/dev_room_mv3dt/run_room_pair.py   --mode live --duration 220 --skip-render   >"$RUN_LOG" 2>&1 &
ROOM_PID=$!

cleanup() {
  if kill -0 "$ROOM_PID" 2>/dev/null; then
    kill "$ROOM_PID" 2>/dev/null || true
    wait "$ROOM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ready=0
for _ in $(seq 1 90); do
  if [[ -s /dev/shm/v11_ui_preview_cam04_v1.bin ]]; then
    ready=1
    break
  fi
  if ! kill -0 "$ROOM_PID" 2>/dev/null; then
    echo "CAM04_LATENCY_GATE status=FAIL reason=room_pair_exited_early" >&2
    tail -80 "$RUN_LOG" >&2 || true
    exit 1
  fi
  sleep 1
done
[[ "$ready" == "1" ]] || {
  echo "CAM04_LATENCY_GATE status=FAIL reason=cam04_preview_not_ready" >&2
  tail -80 "$RUN_LOG" >&2 || true
  exit 1
}

echo "CAM04_LATENCY_GATE preview_ready=true"

set +e
FRONTEND_USE_V11_SHARED_MEMORY=1 MV3DT_PREVIEW_LATENCY_LOG="$TIMING_LOG" timeout --signal=TERM --kill-after=10s 190s python3 -m services.frontend.app.main >"$UI_LOG" 2>&1
UI_RC=$?
set -e

# timeout(1) returns 124 for the expected timed stop.
if [[ "$UI_RC" != "0" && "$UI_RC" != "124" ]]; then
  echo "CAM04_LATENCY_GATE status=FAIL reason=frontend_exit rc=$UI_RC" >&2
  tail -80 "$UI_LOG" >&2 || true
  exit 1
fi

set +e
wait "$ROOM_PID"
ROOM_RC=$?
set -e
trap - EXIT INT TERM
if [[ "$ROOM_RC" != "0" ]]; then
  echo "CAM04_LATENCY_GATE status=FAIL reason=room_pair_exit rc=$ROOM_RC" >&2
  tail -100 "$RUN_LOG" >&2 || true
  exit 1
fi

[[ -s "$TIMING_LOG" ]] || {
  echo "CAM04_LATENCY_GATE status=FAIL reason=no_timing_log" >&2
  exit 1
}

LATEST_RUN="$(ls -1dt "$ROOT"/.runtime/mv3dt/dev-room-cam01-cam04/live-* 2>/dev/null | head -1 || true)"
if [[ -n "$LATEST_RUN" ]]; then
  echo "$LATEST_RUN" > "$OUT/run-path.txt"
  DIAG="$LATEST_RUN/run/logs/probe/preview_decoder_timing.jsonl"
  if [[ -f "$DIAG" ]]; then
    tail -20 "$DIAG" > "$OUT/preview-decoder-timing-tail.jsonl"
    echo "=== CAM-04 decoder/analytics queue diagnostics tail ==="
    grep '"camera_id":"CAM-04"' "$DIAG" | tail -5 || true
  fi
fi

echo "=== CAM-04 latency audit ==="
python3 scripts/audit_six_camera_preview_latency.py   --log "$TIMING_LOG"   --cameras CAM-04   --min-duration-seconds 180

echo "CAM04_LATENCY_GATE status=PASS out=$OUT"
