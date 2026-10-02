#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${FULL_STACK_PYTHON:-$ROOT/.venv/bin/python}"
ROOM_PAIR_DURATION="${FULL_STACK_ROOM_PAIR_DURATION_SEC:-86400}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${FULL_STACK_OUT:-$ROOT/.runtime/full-live-stack-$STAMP}"
LOGS="$OUT/logs"
mkdir -p "$LOGS"

fail() {
  echo "FULL_LIVE_STACK status=FAIL reason=$*" >&2
  exit 1
}

[[ -x "$PY" ]] || fail "python_not_found=$PY"
command -v curl >/dev/null 2>&1 || fail "curl_not_found"
command -v flock >/dev/null 2>&1 || fail "flock_not_found"

exec 9>/tmp/ai_surveillance_full_live_stack.lock
flock -n 9 || fail "another_full_live_stack_launcher_is_running"

# Do not silently take over ports from an unrelated process.
if curl -fsS --max-time 1 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  fail "api_port_8000_already_serving"
fi
if curl -fsS --max-time 1 http://127.0.0.1:8001/health >/dev/null 2>&1; then
  fail "ml_port_8001_already_serving"
fi

PIDS=()
ROOM_PID=""
PREVIEW_PID=""
ML_PID=""
API_PID=""

cleanup() {
  set +e
  echo "FULL_LIVE_STACK stopping..."

  if [[ -n "$API_PID" ]] && kill -0 "$API_PID" 2>/dev/null; then
    kill -TERM "$API_PID" 2>/dev/null || true
  fi
  if [[ -n "$ML_PID" ]] && kill -0 "$ML_PID" 2>/dev/null; then
    kill -TERM "$ML_PID" 2>/dev/null || true
  fi
  if [[ -n "$PREVIEW_PID" ]] && kill -0 "$PREVIEW_PID" 2>/dev/null; then
    kill -TERM "$PREVIEW_PID" 2>/dev/null || true
  fi
  # SIGINT lets run_room_pair unwind its finally block and stop its container,
  # Kafka capture, monitors, and identity worker cleanly.
  if [[ -n "$ROOM_PID" ]] && kill -0 "$ROOM_PID" 2>/dev/null; then
    kill -INT "$ROOM_PID" 2>/dev/null || true
  fi

  for _ in {1..30}; do
    alive=0
    for pid in "$API_PID" "$ML_PID" "$PREVIEW_PID" "$ROOM_PID"; do
      [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && alive=1
    done
    [[ "$alive" -eq 0 ]] && break
    sleep 0.2
  done

  for pid in "$API_PID" "$ML_PID" "$PREVIEW_PID" "$ROOM_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done

  echo "FULL_LIVE_STACK stopped logs=$OUT"
}
trap cleanup EXIT INT TERM

echo "FULL_LIVE_STACK output=$OUT"
echo "FULL_LIVE_STACK starting CAM-01/CAM-04 room-pair live owner..."

MV3DT_DEV_ROOM_SOURCE_MODE=live MV3DT_UI_PREVIEW_DIR=/dev/shm "$PY" -u scripts/dev_room_mv3dt/run_room_pair.py   --mode live   --duration "$ROOM_PAIR_DURATION"   --skip-render   >"$LOGS/room_pair.log" 2>&1 &
ROOM_PID=$!

echo "FULL_LIVE_STACK starting CAM-02/CAM-03/CAM-05/CAM-06 preview owner..."

"$PY" -u -m services.camera_v11.preview_only_runtime   --cameras CAM-02,CAM-03,CAM-05,CAM-06   --stats "$OUT/non_dev_preview_stats.json"   >"$LOGS/non_dev_preview.log" 2>&1 &
PREVIEW_PID=$!

echo "FULL_LIVE_STACK starting ml_service on :8001..."
"$PY" -u -m services.ml_service.app.main   >"$LOGS/ml_service.log" 2>&1 &
ML_PID=$!

for _ in {1..80}; do
  if curl -fsS --max-time 1 http://127.0.0.1:8001/health >"$OUT/ml_health.json" 2>/dev/null; then
    break
  fi
  kill -0 "$ML_PID" 2>/dev/null || {
    tail -80 "$LOGS/ml_service.log" >&2 || true
    fail "ml_service_exited_early"
  }
  sleep 0.25
done
curl -fsS --max-time 2 http://127.0.0.1:8001/health >"$OUT/ml_health.json"   || fail "ml_service_health_timeout"

echo "FULL_LIVE_STACK starting api_service on :8000..."
"$PY" -u -m services.api_service.app.main   >"$LOGS/api_service.log" 2>&1 &
API_PID=$!

for _ in {1..80}; do
  if curl -fsS --max-time 1 http://127.0.0.1:8000/health >"$OUT/api_health.json" 2>/dev/null; then
    break
  fi
  kill -0 "$API_PID" 2>/dev/null || {
    tail -80 "$LOGS/api_service.log" >&2 || true
    fail "api_service_exited_early"
  }
  sleep 0.25
done
curl -fsS --max-time 2 http://127.0.0.1:8000/health >"$OUT/api_health.json"   || fail "api_service_health_timeout"

# Wait for the six shared-memory preview files. CAM-01/CAM-04 are produced by
# the room-pair native runtime; the other four are produced by preview_only_runtime.
PREVIEWS=(
  /dev/shm/v11_ui_preview_cam01_v1.bin
  /dev/shm/v11_ui_preview_cam02_v1.bin
  /dev/shm/v11_ui_preview_cam03_v1.bin
  /dev/shm/v11_ui_preview_cam04_v1.bin
  /dev/shm/v11_ui_preview_cam05_v1.bin
  /dev/shm/v11_ui_preview_cam06_v1.bin
)

for _ in {1..160}; do
  ready=0
  for path in "${PREVIEWS[@]}"; do
    [[ -s "$path" ]] && ready=$((ready + 1))
  done
  [[ "$ready" -eq 6 ]] && break

  kill -0 "$ROOM_PID" 2>/dev/null || {
    tail -120 "$LOGS/room_pair.log" >&2 || true
    fail "room_pair_exited_before_preview_ready"
  }
  kill -0 "$PREVIEW_PID" 2>/dev/null || {
    tail -120 "$LOGS/non_dev_preview.log" >&2 || true
    fail "non_dev_preview_exited_before_preview_ready"
  }
  sleep 0.25
done

ready=0
for path in "${PREVIEWS[@]}"; do
  [[ -s "$path" ]] && ready=$((ready + 1))
done
if [[ "$ready" -ne 6 ]]; then
  echo "=== ROOM PAIR LOG ===" >&2
  tail -120 "$LOGS/room_pair.log" >&2 || true
  echo "=== NON-DEV PREVIEW LOG ===" >&2
  tail -120 "$LOGS/non_dev_preview.log" >&2 || true
  fail "only_${ready}_of_6_previews_ready"
fi

curl -fsS --max-time 2 http://127.0.0.1:8000/api/v1/ml/health   >"$OUT/api_ml_health.json" 2>/dev/null || true
curl -fsS --max-time 2 http://127.0.0.1:8000/api/v1/cameras   >"$OUT/api_cameras.json" 2>/dev/null || true
curl -fsS --max-time 2 http://127.0.0.1:8000/api/v1/room-pair/identity   >"$OUT/api_room_pair_identity.json" 2>/dev/null || true

echo
echo "FULL_LIVE_STACK services_started"
echo "  ML:       http://127.0.0.1:8001/health"
echo "  API:      http://127.0.0.1:8000/health"
echo "  Cameras:  CAM-01..CAM-06 shared-memory previews ready"
echo "  Logs:     $OUT"
echo
echo "Opening PySide6 UI. Close the UI or press Ctrl+C here to stop the stack."

FRONTEND_USE_V11_SHARED_MEMORY=1 FRONTEND_API_BASE_URL=http://127.0.0.1:8000 FRONTEND_ML_VIDEO_BASE_URL=http://127.0.0.1:8001 "$PY" -u -m services.frontend.app.main   2>>"$LOGS/frontend.stderr.log"
