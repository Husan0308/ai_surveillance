#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

resolve_python() {
  local candidates=()

  if [[ -n "${FULL_STACK_PYTHON:-}" ]]; then
    candidates+=("$FULL_STACK_PYTHON")
  fi
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    candidates+=("$VIRTUAL_ENV/bin/python")
  fi
  candidates+=(
    "$ROOT/.venv/bin/python"
    "$ROOT/venv/bin/python"
  )

  local path_python
  path_python="$(command -v python3 2>/dev/null || true)"
  [[ -n "$path_python" ]] && candidates+=("$path_python")
  path_python="$(command -v python 2>/dev/null || true)"
  [[ -n "$path_python" ]] && candidates+=("$path_python")

  local candidate
  local checked=""
  for candidate in "${candidates[@]}"; do
    [[ -n "$candidate" ]] || continue
    checked+=" $candidate"
    [[ -x "$candidate" ]] || continue

    if "$candidate" - <<'PY' >/dev/null 2>&1
import fastapi
import httpx
import numpy
import uvicorn
import yaml
import dotenv
import gi
import PySide6
gi.require_version("Gst", "1.0")
from gi.repository import Gst
PY
    then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  echo "FULL_LIVE_STACK no Python interpreter has all required runtime modules." >&2
  echo "Checked:$checked" >&2
  echo "Required imports: fastapi httpx numpy uvicorn yaml dotenv gi(Gst) PySide6" >&2
  echo "Override explicitly with FULL_STACK_PYTHON=/path/to/python if needed." >&2
  return 1
}

bootstrap_full_stack_python() {
  local base
  base="$(command -v python3 2>/dev/null || true)"
  [[ -n "$base" ]] || {
    echo "FULL_LIVE_STACK cannot bootstrap: python3 not found" >&2
    return 1
  }

  # PyGObject/GStreamer on Ubuntu is normally supplied by the system Python
  # packages. Keep access to those while isolating pip-installed app deps.
  if ! "$base" - <<'PY' >/dev/null 2>&1
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
PY
  then
    echo "FULL_LIVE_STACK system Python is missing gi/GStreamer." >&2
    echo "Install Ubuntu packages: sudo apt install python3-gi gir1.2-gstreamer-1.0 python3-venv" >&2
    return 1
  fi

  local env_dir="$ROOT/.runtime/full-stack-venv"
  local py="$env_dir/bin/python"

  if [[ ! -x "$py" ]]; then
    echo "FULL_LIVE_STACK creating runtime venv=$env_dir"
    "$base" -m venv --system-site-packages "$env_dir" || {
      echo "FULL_LIVE_STACK venv creation failed." >&2
      echo "If Ubuntu reports ensurepip/venv missing: sudo apt install python3-venv" >&2
      return 1
    }
  fi

  echo "FULL_LIVE_STACK installing/updating Python service dependencies..."
  "$py" -m pip install --disable-pip-version-check -q     -r services/ml_service/requirements.txt     -r services/api_service/requirements.txt     -r services/frontend/requirements.txt || return 1

  if ! "$py" - <<'PY' >/dev/null 2>&1
import fastapi
import httpx
import numpy
import uvicorn
import yaml
import dotenv
import gi
import PySide6
gi.require_version("Gst", "1.0")
from gi.repository import Gst
PY
  then
    echo "FULL_LIVE_STACK bootstrapped venv still misses a required module" >&2
    return 1
  fi

  printf '%s\n' "$py"
}

if ! PY="$(resolve_python)"; then
  PY="$(bootstrap_full_stack_python)" || exit 1
fi
ROOM_PAIR_DURATION="${FULL_STACK_ROOM_PAIR_DURATION_SEC:-86400}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${FULL_STACK_OUT:-$ROOT/.runtime/full-live-stack-$STAMP}"
LOGS="$OUT/logs"
mkdir -p "$LOGS"

fail() {
  echo "FULL_LIVE_STACK status=FAIL reason=$*" >&2
  exit 1
}

echo "FULL_LIVE_STACK python=$PY"
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
