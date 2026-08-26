#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

export CAMERA_SERVICE_RTSP_TRANSPORT="${CAMERA_SERVICE_RTSP_TRANSPORT:-tcp}"
export CAMERA_SERVICE_RTSP_LATENCY_MS="${CAMERA_SERVICE_RTSP_LATENCY_MS:-80}"
export CAMERA_SERVICE_SOURCE_FPS="${CAMERA_SERVICE_SOURCE_FPS:-20}"
export CAMERA_SERVICE_EXTRA_SURFACES="${CAMERA_SERVICE_EXTRA_SURFACES:-8}"
export CAMERA_SERVICE_DISPLAY_WIDTH="${CAMERA_SERVICE_DISPLAY_WIDTH:-1280}"
export CAMERA_SERVICE_DISPLAY_HEIGHT="${CAMERA_SERVICE_DISPLAY_HEIGHT:-720}"
export CAMERA_SERVICE_WALL_WIDTH="${CAMERA_SERVICE_WALL_WIDTH:-1920}"
export CAMERA_SERVICE_WALL_HEIGHT="${CAMERA_SERVICE_WALL_HEIGHT:-720}"
export CAMERA_SERVICE_STARTUP_STAGGER_SEC="${CAMERA_SERVICE_STARTUP_STAGGER_SEC:-0.50}"

fail() { printf 'CAMERA_SERVICE_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }

for plugin in nvurisrcbin queue nvstreammux nvmultistreamtiler nvvideoconvert capsfilter nveglglessink; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing plugin: $plugin"
done

MAIN_PYTHON=""
for candidate in "${CAMERA_SERVICE_PYTHON:-}" "$ROOT/.venv/bin/python" "$(command -v python3 2>/dev/null || true)"; do
  [[ -n "$candidate" && -x "$candidate" ]] || continue
  if "$candidate" - <<'PY' >/dev/null 2>&1
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: F401
import yaml, dotenv  # noqa: F401
import services.camera_service.app.runtime  # noqa: F401
PY
  then MAIN_PYTHON="$candidate"; break; fi
done
[[ -n "$MAIN_PYTHON" ]] || fail "no Python can import camera_service runtime"

printf '%s\n' \
  "CAMERA_SERVICE_PROFILE source=${CAMERA_SERVICE_SOURCE_FPS}fps rtsp=${CAMERA_SERVICE_RTSP_LATENCY_MS}ms extra_surfaces=${CAMERA_SERVICE_EXTRA_SURFACES}" \
  "CAMERA_SERVICE_PROFILE display=${CAMERA_SERVICE_DISPLAY_WIDTH}x${CAMERA_SERVICE_DISPLAY_HEIGHT} wall=${CAMERA_SERVICE_WALL_WIDTH}x${CAMERA_SERVICE_WALL_HEIGHT}" \
  "CAMERA_SERVICE_BOUNDARY ai=0 detector=0 tracker=0 reid=0 identity=0 api=0 frontend=0" \
  "CAMERA_SERVICE_MAIN executable=$MAIN_PYTHON"

exec "$MAIN_PYTHON" -u -m services.camera_service.app.runtime
