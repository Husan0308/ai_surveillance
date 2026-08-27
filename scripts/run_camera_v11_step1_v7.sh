#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="${V11_PYTHON:-$ROOT/.venv-trt86/bin/python}"
LOCK_FILE="/tmp/ai_surveillance_camera_v11_step1_v7.lock"

fail() { printf 'CAMERA_V11_STEP1V7_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }
command -v flock >/dev/null 2>&1 || fail "flock missing"
exec 9>"$LOCK_FILE"
flock -n 9 || fail "another V11 Step1 V7 runtime holds $LOCK_FILE; stop it first"
[[ -x "$PY" ]] || fail "Python 3.10 environment missing: $PY"
[[ -n "${DISPLAY:-}" ]] || fail "DISPLAY is empty; independent EGL wall requires X11/XWayland"

for plugin in nvurisrcbin queue nvvideoconvert capsfilter nveglglessink rtspsrc; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing plugin: $plugin"
done

gst-inspect-1.0 nvurisrcbin 2>/dev/null | grep -q 'low-latency-mode' || \
  fail "nvurisrcbin low-latency-mode property missing"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Hold V4 DS100 baseline constant.
export V11_RTSP_TRANSPORT=tcp
export V11_RTSP_LATENCY_MS="${V11_RTSP_LATENCY_MS:-100}"
export V11_DROP_ON_LATENCY="${V11_DROP_ON_LATENCY:-1}"
export V11_EXTRA_SURFACES="${V11_EXTRA_SURFACES:-4}"
export V11_UDP_BUFFER_SIZE="${V11_UDP_BUFFER_SIZE:-8388608}"
export V11_RECONNECT_SEC="${V11_RECONNECT_SEC:-5}"
export V11_STARTUP_STAGGER_SEC="${V11_STARTUP_STAGGER_SEC:-0.40}"
export V11_STATS_INTERVAL_SEC="${V11_STATS_INTERVAL_SEC:-5}"
export V11_TILE_WIDTH="${V11_TILE_WIDTH:-640}"
export V11_TILE_HEIGHT="${V11_TILE_HEIGHT:-360}"
export V11_SCALE_INTERPOLATION="${V11_SCALE_INTERPOLATION:-4}"

# Only decoder policy changes: CAM-02 low latency ON; all others OFF.
export V11_LOWLAT_CAMERAS="${V11_LOWLAT_CAMERAS:-CAM-02}"

"$PY" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"V11 Step1 V7 requires Python 3.10, got {sys.version}")
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)
from services.camera_v11.step1_cam02_lowlat_v7 import V11Step1Cam02LowLatV7  # noqa: F401
print(f"CAMERA_V11_STEP1V7_IMPORT python={sys.version.split()[0]} gst={Gst.version_string()} runtime=OK")
PY

printf '%s\n' \
  "CAMERA_V11_STEP1V7_PREFLIGHT status=OK python=$PY display=${DISPLAY}" \
  "CAMERA_V11_STEP1V7_INVARIANT base=v4-ds100 mux=0 tiler=0 detector=0 tracker=0 latest_only=1 transport=tcp latency_ms=${V11_RTSP_LATENCY_MS}" \
  "CAMERA_V11_STEP1V7_AB lowlat_cameras=${V11_LOWLAT_CAMERAS} bframes_cam02=0"

exec "$PY" -u -m services.camera_v11.step1_cam02_lowlat_v7
