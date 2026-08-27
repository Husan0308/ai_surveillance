#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="${V11_PYTHON:-$ROOT/.venv-trt86/bin/python}"
LOCK_FILE="/tmp/ai_surveillance_camera_v11_step1_v4.lock"

fail() { printf 'CAMERA_V11_STEP1V4_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }

command -v flock >/dev/null 2>&1 || fail "flock missing"
exec 9>"$LOCK_FILE"
flock -n 9 || fail "another V11 Step1 V4 runtime holds $LOCK_FILE; stop it first"

[[ -x "$PY" ]] || fail "Python 3.10 environment missing: $PY"
[[ -n "${DISPLAY:-}" ]] || fail "DISPLAY is empty; independent EGL wall requires X11/XWayland"

for plugin in nvurisrcbin queue nvvideoconvert capsfilter nveglglessink rtspsrc; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing plugin: $plugin"
done
ldconfig -p 2>/dev/null | grep -q 'libX11\.so\.6' || fail "libX11.so.6 missing"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Frozen Step0 ingest policy.
export V11_RTSP_TRANSPORT="${V11_RTSP_TRANSPORT:-tcp}"
export V11_RTSP_LATENCY_MS="${V11_RTSP_LATENCY_MS:-60}"
export V11_DROP_ON_LATENCY="${V11_DROP_ON_LATENCY:-1}"
export V11_EXTRA_SURFACES="${V11_EXTRA_SURFACES:-4}"
export V11_UDP_BUFFER_SIZE="${V11_UDP_BUFFER_SIZE:-8388608}"
export V11_RECONNECT_SEC="${V11_RECONNECT_SEC:-5}"
export V11_STARTUP_STAGGER_SEC="${V11_STARTUP_STAGGER_SEC:-0.40}"
export V11_STATS_INTERVAL_SEC="${V11_STATS_INTERVAL_SEC:-5}"

# V4 independent display policy.
export V11_TILE_WIDTH="${V11_TILE_WIDTH:-640}"
export V11_TILE_HEIGHT="${V11_TILE_HEIGHT:-360}"
# DeepStream nvvideoconvert on dGPU: 4 = GPU Lanczos.
export V11_SCALE_INTERPOLATION="${V11_SCALE_INTERPOLATION:-4}"

"$PY" - <<'PY'
import ctypes
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"V11 Step1 V4 requires Python 3.10, got {sys.version}")
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo
Gst.init(None)
ctypes.CDLL("libX11.so.6")
required = ("nvurisrcbin", "queue", "nvvideoconvert", "capsfilter", "nveglglessink", "rtspsrc")
missing = [name for name in required if Gst.ElementFactory.find(name) is None]
if missing:
    raise SystemExit("missing plugins: " + ",".join(missing))
from services.camera_v11.step1_independent_egl_v4 import V11Step1IndependentEglV4  # noqa: F401
print(
    f"CAMERA_V11_STEP1V4_IMPORT python={sys.version.split()[0]} "
    f"gst={Gst.version_string()} gstvideo=OK x11=OK runtime=OK"
)
PY

printf '%s\n' \
  "CAMERA_V11_STEP1V4_PREFLIGHT status=OK python=$PY single_owner=1 display=${DISPLAY}" \
  "CAMERA_V11_STEP1V4_INVARIANT step0_ingest=frozen independent_pipelines=6 mux=0 tiler=0 tracker=0 detector=0 osd=0 jpeg=0 latest_only=1" \
  "CAMERA_V11_STEP1V4_PROFILE purpose=independent-display-freshness tile=${V11_TILE_WIDTH}x${V11_TILE_HEIGHT} wall=$((V11_TILE_WIDTH*3))x$((V11_TILE_HEIGHT*2)) interpolation=${V11_SCALE_INTERPOLATION}"

exec "$PY" -u -m services.camera_v11.step1_independent_egl_v4
