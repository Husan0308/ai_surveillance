#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="${V11_PYTHON:-$ROOT/.venv-trt86/bin/python}"
LOCK_FILE="/tmp/ai_surveillance_camera_v11_step1_v2.lock"

fail() { printf 'CAMERA_V11_STEP1V2_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }
command -v flock >/dev/null 2>&1 || fail "flock missing"
exec 9>"$LOCK_FILE"
flock -n 9 || fail "another V11 Step1 V2 runtime holds $LOCK_FILE; stop it first"
[[ -x "$PY" ]] || fail "Python 3.10 environment missing: $PY"
[[ -n "${DISPLAY:-}" ]] || fail "DISPLAY is empty; nveglglessink requires an X display"

for plugin in nvurisrcbin queue nvstreammux nvmultistreamtiler nveglglessink rtspsrc; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing plugin: $plugin"
done

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

# Step1 V2 display-only policy. 40 ms is below the nominal 50 ms source period,
# so one slow phase cannot hold a whole batch for another full frame. The tiler
# cache keeps the wall stable when a batch is partial.
export V11_BATCH_TIMEOUT_US="${V11_BATCH_TIMEOUT_US:-40000}"
# DeepStream/NvBufSurfTransform: 4 = GPU Lanczos on dGPU.
export V11_MUX_INTERPOLATION="${V11_MUX_INTERPOLATION:-4}"

"$PY" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"V11 Step1 V2 requires Python 3.10, got {sys.version}")
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)
from services.camera_v11.pts_bridge import ensure_bridge
path = ensure_bridge()
from services.camera_v11.step1_display_v2 import V11Step1DisplayV2  # noqa: F401
print(
    f"CAMERA_V11_STEP1V2_IMPORT python={sys.version.split()[0]} "
    f"gst={Gst.version_string()} pts_bridge={path} runtime=OK"
)
PY

printf '%s\n' \
  "CAMERA_V11_STEP1V2_PREFLIGHT status=OK python=$PY single_owner=1 display=${DISPLAY}" \
  "CAMERA_V11_STEP1V2_INVARIANT old_step1_untouched=1 step0_ingest=frozen tracker=0 detector=0 osd=0 jpeg=0 latest_only=1" \
  "CAMERA_V11_STEP1V2_PROFILE purpose=display-freshness-and-clarity tile=640x360 wall=1920x720 timeout_us=${V11_BATCH_TIMEOUT_US} interpolation=${V11_MUX_INTERPOLATION}"

exec "$PY" -u -m services.camera_v11.step1_display_v2
