#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
LOCK_FILE="/tmp/ai_surveillance_camera_v2_gpu.lock"
PY="${V11_PYTHON:-$ROOT/.venv-trt86/bin/python}"

fail() { printf 'CAMERA_V11_STEP0_PREFLIGHT ERROR: %s\n' "$*" >&2; exit 1; }
command -v flock >/dev/null 2>&1 || fail "flock missing"
exec 9>"$LOCK_FILE"
flock -n 9 || fail "another camera runtime holds $LOCK_FILE; run cleanup first"
[[ -x "$PY" ]] || fail "Python 3.10 environment missing: $PY"

export V11_RTSP_TRANSPORT="${V11_RTSP_TRANSPORT:-tcp}"
export V11_RTSP_LATENCY_MS="${V11_RTSP_LATENCY_MS:-60}"
export V11_DROP_ON_LATENCY="${V11_DROP_ON_LATENCY:-1}"
export V11_EXTRA_SURFACES="${V11_EXTRA_SURFACES:-4}"
export V11_UDP_BUFFER_SIZE="${V11_UDP_BUFFER_SIZE:-8388608}"
export V11_RECONNECT_SEC="${V11_RECONNECT_SEC:-5}"
export V11_STARTUP_STAGGER_SEC="${V11_STARTUP_STAGGER_SEC:-0.40}"
export V11_STATS_INTERVAL_SEC="${V11_STATS_INTERVAL_SEC:-5}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Step0 intentionally has no TensorRT/CUDA compute, tracker, mux or display.
unset NVDS_ENABLE_LATENCY_MEASUREMENT || true
unset NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT || true
export QWEN_REID_ENABLED=0

for plugin in nvurisrcbin rtspsrc queue fakesink; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing plugin: $plugin"
done

"$PY" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"V11 Step0 requires Python 3.10, got {sys.version}")
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)
probe = Gst.ElementFactory.make("rtspsrc", "v11_probe")
tcp_ts = int(probe is not None and probe.find_property("tcp-timestamp") is not None)
from services.camera_v11.step0_ingest import V11Step0Ingest  # noqa: F401
print(
    f"CAMERA_V11_STEP0_IMPORT python={sys.version.split()[0]} "
    f"gst={Gst.version_string()} tcp_timestamp_property={tcp_ts} runtime=OK"
)
PY

printf '%s\n' \
  "CAMERA_V11_STEP0_PREFLIGHT status=OK python=$PY single_owner=1" \
  "CAMERA_V11_STEP0_PROFILE purpose=rtsp-nvdec-baseline transport=$V11_RTSP_TRANSPORT latency_ms=$V11_RTSP_LATENCY_MS extra_surfaces=$V11_EXTRA_SURFACES" \
  "CAMERA_V11_STEP0_INVARIANT no_mux=1 no_tracker=1 no_detector=1 no_display=1 independent_camera_pipelines=1"

exec "$PY" -u -m services.camera_v11.step0_ingest
