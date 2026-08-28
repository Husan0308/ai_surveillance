#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="${V11_PYTHON:-$ROOT/.venv-trt86/bin/python}"
LOCK_FILE="/tmp/ai_surveillance_camera_v11_step1_v20.lock"

fail() {
  printf 'CAMERA_V11_STEP1V20_PREFLIGHT ERROR: %s\n' "$*" >&2
  exit 1
}
command -v flock >/dev/null 2>&1 || fail "flock missing"
exec 9>"$LOCK_FILE"
flock -n 9 || fail "another V11 Step1 V20 runtime holds $LOCK_FILE"
[[ -x "$PY" ]] || fail "Python environment missing: $PY"
[[ -n "${DISPLAY:-}" ]] || fail "DISPLAY is empty"

for plugin in nvurisrcbin nvv4l2decoder queue nvvideoconvert capsfilter nveglglessink rtspsrc; do
  gst-inspect-1.0 "$plugin" >/dev/null 2>&1 || fail "missing plugin: $plugin"
done

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export V11_RTSP_TRANSPORT=tcp
export V11_RTSP_LATENCY_MS=100
export V11_DROP_ON_LATENCY=1
export V11_EXTRA_SURFACES=4
export V11_UDP_BUFFER_SIZE=8388608
export V11_RECONNECT_SEC=5
export V11_STARTUP_STAGGER_SEC=0.40
export V11_STATS_INTERVAL_SEC=5
export V11_TILE_WIDTH=640
export V11_TILE_HEIGHT=360
export V11_SCALE_INTERPOLATION=4
export V11_LOWLAT_CAMERAS=CAM-02

printf '%s\n' \
  "CAMERA_V11_STEP1V7_PREFLIGHT status=OK python=$PY display=${DISPLAY} lowlat_target=nvv4l2decoder" \
  "CAMERA_V11_STEP1V7_INVARIANT base=v4-ds100 mux=0 tiler=0 detector=0 tracker=0 bounded_queue=1 transport=tcp latency_ms=100" \
  "CAMERA_V11_STEP1V7_AB lowlat_cameras=CAM-02 bframes_cam02=0"
exec "$PY" -u -m services.camera_v11.step1_burst_backpressure_v20
