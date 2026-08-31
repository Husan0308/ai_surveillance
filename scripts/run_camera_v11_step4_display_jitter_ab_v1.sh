#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="${V11_PYTHON:-$ROOT/.venv-trt86/bin/python}"

[[ -x "$PY" ]] || { echo "CAMERA_V11_STEP4_DISPLAY_AB_PREFLIGHT result=FAIL reason=python_missing path=$PY" >&2; exit 1; }
[[ -n "${DISPLAY:-}" ]] || { echo "CAMERA_V11_STEP4_DISPLAY_AB_PREFLIGHT result=FAIL reason=DISPLAY_empty" >&2; exit 1; }

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Hold all frozen V7 display variables constant. Only per-camera RTP transport is
# varied by step4_display_jitter_ab_v1.py.
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
export V11_LOWLAT_CAMERAS="${V11_LOWLAT_CAMERAS:-CAM-02}"
export V11_STEP4_UDP_CAMERAS="${V11_STEP4_UDP_CAMERAS:-CAM-01,CAM-03}"

printf 'CAMERA_V11_STEP4_DISPLAY_AB_PREFLIGHT result=PASS udp_cameras=%s latency_ms=%s lowlat_cameras=%s\n' \
  "$V11_STEP4_UDP_CAMERAS" "$V11_RTSP_LATENCY_MS" "$V11_LOWLAT_CAMERAS"

exec "$PY" -u -m services.camera_v11.step4_display_jitter_ab_v1
