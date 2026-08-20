#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
HEAD_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"

echo "STAGE11_BUILD branch=${BRANCH} head=${HEAD_SHA} cameras=6"
echo "STAGE11_CONTRACT cameras=6 mux=1 batch=6 tee=1 display_queue=1 analysis_queue=1 analysis_sink=fakesink tiler=1 convert=1 capsforce=NVMM-RGBA osd=1 sink=nveglglessink qt_parent=1 xid_cross_process=1 controller=spawn detector=0 tracker=0 appsink=0 rtsp=tcp latency=250ms"

export CAMERA_V2_RTSP_LATENCY_MS="${CAMERA_V2_RTSP_LATENCY_MS:-250}"
exec python -m services.camera_v2.stage11_process_tee
