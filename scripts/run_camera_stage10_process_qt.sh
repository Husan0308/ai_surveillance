#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HEAD_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"

echo "STAGE10_BUILD branch=${BRANCH} head=${HEAD_SHA} cameras=6"
echo "STAGE10_CONTRACT cameras=6 mux=1 batch=6 tiler=1 layout=2x3 convert=1 capsforce=NVMM-RGBA osd=1 sink=nveglglessink qt_parent=1 xid_cross_process=1 controller=spawn detector=0 tracker=0 analysis=0 rtsp=tcp latency=250ms"

export CAMERA_V2_RTSP_LATENCY_MS="${CAMERA_V2_RTSP_LATENCY_MS:-250}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

exec python -m services.camera_v2.stage10_process_qt
