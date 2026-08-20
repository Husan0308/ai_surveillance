#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HEAD_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"

echo "STAGE21_BUILD branch=${BRANCH} head=${HEAD_SHA} cameras=6"
echo "STAGE21_CONTRACT cameras=6 mux=1 batch=6 tee=1 display_tiler=2x3 display_interp=4 wall_caps=NVMM-only wall_geometry=0 wall_par=0 wall_queue=0 convert=1 capsforce=NVMM-RGBA osd=1 sink=nveglglessink analysis=stage12-proven detector=0 gate=0 metadata=0 tracker=0 qt_parent=1 xid_cross_process=1 controller=spawn rtsp=tcp latency=250ms"

export CAMERA_V2_RTSP_LATENCY_MS="${CAMERA_V2_RTSP_LATENCY_MS:-250}"

exec python -m services.camera_v2.stage21_wall_caps_nvmm_only
