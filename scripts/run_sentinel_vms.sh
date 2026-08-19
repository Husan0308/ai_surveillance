#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HEAD_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
echo "SENTINEL_BUILD branch=${BRANCH} head=${HEAD_SHA} expected_ui=2026.08.19-r6"

# UI/source contract first: stale dialog fields, page bleed, fullscreen and live
# metric wiring must be valid before opening RTSP sources.
python scripts/preflight_sentinel_ui.py

# Core-only production preflight. This compiles/checks only the native metadata
# bridge and camera-space heatmap filter used by DeepStream/NvDCF. No pose model,
# ReID embedder, Qwen verifier, or face stack is loaded here.
python scripts/preflight_camera_v2_core.py

exec python -m services.camera_v2.monitor_ui
