#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Keep the camera baseline deterministic. RF-DETR/NvDCF/ReID are intentionally
# absent from this launcher until the six-camera core passes soak testing.
export PYTHONUNBUFFERED=1

if [[ -n "${DISPLAY:-}" && -z "${QT_QPA_PLATFORM:-}" ]]; then
  export QT_QPA_PLATFORM=xcb
fi

echo "VISION_V3_CAMERA_CORE branch=$(git branch --show-current 2>/dev/null || echo unknown) head=$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
python scripts/preflight_vision_v3_camera_core.py
exec python -m services.ml_service.vision_v3.camera_core
