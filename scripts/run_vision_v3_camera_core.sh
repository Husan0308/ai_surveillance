#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Load the existing gitignored local credentials when present. Do not print them.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export PYTHONUNBUFFERED=1

if [[ -n "${DISPLAY:-}" && -z "${QT_QPA_PLATFORM:-}" ]]; then
  export QT_QPA_PLATFORM=xcb
fi

echo "VISION_V3_CAMERA_CORE branch=$(git branch --show-current 2>/dev/null || echo unknown) head=$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
python scripts/preflight_vision_v3_camera_core.py
exec python -m services.ml_service.vision_v3.camera_core
