#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export PYTHONUNBUFFERED=1
export RF_HOME="${RF_HOME:-$PWD/.runtime/vision_v3/rfdetr}"
mkdir -p "$RF_HOME"

if [[ -n "${DISPLAY:-}" && -z "${QT_QPA_PLATFORM:-}" ]]; then
  export QT_QPA_PLATFORM=xcb
fi

echo "VISION_V3_RFDETR branch=$(git branch --show-current 2>/dev/null || echo unknown) head=$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown) profile=agent-rfdetr-s-core-final"
python scripts/preflight_vision_v3_camera_core.py
python scripts/preflight_vision_v3_rfdetr.py
exec python -m services.ml_service.vision_v3.rfdetr_proven_entry
