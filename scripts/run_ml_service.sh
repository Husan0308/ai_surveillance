#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "ML_SERVICE_START branch=$(git branch --show-current 2>/dev/null || echo unknown) head=$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
python scripts/preflight_ml_service.py

# Person detection is isolated inside ml_service. A CUDA/model compatibility
# failure must not prevent the six camera pipelines, health API or MJPEG from
# starting. The detector child will expose its own error state through /health.
if ! python scripts/preflight_person_detection.py; then
  echo "PERSON_DETECT_PREFLIGHT_WARNING detector unavailable; starting camera service anyway" >&2
fi

exec python -m services.ml_service.app.main
