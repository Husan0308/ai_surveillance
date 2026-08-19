#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "ML_SERVICE_START branch=$(git branch --show-current 2>/dev/null || echo unknown) head=$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
python scripts/preflight_ml_service.py
python scripts/preflight_person_detection.py
exec python -m services.ml_service.app.main
