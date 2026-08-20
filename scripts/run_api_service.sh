#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x .venv/bin/python ]]; then
  PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

ML_URL="${ML_SERVICE_URL:-http://127.0.0.1:8001}"
API_PORT_VALUE="${API_PORT:-8000}"

echo "API_SERVICE_START branch=$(git branch --show-current 2>/dev/null || true) head=$(git rev-parse --short=12 HEAD 2>/dev/null || true) ml=${ML_URL} port=${API_PORT_VALUE}"
exec "$PYTHON" -m services.api_service.app.main
