#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXPECTED_BRANCH="rebuild/service-architecture-v11-ui-realtime-cameras-v1-20260901"
CURRENT_BRANCH="$(git branch --show-current 2>/dev/null || true)"

printf 'V11_UI_BOOTSTRAP branch=%s expected=%s\n' "${CURRENT_BRANCH:-DETACHED}" "$EXPECTED_BRANCH"

if [[ ! -f tests/test_frontend_realtime_models.py ]]; then
  echo "V11_UI_BOOTSTRAP RESULT=FAIL reason=realtime_branch_files_missing"
  echo "Expected tests/test_frontend_realtime_models.py. Fetch/switch to $EXPECTED_BRANCH or use a clean git worktree."
  exit 2
fi

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PY="${VIRTUAL_ENV}/bin/python"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="$(command -v python3)"
fi

if [[ -z "$PY" || ! -x "$PY" ]]; then
  echo "V11_UI_BOOTSTRAP RESULT=FAIL reason=python_missing"
  exit 3
fi

printf 'V11_UI_BOOTSTRAP python=%s\n' "$PY"

"$PY" -m pip install -U \
  -r services/api_service/requirements.txt \
  -r services/ml_service/requirements.txt \
  -r services/frontend/requirements.txt \
  pytest

"$PY" - <<'PY'
import fastapi, httpx, uvicorn
from PySide6.QtCore import QT_VERSION_STR
from PySide6.QtWebSockets import QWebSocket
print("V11_UI_IMPORTS_OK", "fastapi", fastapi.__version__, "httpx", httpx.__version__, "uvicorn", uvicorn.__version__, "qt", QT_VERSION_STR, "qwebsocket", QWebSocket.__name__)
PY

"$PY" -m pytest -q tests/test_frontend_realtime_models.py

echo "V11_UI_BOOTSTRAP RESULT=PASS"
