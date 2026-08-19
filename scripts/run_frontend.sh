#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[FRONTEND] project_root=$ROOT"
python scripts/preflight_frontend.py

echo "[FRONTEND] starting PySide6 client"
exec python -m services.frontend.app.main
