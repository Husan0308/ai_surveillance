#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Explicit setup only. No sudo, camera access, NVIDIA package installation, or
# implicit setup from the launcher. stdout is the resulting interpreter path.
exec /usr/bin/python3 -I "$ROOT/scripts/runtime_foundation.py" setup --root "$ROOT" "$@"
