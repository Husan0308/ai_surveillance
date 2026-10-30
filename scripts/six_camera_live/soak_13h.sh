#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$root/.runtime/full-stack-venv/bin/python" -B "$root/scripts/six_camera_live/soak_13h.py" "$@"
