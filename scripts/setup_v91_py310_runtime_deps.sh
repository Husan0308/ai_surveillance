#!/usr/bin/env bash
# Deprecated filename retained for existing operator bookmarks. No Python 3.10
# or legacy inference environment is created. Use setup_runtime.sh directly.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
printf 'Deprecated setup alias; forwarding to scripts/setup_runtime.sh\n' >&2
exec bash "$ROOT/scripts/setup_runtime.sh" "$@"
