#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "ERROR: activate .venv first: source .venv/bin/activate" >&2
  exit 2
fi

# Do not inherit experimental camera knobs from an interactive shell. The Python
# entrypoint installs the complete deterministic profile before camera modules
# are imported.
while IFS='=' read -r name _; do
  case "$name" in
    CAMERA_V2_*|QWEN_*|NVDS_*) unset "$name" || true ;;
  esac
done < <(env)

unset CUDA_VISIBLE_DEVICES || true

exec python -u -m services.camera_v2.cam01_lowlat_gpu
