#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="${CAMERA_V2_QWEN_HF_REPO:-Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M}"
HOST="${CAMERA_V2_QWEN_HOST:-127.0.0.1}"
PORT="${CAMERA_V2_QWEN_PORT:-8080}"
THREADS="${CAMERA_V2_QWEN_THREADS:-4}"
CTX="${CAMERA_V2_QWEN_CTX:-2048}"
IMAGE_TOKENS="${CAMERA_V2_QWEN_IMAGE_TOKENS:-256}"

if command -v llama-server >/dev/null 2>&1; then
  BIN="$(command -v llama-server)"
elif command -v llama >/dev/null 2>&1; then
  # New llama.cpp installer exposes `llama`; its `serve` subcommand accepts the
  # same model repository form. Keep the explicit llama-server path preferred.
  exec llama serve \
    -hf "$MODEL_REPO" \
    --host "$HOST" \
    --port "$PORT" \
    --ctx-size "$CTX" \
    --threads "$THREADS" \
    --threads-batch "$THREADS" \
    --parallel 1 \
    --n-gpu-layers 0 \
    --no-mmproj-offload \
    --image-max-tokens "$IMAGE_TOKENS" \
    --alias qwen3-vl-reid
else
  echo "llama.cpp server not found. Install current llama.cpp first." >&2
  echo "Official installer: curl -LsSf https://llama.app/install.sh | sh" >&2
  exit 2
fi

exec "$BIN" \
  -hf "$MODEL_REPO" \
  --host "$HOST" \
  --port "$PORT" \
  --ctx-size "$CTX" \
  --threads "$THREADS" \
  --threads-batch "$THREADS" \
  --parallel 1 \
  --n-gpu-layers 0 \
  --no-mmproj-offload \
  --image-max-tokens "$IMAGE_TOKENS" \
  --alias qwen3-vl-reid
