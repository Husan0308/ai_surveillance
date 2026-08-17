#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="${CAMERA_V2_QWEN_HF_REPO:-Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M}"
HOST="${CAMERA_V2_QWEN_HOST:-127.0.0.1}"
PORT="${CAMERA_V2_QWEN_PORT:-8080}"
THREADS="${CAMERA_V2_QWEN_THREADS:-4}"
CTX="${CAMERA_V2_QWEN_CTX:-4096}"
IMAGE_MIN_TOKENS="${CAMERA_V2_QWEN_IMAGE_MIN_TOKENS:-1024}"
IMAGE_MAX_TOKENS="${CAMERA_V2_QWEN_IMAGE_MAX_TOKENS:-1024}"

# GTX 1050 Ti has only 4 GB VRAM and the live YOLO/NvDCF/DeepStream pipeline
# already owns part of it. Keep transformer offload bounded; the vision projector
# is now GPU-offloaded by default because visual encoding was the dominant ~25 s
# latency in the previous CPU-mmproj setup.
GPU_LAYERS="${CAMERA_V2_QWEN_GPU_LAYERS:-8}"
GPU_DEVICE="${CAMERA_V2_QWEN_GPU_DEVICE:-0}"
MMPROJ_GPU="${CAMERA_V2_QWEN_MMPROJ_GPU:-1}"

# The llama.app installer currently places the unified CLI here. It may not be
# visible in PATH until a new shell is opened, so probe the installed location
# explicitly before falling back to PATH binaries.
if [[ -x "${HOME}/.llama-app/llama" ]]; then
  LLAMA_BIN="${HOME}/.llama-app/llama"
  MODE="app"
elif command -v llama >/dev/null 2>&1; then
  LLAMA_BIN="$(command -v llama)"
  MODE="app"
elif command -v llama-server >/dev/null 2>&1; then
  LLAMA_BIN="$(command -v llama-server)"
  MODE="server"
else
  echo "llama.cpp server not found." >&2
  echo "Expected ~/.llama-app/llama after: curl -LsSf https://llama.app/install.sh | sh" >&2
  exit 2
fi

# Avoid the confusing bind error when an older Qwen server is still alive.
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -qE "[.:]${PORT}[[:space:]]"; then
  echo "QWEN_REID_SERVER port ${PORT} is already in use." >&2
  echo "Stop the old server first (Ctrl+C), or run: fuser -k ${PORT}/tcp" >&2
  exit 3
fi

ARGS=(
  -hf "$MODEL_REPO"
  --host "$HOST"
  --port "$PORT"
  --ctx-size "$CTX"
  --threads "$THREADS"
  --threads-batch "$THREADS"
  --parallel 1
  --n-gpu-layers "$GPU_LAYERS"
  --image-min-tokens "$IMAGE_MIN_TOKENS"
  --image-max-tokens "$IMAGE_MAX_TOKENS"
  --alias qwen3-vl-reid
)

# llama.cpp normally offloads the multimodal projector to GPU. On this machine
# that is desirable because Qwen is only an asynchronous verifier and the current
# VRAM trace leaves enough headroom. Set CAMERA_V2_QWEN_MMPROJ_GPU=0 to fall back
# to CPU if the full surveillance runtime ever hits CUDA OOM.
if [[ "$MMPROJ_GPU" != "1" ]]; then
  ARGS+=(--no-mmproj-offload)
fi

# Restrict Qwen to the requested CUDA device without changing the parent shell.
export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"

printf 'QWEN_REID_SERVER binary=%s mode=%s model=%s gpu_layers=%s gpu_device=%s ctx=%s image_tokens=%s..%s mmproj_gpu=%s\n' \
  "$LLAMA_BIN" "$MODE" "$MODEL_REPO" "$GPU_LAYERS" "$GPU_DEVICE" "$CTX" \
  "$IMAGE_MIN_TOKENS" "$IMAGE_MAX_TOKENS" "$MMPROJ_GPU"

if [[ "$MODE" == "app" ]]; then
  exec "$LLAMA_BIN" serve "${ARGS[@]}"
else
  exec "$LLAMA_BIN" "${ARGS[@]}"
fi
