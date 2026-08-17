#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="${CAMERA_V2_QWEN_HF_REPO:-Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M}"
HOST="${CAMERA_V2_QWEN_HOST:-127.0.0.1}"
PORT="${CAMERA_V2_QWEN_PORT:-8080}"
THREADS="${CAMERA_V2_QWEN_THREADS:-4}"
CTX="${CAMERA_V2_QWEN_CTX:-2048}"

# Qwen/llama.cpp warns that 1024 image tokens are important for *grounding*.
# Our task is not grounding/OCR/coordinate prediction: it is a tight two-person
# classification sheet. 1024 tokens made each request exceed ~20 s on GTX 1050 Ti,
# so use 512 dynamic-resolution image tokens by default. This keeps clothes/body
# cues while roughly halving vision work. Override to 768/1024 if desired.
IMAGE_MIN_TOKENS="${CAMERA_V2_QWEN_IMAGE_MIN_TOKENS:-512}"
IMAGE_MAX_TOKENS="${CAMERA_V2_QWEN_IMAGE_MAX_TOKENS:-512}"

# Keep a bounded transformer offload. The multimodal projector stays on CUDA by
# default because CPU mmproj was the dominant latency. Qwen is asynchronous, so
# the detector/tracker never waits for it.
GPU_LAYERS="${CAMERA_V2_QWEN_GPU_LAYERS:-8}"
GPU_DEVICE="${CAMERA_V2_QWEN_GPU_DEVICE:-0}"
MMPROJ_GPU="${CAMERA_V2_QWEN_MMPROJ_GPU:-1}"

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
  --reasoning off
  --alias qwen3-vl-reid
)

if [[ "$MMPROJ_GPU" != "1" ]]; then
  ARGS+=(--no-mmproj-offload)
fi

export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"

printf 'QWEN_REID_SERVER binary=%s mode=%s model=%s gpu_layers=%s gpu_device=%s ctx=%s image_tokens=%s..%s mmproj_gpu=%s reasoning=off\n' \
  "$LLAMA_BIN" "$MODE" "$MODEL_REPO" "$GPU_LAYERS" "$GPU_DEVICE" "$CTX" \
  "$IMAGE_MIN_TOKENS" "$IMAGE_MAX_TOKENS" "$MMPROJ_GPU"

if [[ "$MODE" == "app" ]]; then
  exec "$LLAMA_BIN" serve "${ARGS[@]}"
else
  exec "$LLAMA_BIN" "${ARGS[@]}"
fi
