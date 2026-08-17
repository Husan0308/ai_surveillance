#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="${CAMERA_V2_QWEN_HF_REPO:-Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M}"
HOST="${CAMERA_V2_QWEN_HOST:-127.0.0.1}"
PORT="${CAMERA_V2_QWEN_PORT:-8080}"
THREADS="${CAMERA_V2_QWEN_THREADS:-4}"
CTX="${CAMERA_V2_QWEN_CTX:-2048}"
IMAGE_TOKENS="${CAMERA_V2_QWEN_IMAGE_TOKENS:-256}"

# GTX 1050 Ti has only 4 GB VRAM and the live YOLO/NvDCF/DeepStream pipeline
# already owns most of it. Offload a bounded number of LLM layers to CUDA for a
# useful speedup without letting Qwen evict the surveillance hot path.
GPU_LAYERS="${CAMERA_V2_QWEN_GPU_LAYERS:-8}"
GPU_DEVICE="${CAMERA_V2_QWEN_GPU_DEVICE:-0}"

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

ARGS=(
  -hf "$MODEL_REPO"
  --host "$HOST"
  --port "$PORT"
  --ctx-size "$CTX"
  --threads "$THREADS"
  --threads-batch "$THREADS"
  --parallel 1
  --n-gpu-layers "$GPU_LAYERS"
  --image-max-tokens "$IMAGE_TOKENS"
  --alias qwen3-vl-reid
)

# Keep the large multimodal projector on CPU. Current llama.cpp can offload it,
# but on a 4 GB card shared with YOLO/NvDCF that creates avoidable VRAM pressure.
# The transformer layers still use CUDA, which is the useful speedup here.
if [[ "${CAMERA_V2_QWEN_MMPROJ_GPU:-0}" != "1" ]]; then
  ARGS+=(--no-mmproj-offload)
fi

# Restrict Qwen to the requested CUDA device without changing the parent shell.
export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"

printf 'QWEN_REID_SERVER binary=%s mode=%s model=%s gpu_layers=%s gpu_device=%s ctx=%s mmproj_gpu=%s\n' \
  "$LLAMA_BIN" "$MODE" "$MODEL_REPO" "$GPU_LAYERS" "$GPU_DEVICE" "$CTX" \
  "${CAMERA_V2_QWEN_MMPROJ_GPU:-0}"

if [[ "$MODE" == "app" ]]; then
  exec "$LLAMA_BIN" serve "${ARGS[@]}"
else
  exec "$LLAMA_BIN" "${ARGS[@]}"
fi
