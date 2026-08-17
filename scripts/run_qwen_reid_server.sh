#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="${CAMERA_V2_QWEN_HF_REPO:-Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M}"
HOST="${CAMERA_V2_QWEN_HOST:-127.0.0.1}"
PORT="${CAMERA_V2_QWEN_PORT:-8080}"
THREADS="${CAMERA_V2_QWEN_THREADS:-4}"

# The verifier sends two person montages in one request. Qwen3-VL/llama.cpp warns
# that Qwen-VL needs at least 1024 image tokens for reliable visual reasoning.
# 4096 context leaves room for 2x1024 image tokens + prompt + short JSON output.
CTX="${CAMERA_V2_QWEN_CTX:-4096}"
IMAGE_MIN_TOKENS="${CAMERA_V2_QWEN_IMAGE_MIN_TOKENS:-1024}"
IMAGE_MAX_TOKENS="${CAMERA_V2_QWEN_IMAGE_MAX_TOKENS:-1024}"

# GTX 1050 Ti has only 4 GB VRAM and the live YOLO/NvDCF/DeepStream pipeline
# already uses it. A bounded transformer offload gives useful speedup without
# handing the whole card to Qwen. Override to 12 after confirming free VRAM.
GPU_LAYERS="${CAMERA_V2_QWEN_GPU_LAYERS:-8}"
GPU_DEVICE="${CAMERA_V2_QWEN_GPU_DEVICE:-0}"

# The llama.app installer places the unified CLI here and it may not be in PATH
# until a new shell is opened.
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

# Do not produce the cryptic 'couldn't bind' error when an older Qwen server is
# still alive. Changing GPU layer count requires restarting that existing server.
if command -v curl >/dev/null 2>&1; then
  if curl -fsS --max-time 1 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "QWEN_REID_SERVER already running on http://${HOST}:${PORT}"
    echo "To change GPU layers, stop the old server (Ctrl-C in its terminal) and run this script again."
    echo "Current requested gpu_layers=${GPU_LAYERS}; a running process cannot be reconfigured in-place."
    exit 0
  fi
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

# Keep the multimodal projector on CPU by default. On this 4 GB Pascal card the
# projector plus YOLO/NvDCF can push VRAM over the edge. The LLM transformer layers
# still use CUDA. CAMERA_V2_QWEN_MMPROJ_GPU=1 is an explicit opt-in experiment.
if [[ "${CAMERA_V2_QWEN_MMPROJ_GPU:-0}" != "1" ]]; then
  ARGS+=(--no-mmproj-offload)
fi

export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"

printf 'QWEN_REID_SERVER binary=%s mode=%s model=%s gpu_layers=%s gpu_device=%s ctx=%s image_tokens=%s..%s mmproj_gpu=%s\n' \
  "$LLAMA_BIN" "$MODE" "$MODEL_REPO" "$GPU_LAYERS" "$GPU_DEVICE" "$CTX" \
  "$IMAGE_MIN_TOKENS" "$IMAGE_MAX_TOKENS" "${CAMERA_V2_QWEN_MMPROJ_GPU:-0}"

if [[ "$MODE" == "app" ]]; then
  exec "$LLAMA_BIN" serve "${ARGS[@]}"
else
  exec "$LLAMA_BIN" "${ARGS[@]}"
fi
