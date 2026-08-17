#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
MODEL_DIR="$ROOT/.runtime/kpr"
MODEL_FILE="kpr_dancetrack_sportsmot_posetrack21_occludedduke_market_split0.pth.tar"
mkdir -p "$MODEL_DIR"

echo "[KPR] Checking PyTorch..."
"$PYTHON" - <<'PY'
import torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY

echo "[KPR] Checking the TrackLab-compatible KPR Torchreid fork..."
if ! "$PYTHON" - <<'PY' >/dev/null 2>&1
from torchreid.tools.feature_extractor import KPRFeatureExtractor
from torchreid.metrics.distance import compute_distance_matrix_using_bp_features
PY
then
  # This is the installation path documented by the current official TrackLab
  # project for its KPReID/BPBReID modules. It does not replace the project's
  # existing PyTorch package.
  "$PYTHON" -m pip install --upgrade \
    "torchreid @ git+https://github.com/victorjoos/keypoint_promptable_reidentification.git"
fi

"$PYTHON" -m pip install --upgrade huggingface-hub

echo "[KPR] Downloading the official multi-dataset KPR checkpoint..."
"$PYTHON" - <<PY
from pathlib import Path
from huggingface_hub import hf_hub_download
root = Path(r"$MODEL_DIR")
path = hf_hub_download(
    repo_id="trackinglaboratory/keypoint_promptable_reid",
    filename="$MODEL_FILE",
    local_dir=str(root),
)
print(f"checkpoint={path}")
print(f"size_mb={Path(path).stat().st_size / (1024**2):.1f}")
PY

echo "[KPR] Setup complete. Run: python scripts/preflight_kpr_reid.py"
