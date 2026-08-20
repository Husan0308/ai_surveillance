#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export RF_HOME="${RF_HOME:-$PWD/.runtime/vision_v3/rfdetr}"
mkdir -p "$RF_HOME"

python -m pip install "rfdetr>=1.6.2,<2"

python - <<'PY'
import numpy as np
import torch
from rfdetr import RFDETRSmall

if not torch.cuda.is_available():
    raise SystemExit("RF-DETR setup failed: PyTorch CUDA unavailable")

print("RF-DETR setup: loading RFDETRSmall on", torch.cuda.get_device_name(0), flush=True)
model = RFDETRSmall(device="cuda:0")
frame = np.zeros((432, 768, 3), dtype=np.uint8)
with torch.inference_mode():
    detections = model.predict(frame, threshold=0.18)
count = len(getattr(detections, "xyxy", []))
print(f"VISION_V3_RFDETR_SETUP=PASS warmup_detections={count}", flush=True)
PY
