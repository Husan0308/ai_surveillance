#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 4.4a: prepare one exact RF-DETR input tensor in the PyTorch env for ONNX/TensorRT parity."
    )
    p.add_argument("image", nargs="?", type=Path)
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=448)
    p.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/rfdetr_step4/parity")
    )
    return p.parse_args()


def _pick_image(explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            Path("/tmp/rfdetr_cam03_frozen.jpg"),
            Path("artifacts/rfdetr_step1/cam_03_input.jpg"),
            Path("artifacts/rfdetr_step2_sequence/frames/frame_001.jpg"),
        ]
    )
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise SystemExit(
        "STEP4_4_PREP_FAIL image_not_found provide_image_or_keep_cam03_frozen_frame"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    args = _args()
    if args.width <= 0 or args.height <= 0 or args.width % 32 or args.height % 32:
        raise SystemExit(
            f"STEP4_4_PREP_FAIL invalid_shape={args.width}x{args.height}"
        )

    import torch
    import torchvision.transforms.functional as F

    image_path = _pick_image(args.image)
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    source_w, source_h = image.size

    # Exact RF-DETR predict/export preprocessing contract:
    # to_tensor -> bilinear resize(antialias=False) -> ImageNet normalize.
    with torch.no_grad():
        tensor = F.to_tensor(image)
        tensor = F.resize(tensor, [int(args.height), int(args.width)], antialias=False)
        tensor = F.normalize(
            tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        array = tensor.unsqueeze(0).contiguous().cpu().numpy().astype(np.float32, copy=False)

    expected_shape = (1, 3, int(args.height), int(args.width))
    if array.shape != expected_shape:
        raise SystemExit(
            f"STEP4_4_PREP_FAIL tensor_shape expected={expected_shape} got={array.shape}"
        )
    if not np.isfinite(array).all():
        raise SystemExit("STEP4_4_PREP_FAIL tensor_nonfinite")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.output_dir / "input_800x448.npy"
    np.save(input_path, array, allow_pickle=False)

    image_bytes = image_path.read_bytes()
    tensor_bytes = array.tobytes(order="C")
    manifest = {
        "stage": "4.4a",
        "source_image": str(image_path),
        "source_size_wh": [source_w, source_h],
        "input_path": str(input_path),
        "input_shape": list(array.shape),
        "input_dtype": str(array.dtype),
        "preprocess": {
            "resize": "torchvision bilinear antialias=False",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "source_sha256": _sha256_bytes(image_bytes),
        "input_sha256": _sha256_bytes(tensor_bytes),
        "input_stats": {
            "min": float(array.min()),
            "max": float(array.max()),
            "mean": float(array.mean()),
            "std": float(array.std()),
        },
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
    }
    manifest_path = args.output_dir / "parity_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        "STEP4_4_PREP_RESULT "
        f"image={image_path} source={source_w}x{source_h} "
        f"input={input_path} shape={list(array.shape)} "
        f"sha256={manifest['input_sha256'][:16]} "
        f"range=[{array.min():.4f},{array.max():.4f}]",
        flush=True,
    )
    print(f"STEP4_4_PREP_JSON={manifest_path}", flush=True)
    print("STEP4_4_PREP_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
