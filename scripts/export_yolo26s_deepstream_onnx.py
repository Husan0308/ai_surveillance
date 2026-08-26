#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a static batch-6 YOLO26s end-to-end ONNX for DeepStream nvinfer."
    )
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=672)
    args = parser.parse_args()

    if args.batch != 6:
        raise SystemExit("This six-camera DeepStream profile requires --batch 6")

    from ultralytics import YOLO

    root = Path(__file__).resolve().parents[1]
    model_arg = args.model
    local_model = root / model_arg
    model_spec = str(local_model) if local_model.is_file() else model_arg

    output_dir = root / "artifacts" / "yolo26s_deepstream"
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"yolo26s-{args.width}x{args.height}-b{args.batch}-e2e.onnx"

    print(
        "DEEPSTREAM_YOLO26_EXPORT "
        f"model={model_spec} batch={args.batch} input={args.width}x{args.height} "
        "end2end=1 dynamic=0",
        flush=True,
    )

    model = YOLO(model_spec)
    if str(getattr(model, "task", "") or "detect") != "detect":
        raise RuntimeError(f"Expected detect model, got task={getattr(model, 'task', None)!r}")

    exported = model.export(
        format="onnx",
        imgsz=(args.height, args.width),
        batch=args.batch,
        dynamic=False,
        simplify=True,
        end2end=True,
        device=0,
        verbose=False,
    )
    exported_path = Path(str(exported)).resolve()
    if not exported_path.is_file():
        raise RuntimeError(f"Ultralytics export did not produce a file: {exported_path}")

    if exported_path != target.resolve():
        shutil.copy2(exported_path, target)

    print(f"DEEPSTREAM_YOLO26_EXPORT status=OK onnx={target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
