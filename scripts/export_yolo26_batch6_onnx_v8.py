#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLO26 person detector as batch-6 ONNX")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(f"V8_ONNX FAIL ultralytics import: {type(exc).__name__}: {exc}")

    model_arg = args.model
    candidate = Path(model_arg).expanduser()
    if candidate.is_file():
        model_spec = str(candidate.resolve())
    elif args.allow_download and candidate.name == model_arg and model_arg.endswith(".pt"):
        # Ultralytics resolves/downloads its canonical model weight by name.
        model_spec = model_arg
        print(f"V8_ONNX_DOWNLOAD model={model_spec} source=ultralytics-canonical", flush=True)
    else:
        raise SystemExit(f"V8_ONNX FAIL model missing: {candidate}")

    output_path = Path(args.output).expanduser().resolve()
    print(
        f"V8_ONNX_EXPORT model={model_spec} batch=6 imgsz=384x672 dynamic=0 nms=1 fp32=1",
        flush=True,
    )
    model = YOLO(model_spec)
    exported = model.export(
        format="onnx",
        imgsz=(384, 672),
        batch=6,
        dynamic=False,
        half=False,
        simplify=False,
        nms=True,
        opset=17,
    )
    produced = Path(str(exported)).expanduser().resolve()
    if not produced.is_file():
        raise SystemExit(f"V8_ONNX FAIL export path not found: {produced}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if produced != output_path:
        shutil.copy2(produced, output_path)
    print(f"V8_ONNX PASS output={output_path} bytes={output_path.stat().st_size}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
