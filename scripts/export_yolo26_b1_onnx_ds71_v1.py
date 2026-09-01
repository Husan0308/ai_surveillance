#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export YOLO26 as fixed batch-1 672x384 ONNX with NMS for DeepStream 7.1"
    )
    parser.add_argument("--model", required=True, help="Local .pt path or canonical Ultralytics name")
    parser.add_argument(
        "--output",
        default="artifacts/yolo26s_ds71/yolo26s-672x384-b1-nms.onnx",
    )
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(
            f"V11_DS_YOLO_ONNX RESULT=FAIL reason=ultralytics_import "
            f"error={type(exc).__name__}:{exc}"
        )

    model_arg = str(args.model)
    candidate = Path(model_arg).expanduser()
    if candidate.is_file():
        model_spec = str(candidate.resolve())
    elif args.allow_download and candidate.name == model_arg and model_arg.endswith(".pt"):
        model_spec = model_arg
    else:
        raise SystemExit(f"V11_DS_YOLO_ONNX RESULT=FAIL reason=model_missing path={candidate}")

    output = Path(args.output).expanduser().resolve()
    print(
        "V11_DS_YOLO_ONNX_EXPORT "
        f"model={model_spec} batch=1 imgsz=384x672 dynamic=0 nms=1 fp32=1 opset=17",
        flush=True,
    )

    model = YOLO(model_spec)
    exported = model.export(
        format="onnx",
        imgsz=(384, 672),
        batch=1,
        dynamic=False,
        half=False,
        simplify=False,
        nms=True,
        opset=17,
    )
    produced = Path(str(exported)).expanduser().resolve()
    if not produced.is_file():
        raise SystemExit(
            f"V11_DS_YOLO_ONNX RESULT=FAIL reason=export_missing path={produced}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    if produced != output:
        shutil.copy2(produced, output)

    print(
        f"V11_DS_YOLO_ONNX RESULT=PASS output={output} bytes={output.stat().st_size}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
