from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "models" / "yolo26m_b6_416x736.onnx"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export YOLO26 to a static batch=6 ONNX for DeepStream nvinfer."
    )
    parser.add_argument("--model", default="yolo26m.pt")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    exported = model.export(
        format="onnx",
        imgsz=(416, 736),
        batch=6,
        dynamic=False,
        simplify=True,
        nms=False,
        device=args.device,
    )
    exported_path = Path(str(exported)).resolve()
    if exported_path != output:
        shutil.copy2(exported_path, output)

    print(f"BATCH6_ONNX={output}")
    print("shape: batch=6, input=416x736, YOLO26 end-to-end output")
    print("next: python -m services.frontend.core_v1.main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
