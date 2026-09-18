#!/usr/bin/env python3
"""Export YOLO26m detection to a DeepStream-friendly NMS-free ONNX artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


def _shape(value_info):
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(dim.dim_value)
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="yolo26m.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(".runtime/models/yolo26m"),
    )
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
        import onnx
    except ImportError as exc:
        print(
            "Missing dependency. Install with: "
            "python3 -m pip install -U ultralytics onnx",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            batch=args.batch,
            dynamic=True,
            simplify=True,
            nms=False,
            opset=17,
            device=0,
        )
    ).resolve()

    target = (args.out_dir / "yolo26m_nmsfree_b6_dynamic.onnx").resolve()
    if exported != target:
        shutil.copy2(exported, target)

    graph = onnx.load(str(target))
    onnx.checker.check_model(graph)

    inputs = [
        {"name": value.name, "shape": _shape(value)}
        for value in graph.graph.input
    ]
    outputs = [
        {"name": value.name, "shape": _shape(value)}
        for value in graph.graph.output
    ]

    if len(inputs) != 1:
        raise RuntimeError(f"Expected one image input, found {len(inputs)}")
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one detection output, found {len(outputs)}")

    out_shape = outputs[0]["shape"]
    # YOLO26 nms=False detect export is expected to expose rows of
    # [x1,y1,x2,y2,confidence,class_id], normally (N, 300, 6).
    if not out_shape or out_shape[-1] != 6:
        raise RuntimeError(
            f"Unexpected NMS-free output shape {out_shape}; expected last dimension 6"
        )

    report = {
        "status": "PASS",
        "model": args.model,
        "artifact": str(target),
        "sha256": sha256(target),
        "imgsz": args.imgsz,
        "batch_max": args.batch,
        "dynamic": True,
        "nms": False,
        "inputs": inputs,
        "outputs": outputs,
        "expected_output_semantics": [
            "x1",
            "y1",
            "x2",
            "y2",
            "confidence",
            "class_id",
        ],
    }
    report_path = args.out_dir / "export_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
