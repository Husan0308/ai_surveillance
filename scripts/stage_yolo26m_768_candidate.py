#!/usr/bin/env python3
"""Stage a 768 diagnostic ONNX from the F5R dynamic-graph export and frozen weights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".runtime/freeze/F5R-six-camera-yolo-resolution-20261003"
SOURCE_ONNX = STAGE / "models/960/yolo26m_raw_otm.onnx"
SOURCE_WEIGHTS = STAGE / "models/960/yolo26m_raw_otm.pt"
SOURCE_MANIFEST = STAGE / "models/960/export_960.json"
PACKAGES = STAGE / "export-packages"
OUTPUT = ROOT / ".runtime/f5-cam05-safety-20261007-r1/yolo768-candidate-r2"
MODEL_DIR = OUTPUT / "models"
ONNX_OUT = MODEL_DIR / "yolo26m_raw_otm_768.onnx"
SIDE = 768
ANCHORS = (SIDE // 8) ** 2 + (SIDE // 16) ** 2 + (SIDE // 32) ** 2


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    parent = json.loads(SOURCE_MANIFEST.read_text())
    if sha(SOURCE_WEIGHTS) != parent["weights_sha256"] or sha(SOURCE_ONNX) != parent["onnx_sha256"]:
        raise RuntimeError("F5R parent weight/ONNX provenance mismatch")
    if not parent["export"].get("dynamic") or parent["nms_nodes"] != 0:
        raise RuntimeError("parent export is not a raw dynamic detector graph")
    sys.path.insert(0, str(PACKAGES))
    import onnx  # noqa: E402

    MODEL_DIR.mkdir(parents=True)
    shutil.copy2(SOURCE_ONNX, ONNX_OUT)
    graph = onnx.load(str(ONNX_OUT))
    inputs, outputs = graph.graph.input, graph.graph.output
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError("expected one raw input and output")
    in_dims = inputs[0].type.tensor_type.shape.dim
    out_dims = outputs[0].type.tensor_type.shape.dim
    if len(in_dims) != 4 or len(out_dims) != 3:
        raise RuntimeError("unexpected tensor ranks")
    in_dims[1].dim_value, in_dims[2].dim_value, in_dims[3].dim_value = 3, SIDE, SIDE
    out_dims[1].dim_value, out_dims[2].dim_value = 84, ANCHORS
    if any(node.op_type == "NonMaxSuppression" for node in graph.graph.node):
        raise RuntimeError("NMS must remain in DeepStream")
    onnx.checker.check_model(graph)
    onnx.save(graph, str(ONNX_OUT))
    record = {"status": "EXPERIMENTAL_CANDIDATE", "source_weights_sha256": parent["weights_sha256"],
        "source_960_onnx_sha256": parent["onnx_sha256"], "candidate_onnx_sha256": sha(ONNX_OUT),
        "source_parser_semantics": "same raw 80-class OTM; class-0 confidence; DeepStream NMS only",
        "input_name": inputs[0].name, "input_shape": ["batch", 3, SIDE, SIDE],
        "output_name": outputs[0].name, "output_shape": ["batch", 84, ANCHORS],
        "shape_conversion": "The F5R ONNX was exported with dynamic=True; only its declared fixed spatial/output dimensions were changed in this isolated copy. Weights and graph operators are unchanged.",
        "nms_nodes": 0, "parent_export_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "parent_export_manifest_sha256": sha(SOURCE_MANIFEST)}
    (OUTPUT / "candidate-manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
