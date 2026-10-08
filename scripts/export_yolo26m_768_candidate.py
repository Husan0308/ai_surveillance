#!/usr/bin/env python3
"""Export a diagnostic 768x768 YOLO26m candidate from the already frozen F5R weights."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_STAGE = ROOT / ".runtime/freeze/F5R-six-camera-yolo-resolution-20261003"
PACKAGES = SOURCE_STAGE / "export-packages"
SOURCE_MODEL_DIR = SOURCE_STAGE / "models/960"
SOURCE_WEIGHTS = SOURCE_MODEL_DIR / "yolo26m_raw_otm.pt"
OUTPUT_DIR = ROOT / ".runtime/f5-cam05-safety-20261007-r1/yolo768-candidate-r1"
MODEL_DIR = OUTPUT_DIR / "models"
WEIGHTS = MODEL_DIR / "yolo26m_raw_otm.pt"
ONNX_OUT = MODEL_DIR / "yolo26m_raw_otm_768.onnx"
EXPECTED_WEIGHTS = "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7"
ULTRALYTICS_ROOT = Path("/home/apsidal/ai_surveillance/.runtime/exporter-8.4.155")
SIDE = 768
ANCHORS = (SIDE // 8) ** 2 + (SIDE // 16) ** 2 + (SIDE // 32) ** 2


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not SOURCE_WEIGHTS.is_file() or sha(SOURCE_WEIGHTS) != EXPECTED_WEIGHTS:
        raise RuntimeError("F5R YOLO26m checkpoint missing or hash mismatch")
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite candidate output: {OUTPUT_DIR}")
    if not ULTRALYTICS_ROOT.is_dir():
        raise FileNotFoundError(ULTRALYTICS_ROOT)
    MODEL_DIR.mkdir(parents=True)
    shutil.copy2(SOURCE_WEIGHTS, WEIGHTS)
    sys.path.insert(0, str(PACKAGES))
    sys.path.insert(1, str(ULTRALYTICS_ROOT))

    import onnx  # noqa: E402
    import torch  # noqa: E402
    import ultralytics  # noqa: E402
    from ultralytics import YOLO  # noqa: E402

    if ultralytics.__version__ != "8.4.155":
        raise RuntimeError(f"unexpected Ultralytics version: {ultralytics.__version__}")
    torch.set_num_threads(4)
    model = YOLO(str(WEIGHTS), task="detect")
    if model.task != "detect" or len(model.names) != 80 or model.names[0] != "person":
        raise RuntimeError("YOLO26m class mapping changed")
    if model.model.yaml["scale"] != "m" or "26" not in str(model.model.yaml["yaml_file"]):
        raise RuntimeError("candidate checkpoint is not YOLO26m")
    model.model.end2end = False
    model.model.eval().to("cuda:0")
    with torch.no_grad():
        output = model.model(torch.zeros(1, 3, SIDE, SIDE, device="cuda:0"))
        raw = output[0] if isinstance(output, tuple) else output
        if tuple(raw.shape) != (1, 84, ANCHORS):
            raise RuntimeError(f"unexpected 768 raw tensor shape: {tuple(raw.shape)}")
    del output, raw

    options = {"format": "onnx", "imgsz": SIDE, "batch": 6, "dynamic": True,
        "simplify": True, "opset": 17, "device": 0, "nms": None, "half": False}
    result = Path(model.export(**options))
    if result.resolve() == WEIGHTS.resolve():
        raise RuntimeError("export unexpectedly resolved to checkpoint path")
    if result.resolve() != ONNX_OUT.resolve():
        if ONNX_OUT.exists():
            raise FileExistsError(ONNX_OUT)
        result.replace(ONNX_OUT)
    graph = onnx.load(str(ONNX_OUT))
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise RuntimeError("expected single raw input/output tensor")
    in_dims = graph.graph.input[0].type.tensor_type.shape.dim
    out_dims = graph.graph.output[0].type.tensor_type.shape.dim
    if len(in_dims) != 4 or len(out_dims) != 3:
        raise RuntimeError("unexpected raw tensor ranks")
    in_dims[1].dim_value, in_dims[2].dim_value, in_dims[3].dim_value = 3, SIDE, SIDE
    out_dims[1].dim_value, out_dims[2].dim_value = 84, ANCHORS
    if any(node.op_type == "NonMaxSuppression" for node in graph.graph.node):
        raise RuntimeError("NMS must remain in DeepStream")
    onnx.checker.check_model(graph)
    onnx.save(graph, str(ONNX_OUT))
    record = {"status": "EXPERIMENTAL_CANDIDATE", "weights_sha256": sha(WEIGHTS),
        "onnx_sha256": sha(ONNX_OUT), "ultralytics": ultralytics.__version__,
        "torch": torch.__version__, "onnx": onnx.__version__, "export": options,
        "input_name": graph.graph.input[0].name, "input_shape": ["batch", 3, SIDE, SIDE],
        "output_name": graph.graph.output[0].name, "output_shape": ["batch", 84, ANCHORS],
        "raw_torch_shape_check": [1, 84, ANCHORS], "nms_nodes": 0,
        "parent_frozen_weights_sha256": EXPECTED_WEIGHTS}
    (OUTPUT_DIR / "export-manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
