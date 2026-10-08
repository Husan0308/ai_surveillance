#!/usr/bin/env python3
"""Build isolated FP16 YOLO26m context-verifier engines at smaller square shapes.

These engines are diagnostic sidecars only. They are derived from the already
validated F5R dynamic graph and never replace the frozen 640 primary detector.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
F5R = ROOT / ".runtime/freeze/F5R-six-camera-yolo-resolution-20261003"
SOURCE_MODEL = F5R / "models/960"
SOURCE_MANIFEST = SOURCE_MODEL / "export_960.json"
SOURCE_CONFIG = ROOT / "config/deepstream/config_infer_primary_yolo26m_raw_otm.txt"
EXPECTED_PRIMARY_CONFIG = "fbfa196338b6781b7d65c3058d4f02069e9116963804d0c3b23d1bf2301e4113"
PARSER = ROOT / ".runtime/f5-cam05-safety-20261007-r1/yolo768-candidate-r2/models/libyolo26rawperson_768.so"
PARSER_SOURCE = F5R / "src/parser_raw_dynamic.cpp"
OUT = Path(os.environ.get(
    "CAM05_CONTEXT_SHAPE_OUT",
    str(ROOT / ".runtime/cam05-verifier-shapes-20261007-r1"),
)).resolve()
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
WEIGHTS_SHA256 = "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7"
SHAPES = (576, 512, 448)


def selected_shapes() -> tuple[int, ...]:
    raw = os.environ.get("CAM05_CONTEXT_SHAPES", "")
    if not raw:
        return SHAPES
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value < 320 or value % 32 for value in values):
        raise ValueError("CAM05_CONTEXT_SHAPES must be comma-separated multiples of 32 >= 320")
    if len(set(values)) != len(values):
        raise ValueError("CAM05_CONTEXT_SHAPES must not contain duplicates")
    return values


def selected_batch() -> int:
    batch = int(os.environ.get("CAM05_CONTEXT_BATCH", "2"))
    if batch < 1 or batch > 6:
        raise ValueError("CAM05_CONTEXT_BATCH must be between 1 and 6")
    return batch


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_shape_graph(source: Path, target: Path, side: int) -> int:
    graph = __import__("onnx").load(str(source))
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise RuntimeError("expected one raw YOLO input and output")
    in_dims = graph.graph.input[0].type.tensor_type.shape.dim
    out_dims = graph.graph.output[0].type.tensor_type.shape.dim
    if len(in_dims) != 4 or len(out_dims) != 3:
        raise RuntimeError("unexpected raw YOLO tensor rank")
    anchors = sum((side // stride) ** 2 for stride in (8, 16, 32))
    in_dims[1].dim_value, in_dims[2].dim_value, in_dims[3].dim_value = 3, side, side
    out_dims[1].dim_value, out_dims[2].dim_value = 84, anchors
    if any(node.op_type == "NonMaxSuppression" for node in graph.graph.node):
        raise RuntimeError("verifier graph must remain raw; DeepStream owns NMS")
    onnx = __import__("onnx")
    onnx.checker.check_model(graph)
    onnx.save(graph, str(target))
    return anchors


def main() -> int:
    shapes = selected_shapes()
    batch = selected_batch()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite verifier evidence: {OUT}")
    if sha(SOURCE_CONFIG) != EXPECTED_PRIMARY_CONFIG:
        raise RuntimeError("frozen primary detector config changed")
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    source_onnx = SOURCE_MODEL / "yolo26m_raw_otm.onnx"
    source_weights = SOURCE_MODEL / "yolo26m_raw_otm.pt"
    if (sha(source_onnx) != manifest["onnx_sha256"]
            or sha(source_weights) != WEIGHTS_SHA256
            or manifest["weights_sha256"] != WEIGHTS_SHA256
            or not manifest["export"].get("dynamic")
            or manifest.get("nms_nodes") != 0):
        raise RuntimeError("F5R dynamic graph/weights provenance mismatch")
    if not PARSER.is_file() or not PARSER_SOURCE.is_file():
        raise FileNotFoundError("validated dynamic parser source/binary is missing")
    if OUT.resolve().parent != (ROOT / ".runtime").resolve():
        raise RuntimeError("candidate output must be a direct child of .runtime")
    sys.path.insert(0, str(F5R / "export-packages"))
    import onnx  # noqa: E402

    OUT.mkdir(parents=True)
    rows = []
    for side in shapes:
        model_dir = OUT / f"{side}/models"
        config_dir = OUT / f"{side}/config"
        model_dir.mkdir(parents=True)
        config_dir.mkdir()
        candidate_onnx = model_dir / f"yolo26m_context_{side}.onnx"
        engine = model_dir / f"yolo26m_context_{side}_b{batch}_fp16.engine"
        parser = model_dir / "libyolo26rawperson.so"
        labels = model_dir / "labels.txt"
        shutil.copy2(PARSER, parser)
        shutil.copy2(SOURCE_MODEL / "labels.txt", labels)
        anchors = write_shape_graph(source_onnx, candidate_onnx, side)
        config = config_dir / f"context-{side}-batch{batch}.txt"
        config.write_text(f"""[property]
gpu-id=0
net-scale-factor=0.00392156862745098
model-color-format=0
onnx-file=/models/yolo26m_context_{side}.onnx
model-engine-file=/models/yolo26m_context_{side}_b{batch}_fp16.engine
labelfile-path=/models/labels.txt
custom-lib-path=/models/libyolo26rawperson.so
parse-bbox-func-name=NvDsInferParseYolo26RawPerson
batch-size={batch}
network-mode=2
num-detected-classes=80
interval=0
gie-unique-id=1
process-mode=1
network-type=0
warmup-engine=1
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1
infer-dims=3;{side};{side}
output-blob-names=output0

[class-attrs-all]
pre-cluster-threshold=0.20
nms-iou-threshold=0.45
topk=300
""")
        cmd = ["docker", "run", "--rm", "--pull=never", "--network=none", "--gpus", "device=0",
            "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility", "-v", f"{model_dir}:/models:rw",
            "--entrypoint", "/usr/src/tensorrt/bin/trtexec", IMAGE,
            f"--onnx=/models/{candidate_onnx.name}", f"--saveEngine=/models/{engine.name}", "--fp16",
            f"--minShapes=images:1x3x{side}x{side}", f"--optShapes=images:{batch}x3x{side}x{side}",
            f"--maxShapes=images:{batch}x3x{side}x{side}", "--memPoolSize=workspace:2048",
            "--profilingVerbosity=detailed", "--skipInference"]
        log = model_dir / "trtexec-build.log"
        with log.open("x") as stream:
            subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        if "Engine built in" not in log.read_text(errors="replace") or not engine.is_file():
            raise RuntimeError(f"TensorRT build failed for {side}; inspect {log}")
        rows.append({
            "side": side, "input": [3, side, side], "output": [84, anchors],
            "batch": {"min": 1, "opt": batch, "max": batch}, "precision": "FP16",
            "source_weights_sha256": WEIGHTS_SHA256,
            "source_dynamic_onnx_sha256": sha(source_onnx),
            "candidate_onnx_sha256": sha(candidate_onnx),
            "engine_sha256": sha(engine), "parser_source_sha256": sha(PARSER_SOURCE),
            "parser_sha256": sha(parser), "config_sha256": sha(config),
            "build_log_sha256": sha(log), "threshold": 0.20, "nms_iou": 0.45,
            "maintain_aspect_ratio": 1, "symmetric_padding": 1,
            "role": "candidate-local context verification only; primary detector unchanged",
        })
    report = {"status": "DIAGNOSTIC_CANDIDATES_BUILT", "deepstream_image": IMAGE,
        "frozen_primary_config_sha256": EXPECTED_PRIMARY_CONFIG, "candidates": rows}
    (OUT / "candidate-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
