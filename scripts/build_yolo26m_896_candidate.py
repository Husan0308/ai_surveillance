#!/usr/bin/env python3
"""Build an isolated 896x896 YOLO26m FP16 candidate from the frozen dynamic graph."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
F5R = ROOT / ".runtime/freeze/F5R-six-camera-yolo-resolution-20261003"
SOURCE_MODEL = F5R / "models/960"
SOURCE_MANIFEST = SOURCE_MODEL / "export_960.json"
SOURCE_CONFIG = ROOT / "config/deepstream/config_infer_primary_yolo26m_raw_otm.txt"
EXPECTED_CONFIG_SHA = "fbfa196338b6781b7d65c3058d4f02069e9116963804d0c3b23d1bf2301e4113"
DYNAMIC_PARSER_SOURCE = F5R / "src/parser_raw_dynamic.cpp"
PARSER_768 = ROOT / ".runtime/f5-cam05-safety-20261007-r1/yolo768-candidate-r2/models/libyolo26rawperson_768.so"
OUTPUT = ROOT / ".runtime/f5-cam05-safety-20261007-r1/yolo896-candidate-r1"
MODELS = OUTPUT / "models"
CONFIG_DIR = OUTPUT / "config"
ONNX = MODELS / "yolo26m_raw_otm_896.onnx"
ENGINE = MODELS / "yolo26m_raw_otm_896_b6_fp16.engine"
PARSER = MODELS / "libyolo26rawperson_896.so"
PARSER_ALIAS = MODELS / "libyolo26rawperson.so"
CONFIG = CONFIG_DIR / "config_infer_yolo26m_896_b6.txt"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
SIDE = 896
ANCHORS = sum((SIDE // stride) ** 2 for stride in (8, 16, 32))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite candidate evidence: {OUTPUT}")
    if SIDE % 32 or ANCHORS != 16464:
        raise RuntimeError("unexpected YOLO26 stride-8/16/32 output shape")
    if sha(SOURCE_CONFIG) != EXPECTED_CONFIG_SHA:
        raise RuntimeError("frozen 640 PGIE config changed")
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    source_onnx = SOURCE_MODEL / "yolo26m_raw_otm.onnx"
    source_weights = SOURCE_MODEL / "yolo26m_raw_otm.pt"
    if sha(source_onnx) != source_manifest["onnx_sha256"] or sha(source_weights) != source_manifest["weights_sha256"]:
        raise RuntimeError("F5R source ONNX/weight provenance mismatch")
    if not source_manifest["export"].get("dynamic") or source_manifest["nms_nodes"] != 0:
        raise RuntimeError("source graph must be raw, dynamic, and NMS-free")
    if not all(path.is_file() for path in (PARSER_768, DYNAMIC_PARSER_SOURCE, SOURCE_CONFIG)):
        raise FileNotFoundError("dynamic parser source or validated parser binary missing")

    pack_root = F5R / "export-packages"
    sys.path.insert(0, str(pack_root))
    import onnx  # type: ignore[import-not-found]  # noqa: E402

    MODELS.mkdir(parents=True)
    CONFIG_DIR.mkdir()
    shutil.copy2(source_onnx, ONNX)
    graph = onnx.load(str(ONNX))
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise RuntimeError("unexpected YOLO26 ONNX I/O")
    in_dims = graph.graph.input[0].type.tensor_type.shape.dim
    out_dims = graph.graph.output[0].type.tensor_type.shape.dim
    if len(in_dims) != 4 or len(out_dims) != 3:
        raise RuntimeError("unexpected YOLO26 tensor ranks")
    in_dims[1].dim_value, in_dims[2].dim_value, in_dims[3].dim_value = 3, SIDE, SIDE
    out_dims[1].dim_value, out_dims[2].dim_value = 84, ANCHORS
    if any(node.op_type == "NonMaxSuppression" for node in graph.graph.node):
        raise RuntimeError("DeepStream must remain the only NMS stage")
    onnx.checker.check_model(graph)
    onnx.save(graph, str(ONNX))

    # This parser is shape-derived from DeepStream's network dimensions; reuse
    # the already compiled dynamic parser and record its exact source/binary.
    shutil.copy2(PARSER_768, PARSER)
    shutil.copy2(PARSER_768, PARSER_ALIAS)
    labels = SOURCE_MODEL / "labels.txt"
    shutil.copy2(labels, MODELS / "labels.txt")
    config_text = SOURCE_CONFIG.read_text()
    replacements = {
        "onnx-file=/models/yolo26m_raw_otm.onnx": "onnx-file=/models/yolo26m_raw_otm_896.onnx",
        "model-engine-file=/models/yolo26m_raw_otm_b6_fp16.engine": "model-engine-file=/models/yolo26m_raw_otm_896_b6_fp16.engine",
        "custom-lib-path=/models/libyolo26rawperson.so": "custom-lib-path=/models/libyolo26rawperson_896.so",
        "infer-dims=3;640;640": "infer-dims=3;896;896",
    }
    for old, new in replacements.items():
        if config_text.count(old) != 1:
            raise RuntimeError(f"expected exactly one frozen config value: {old}")
        config_text = config_text.replace(old, new)
    CONFIG.write_text(config_text)

    command = ["docker", "run", "--rm", "--pull=never", "--network=none", "--gpus", "device=0",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility", "-v", f"{MODELS}:/models:rw",
        "--entrypoint", "/usr/src/tensorrt/bin/trtexec", IMAGE,
        "--onnx=/models/yolo26m_raw_otm_896.onnx", "--saveEngine=/models/yolo26m_raw_otm_896_b6_fp16.engine",
        "--fp16", "--minShapes=images:1x3x896x896", "--optShapes=images:6x3x896x896",
        "--maxShapes=images:6x3x896x896", "--memPoolSize=workspace:2048",
        "--profilingVerbosity=detailed", "--skipInference"]
    build_log = MODELS / "trtexec-build.log"
    with build_log.open("x") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    log_text = build_log.read_text(errors="replace")
    if "Engine built in" not in log_text:
        raise RuntimeError("TensorRT did not report a completed engine build")

    manifest = {
        "status": "EXPERIMENTAL_CANDIDATE",
        "candidate": "YOLO26m FP16 896x896, DeepStream nvinfer batch 6",
        "source_weights_sha256": source_manifest["weights_sha256"],
        "source_dynamic_onnx_sha256": sha(source_onnx),
        "candidate_onnx_sha256": sha(ONNX),
        "engine_sha256": sha(ENGINE),
        "parser_source": str(DYNAMIC_PARSER_SOURCE.relative_to(ROOT)),
        "parser_source_sha256": sha(DYNAMIC_PARSER_SOURCE),
        "parser_sha256": sha(PARSER),
        "parser_alias_sha256": sha(PARSER_ALIAS),
        "candidate_config_sha256": sha(CONFIG),
        "frozen_primary_config_sha256": EXPECTED_CONFIG_SHA,
        "input": [SIDE, SIDE], "output": [84, ANCHORS],
        "batch": {"min": 1, "opt": 6, "max": 6}, "precision": "FP16",
        "threshold": 0.25, "nms_iou": 0.45,
        "maintain_aspect_ratio": 1, "symmetric_padding": 1, "interval": 0,
        "weights_graph_operators_and_parser_semantics_unchanged": True,
        "candidate_only_no_production_asset_modified": True,
        "engine_build_log_sha256": sha(build_log),
    }
    (OUTPUT / "candidate-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
