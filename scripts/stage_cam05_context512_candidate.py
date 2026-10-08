#!/usr/bin/env python3
"""Stage an isolated square, batch-2 verifier on the exact parent candidate."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PARENT = Path(os.environ.get("CAM05_VERIFIER_PARENT",
    ROOT / ".runtime/sixcam-cam05-cascade-20261007-r13-aux-interval2")).resolve()
SIDE = int(os.environ.get("CAM05_VERIFIER_SIDE", "512"))
if SIDE not in (448, 512, 576):
    raise ValueError("CAM05_VERIFIER_SIDE must be one of 448, 512, or 576")
SHAPES = ROOT / f".runtime/cam05-verifier-shapes-20261007-r1/{SIDE}"
OUT = Path(os.environ.get("CAM05_VERIFIER_OUT",
    ROOT / f".runtime/sixcam-cam05-cascade-20261008-r38-shadow-context-{SIDE}-b2")).resolve()
ENGINE_SOURCE = Path(os.environ.get("CAM05_VERIFIER_ENGINE_SOURCE",
    SHAPES / "models" / f"yolo26m_context_{SIDE}_b2_fp16.engine")).resolve()
CONFIG_SOURCE = SHAPES / f"config/context-{SIDE}-batch2.txt"
ENGINE_NAME = f"yolo26m_context_verifier_{SIDE}_b2_fp16.engine"
CONFIG_NAME = f"yolo26m_context_verifier_{SIDE}.txt"
PARSER_NAME = "libyolo26rawperson_context_dynamic.so"
RUNTIME_BATCH = 2


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {OUT}")
    manifest_path = PARENT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if sha(PARENT / "bin/cam-six-yolo-local6-posttracker") != manifest["binary_sha256"]:
        raise RuntimeError("parent candidate binary provenance mismatch")
    if not ENGINE_SOURCE.is_file():
        raise FileNotFoundError(ENGINE_SOURCE)
    shutil.copytree(PARENT, OUT)
    engine = OUT / "models" / ENGINE_NAME
    shutil.copy2(ENGINE_SOURCE, engine)
    parser_source = SHAPES / "models/libyolo26rawperson.so"
    parser_target = OUT / "models" / PARSER_NAME
    # The candidate verifier parser derives its anchor count from the configured
    # input dimensions; retain the frozen 640 parser for primary/896 paths.
    shutil.copy2(parser_source, parser_target)
    text = (OUT / "config/yolo26m_context_verifier.txt").read_text()
    batch_lines = [line for line in text.splitlines() if line.startswith("batch-size=")]
    if len(batch_lines) != 1:
        raise RuntimeError("expected one baseline verifier batch-size field")
    text = text.replace(batch_lines[0], f"batch-size={RUNTIME_BATCH}")
    replacements = {
        "gie-unique-id=1": "gie-unique-id=2",
        "onnx-file=/models/yolo26m_raw_otm.onnx": f"onnx-file=/models/yolo26m_context_{SIDE}.onnx",
        "model-engine-file=/models/yolo26m_raw_otm_b6_fp16.engine": f"model-engine-file=/models/{ENGINE_NAME}",
        "infer-dims=3;640;640": f"infer-dims=3;{SIDE};{SIDE}",
        "custom-lib-path=/models/libyolo26rawperson.so": f"custom-lib-path=/models/{PARSER_NAME}",
    }
    for old, new in replacements.items():
        if text.count(old) == 0 and text.count(new) == 1:
            continue
        if text.count(old) != 1:
            raise RuntimeError(f"expected one baseline verifier setting: {old}")
        text = text.replace(old, new)
    # The diagnostic model ONNX is mounted with the candidate models for full
    # runtime provenance, though nvinfer loads the prebuilt engine.
    shutil.copy2(SHAPES / "models" / f"yolo26m_context_{SIDE}.onnx",
        OUT / "models" / f"yolo26m_context_{SIDE}.onnx")
    config = OUT / "config" / CONFIG_NAME
    config.write_text(text)
    required = (f"batch-size={RUNTIME_BATCH}", "gie-unique-id=2", f"infer-dims=3;{SIDE};{SIDE}",
        f"model-engine-file=/models/{ENGINE_NAME}", f"custom-lib-path=/models/{PARSER_NAME}",
        "pre-cluster-threshold=0.20",
        "nms-iou-threshold=0.45", "maintain-aspect-ratio=1", "symmetric-padding=1")
    if any(text.count(field) != 1 for field in required):
        raise RuntimeError(f"{SIDE} verifier config violates the batch-2 FP16 context contract")
    manifest = json.loads((OUT / "build-manifest.json").read_text())
    manifest.setdefault("config_sha256", {})[CONFIG_NAME] = sha(config)
    manifest.setdefault("model_sha256", {})[ENGINE_NAME] = sha(engine)
    manifest["model_sha256"][PARSER_NAME] = sha(parser_target)
    manifest["model_sha256"][f"yolo26m_context_{SIDE}.onnx"] = sha(OUT / "models" / f"yolo26m_context_{SIDE}.onnx")
    variant = {
        "config": CONFIG_NAME, "config_sha256": sha(config),
        "engine": ENGINE_NAME, "engine_sha256": sha(engine),
        "engine_source": str(ENGINE_SOURCE.relative_to(ROOT)),
        "engine_profile": {
            "min": [1, 3, SIDE, SIDE],
            "opt": [RUNTIME_BATCH, 3, SIDE, SIDE],
            "max": [RUNTIME_BATCH, 3, SIDE, SIDE],
        },
        "onnx": f"yolo26m_context_{SIDE}.onnx", "onnx_sha256": sha(OUT / "models" / f"yolo26m_context_{SIDE}.onnx"),
        "parser": PARSER_NAME, "parser_sha256": sha(parser_target),
        "parser_source": "services/camera_v11/native/six_camera_v13/parser_raw_dynamic.cpp",
        "parser_source_sha256": sha(ROOT / "services/camera_v11/native/six_camera_v13/parser_raw_dynamic.cpp"),
        "source_weights_sha256": json.loads((SHAPES.parent / "candidate-manifest.json").read_text())[
            "candidates"][1]["source_weights_sha256"],
        "input": [3, SIDE, SIDE], "batch": {"runtime": RUNTIME_BATCH,
            "engine_min": 1, "engine_opt": RUNTIME_BATCH, "engine_max": RUNTIME_BATCH},
        "precision": "FP16", "threshold": 0.20, "nms_iou": 0.45,
        "maintain_aspect_ratio": 1, "symmetric_padding": 1,
        "purpose": "isolated crop verifier only; does not enter or modify NvDCF input",
    }
    manifest["yolo_context_verifier_variants"] = {CONFIG_NAME: variant}
    # Select this crop model only for the isolated candidate. The normal
    # primary PGIE remains the frozen 640 engine and retains its NvDCF input.
    manifest["yolo_context_verifier"] = {
        "config": CONFIG_NAME,
        "config_sha256": sha(config),
        "engine": ENGINE_NAME,
        "engine_sha256": sha(engine),
        "parser": PARSER_NAME,
        "parser_sha256": sha(parser_target),
        "runtime_batch_size": RUNTIME_BATCH,
        "network_input": [3, SIDE, SIDE],
        "threshold": 0.20,
        "nms_iou": 0.45,
        "use": "current-frame local person confirmation for low-confidence sidecar, verified target births, and immediate INACTIVE-shadow continuity; never changes NvDCF input",
        "scheduling": {"worker_contexts": 3, "camera_workers": 6,
            "queue_capacity_per_camera": 4,
            "global_pending_limit": 32,
            "policy": "six bounded per-camera crop workers feed three persistent batch-2 TensorRT contexts; exact source frame/PTS; 26ms deadline; late result fails closed"},
    }
    manifest["shadow_context_refresh_enabled"] = True
    manifest["candidate_label"] = (
        f"{OUT.name}: exact parent detector/tracker/runtime plus isolated {SIDE} batch-2 context verifier")
    (OUT / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "STAGED_EXPERIMENTAL", "candidate": str(OUT.relative_to(ROOT)),
        "binary_sha256": manifest["binary_sha256"], "primary_engine_sha256": manifest["detector_profile"]["primary_engine_sha256"],
        "primary_config_sha256": manifest["config_sha256"]["config_infer_primary_yolo26m_raw_otm_lowconf023.txt"],
        "tracker_sha256": manifest["tracker_config_sha256"], "verifier": variant}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
