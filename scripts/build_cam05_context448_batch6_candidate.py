"""Build an isolated six-camera batched direct-NvDsInferContext verifier from r63."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-context448-direct-context-20261008-r63"
OUT = ROOT / ".runtime/sixcam-cam05-context448-direct-context-20261008-r64"
SHAPE = ROOT / ".runtime/cam05-verifier-shapes-20261008-448-b6/448"
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
CUDA_HEADERS = Path("/usr/local/cuda-12.6/targets/x86_64-linux/include")
ENGINE_NAME = "yolo26m_context_verifier_448_b6_fp16.engine"
CONFIG_NAME = "yolo26m_context_verifier_448_b6.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {path.name: sha256(path) for path in sorted(SOURCE.iterdir()) if path.is_file()}


def build_binary(path: Path, *, direct_file: bool) -> None:
    macros = "-DYOLO26_PERSON -DYOLO26_TRACKER -DYOLO26_AUXILIARY_DETECTOR"
    if direct_file:
        macros += " -DPGIE_ISOLATION"
    compile_script = (
        "g++ -std=c++17 -O2 " + macros
        + " -I/cuda-host/include"
        + " -I/opt/nvidia/deepstream/deepstream/sources/includes"
        + " $(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " /src/main.cpp -o /work/bin/" + path.name
        + " $(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " -L/opt/nvidia/deepstream/deepstream/lib -L/usr/local/cuda/targets/x86_64-linux/lib -lnvdsgst_meta -lnvds_meta -lnvbufsurface -lnvbufsurftransform -lnvds_infer -lnvinfer -lnvinfer_plugin -lcudart -ldl -pthread"
        + " -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib"
    )
    subprocess.run([
        "docker", "run", "--rm", "--pull=never", "--gpus", "all", "--user", f"{os.getuid()}:{os.getgid()}",
        "--mount", f"type=bind,src={CUDA_HEADERS},dst=/cuda-host/include,readonly",
        "-v", f"{SOURCE}:/src:ro", "-v", f"{OUT}:/work", IMAGE,
        "bash", "-lc", compile_script,
    ], cwd=ROOT, check=True)


def main() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite isolated candidate: {OUT}")
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    for name, key in (("cam-six-yolo-local6-posttracker", "binary_sha256"),
                      ("cam-six-yolo-local6-direct-gatea", "direct_file_binary_sha256")):
        if sha256(PARENT / "bin" / name) != parent.get(key):
            raise RuntimeError(f"r54 parent binary does not match manifest: {name}")
    if parent.get("deepstream_image") != IMAGE:
        raise RuntimeError("parent does not use the pinned DeepStream 9.1 image")
    source_engine = SHAPE / "models/yolo26m_context_448_b6_fp16.engine"
    source_onnx = SHAPE / "models/yolo26m_context_448.onnx"
    source_parser = SHAPE / "models/libyolo26rawperson.so"
    template = SHAPE / "config/context-448-batch6.txt"
    if not all(path.is_file() for path in (source_engine, source_onnx, source_parser, template)):
        raise FileNotFoundError("reviewed 448 batch-6 verifier artifacts are incomplete")
    if sha256(source_parser) != parent["yolo_context_verifier"]["parser_sha256"]:
        raise RuntimeError("context parser differs from r54's reviewed parser")
    if sha256(source_onnx) != parent["yolo_context_verifier_variants"][
            "yolo26m_context_verifier_448.txt"]["onnx_sha256"]:
        raise RuntimeError("448 ONNX differs from r54's reviewed verifier graph")

    shutil.copytree(PARENT, OUT)
    models, configs = OUT / "models", OUT / "config"
    shutil.copy2(source_engine, models / ENGINE_NAME)
    config_text = template.read_text().replace(
        "/models/yolo26m_context_448_b6_fp16.engine", f"/models/{ENGINE_NAME}")
    if config_text.count("gie-unique-id=1") != 1:
        raise RuntimeError("verifier template must contain exactly one unique component id")
    config_text = config_text.replace("gie-unique-id=1", "gie-unique-id=2", 1)
    context_parser = parent["yolo_context_verifier"]["parser"]
    if config_text.count("custom-lib-path=/models/libyolo26rawperson.so") != 1:
        raise RuntimeError("verifier template must contain exactly one parser path")
    config_text = config_text.replace(
        "custom-lib-path=/models/libyolo26rawperson.so",
        f"custom-lib-path=/models/{context_parser}", 1)
    if (config_text.count("batch-size=6") != 1
            or config_text.count(f"model-engine-file=/models/{ENGINE_NAME}") != 1
            or "infer-dims=3;448;448" not in config_text
            or "pre-cluster-threshold=0.20" not in config_text
            or "nms-iou-threshold=0.45" not in config_text):
        raise RuntimeError("batch-6 verifier config violates the reviewed inference contract")
    (configs / CONFIG_NAME).write_text(config_text)

    build_binary(OUT / "bin/cam-six-yolo-local6-posttracker", direct_file=False)
    build_binary(OUT / "bin/cam-six-yolo-local6-direct-gatea", direct_file=True)

    manifest = json.loads((OUT / "build-manifest.json").read_text())
    # This candidate is for the bounded, exact-frame diagnostic that refreshes
    # an already verified auxiliary NvDCF target from its current shadow state.
    # It does not alter the primary PGIE/NvDCF inputs or relax publication gates.
    manifest["shadow_context_refresh_enabled"] = True
    engine_sha, config_sha = sha256(models / ENGINE_NAME), sha256(configs / CONFIG_NAME)
    parser_name = parent["yolo_context_verifier"]["parser"]
    parser_sha = sha256(models / parser_name)
    onnx_name = parent["yolo_context_verifier_variants"][
        "yolo26m_context_verifier_448.txt"]["onnx"]
    onnx_sha = sha256(models / onnx_name)
    manifest["yolo_context_verifier"] = {
        **manifest["yolo_context_verifier"],
        "config": CONFIG_NAME,
        "config_sha256": config_sha,
        "engine": ENGINE_NAME,
        "engine_sha256": engine_sha,
        "runtime_batch_size": 6,
        "network_input": [3, 448, 448],
        "engine_batch_profile": {"min": 1, "opt": 6, "max": 6},
        "scheduling": {
            "worker_contexts": 1,
            "camera_workers": 6,
            "queue_capacity_per_camera": 4,
            "global_pending_limit": 32,
            "policy": (
                "one persistent NvDsInferContext with dynamic batch<=6 fed by a bounded "
                "ready-crop queue and at most 750us opportunistic microbatch wait; NVMM GPU "
                "crop; exact source frame/PTS; 26ms deadline; late results fail closed"
            ),
        },
    }
    manifest["gpu_context_crop"] = True
    manifest["native_context_backend"] = True
    manifest.setdefault("yolo_context_verifier_variants", {})[CONFIG_NAME] = {
        "config": CONFIG_NAME,
        "config_sha256": config_sha,
        "engine": ENGINE_NAME,
        "engine_sha256": engine_sha,
        "engine_source": str(source_engine.relative_to(ROOT)),
        "engine_profile": {"min": [1, 3, 448, 448], "opt": [6, 3, 448, 448],
                           "max": [6, 3, 448, 448]},
        "onnx": onnx_name,
        "onnx_sha256": onnx_sha,
        "parser": parser_name,
        "parser_sha256": parser_sha,
        "input": [3, 448, 448],
        "batch": {"runtime": 6, "engine_min": 1, "engine_opt": 6, "engine_max": 6},
        "precision": "FP16",
        "threshold": 0.20,
        "nms_iou": 0.45,
        "maintain_aspect_ratio": 1,
        "symmetric_padding": 1,
        "purpose": "isolated context verification only; does not enter or modify NvDCF input",
    }
    manifest["config_sha256"] = {
        path.name: sha256(path) for path in sorted(configs.iterdir()) if path.is_file()
    }
    manifest["model_sha256"] = {
        str(path.relative_to(models)): sha256(path)
        for path in sorted(models.rglob("*")) if path.is_file()
    }
    manifest["source_sha256"] = source_hashes()
    live_sha = sha256(OUT / "bin/cam-six-yolo-local6-posttracker")
    direct_sha = sha256(OUT / "bin/cam-six-yolo-local6-direct-gatea")
    manifest["binary_sha256"] = live_sha
    manifest["direct_file_binary_sha256"] = direct_sha
    manifest["recovery"] = {
        "required_environment": "V11_SOURCE_STALL_RECOVERY=1",
        "candidate_binary_sha256": live_sha,
        "validated_cycles": 0,
        "status": "REQUIRES_REVALIDATION_ON_EXACT_BINARY",
    }
    manifest["candidate_label"] = (
        f"{OUT.name}: shared batch-6 448px verifier with exact-frame shadow refresh diagnostic"
    )
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "parent_binary_sha256": parent["binary_sha256"],
        "parent_direct_file_binary_sha256": parent["direct_file_binary_sha256"],
        "change": "replace six independent NvDsInferContext instances with one persistent shared NvDsInferContext fed by a bounded opportunistic crop microbatch; enable the existing exact-frame shadow-context refresh diagnostic; one engine context consumes up to six GPU-prepared RGB crops in a batch",
        "reason": "remove secondary GStreamer scheduling delay that made otherwise verified low-confidence real-person crops miss the same-frame publication deadline",
        "source_sha256": manifest["source_sha256"],
        "binary_sha256": live_sha,
        "direct_file_binary_sha256": direct_sha,
        "invariants": [
            "frozen 640 primary engine/parser/config and >=0.25 NvDCF inputs are unchanged",
            "frozen NvDCF local6 profile and calibration are unchanged",
            "current-frame candidate-local verifier gates remain unchanged",
            "each context loads the manifest-pinned 448 FP16 engine and the same parser/NMS configuration",
            "no batch wait; inference output remains same-frame and late results fail closed",
            "exact frame/PTS and strict spatial match remain required; late results fail closed",
            "shadow refresh uses only prior verified auxiliary targets and current INACTIVE NvDCF state",
            "CAM-01/CAM-03 accepted human evidence remains outside candidate/evidence writes",
        ],
    }
    (OUT / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "BUILT_ISOLATED_DIAGNOSTIC_CANDIDATE",
        "candidate": str(OUT.relative_to(ROOT)),
        "live_binary_sha256": live_sha,
        "direct_file_binary_sha256": direct_sha,
        "context_engine_sha256": engine_sha,
        "context_config_sha256": config_sha,
        "recovery": "requires V11_SOURCE_STALL_RECOVERY=1; not yet validated on this binary",
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
