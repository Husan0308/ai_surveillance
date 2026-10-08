#!/usr/bin/env python3
"""Build a fresh isolated candidate for bounded verified-track CAM-05 rescue."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-cascade-20261007-r12-aux-interval2"
VERIFIER = ROOT / ".runtime/sixcam-cam05-aux448-no-shadow-refresh-20261008-r40"
OUT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r2"
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
CONTEXT_CONFIG = "yolo26m_context_verifier_448.txt"
CONTEXT_ENGINE = "yolo26m_context_verifier_448_b2_fp16.engine"
CONTEXT_ONNX = "yolo26m_context_448.onnx"
CONTEXT_PARSER = "libyolo26rawperson_context_dynamic.so"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {path.name: sha256(path) for path in sorted(SOURCE.iterdir()) if path.is_file()}


def compile_binary(name: str, *, direct_file: bool) -> None:
    macros = "-DYOLO26_PERSON -DYOLO26_TRACKER -DYOLO26_AUXILIARY_896"
    if direct_file:
        macros += " -DPGIE_ISOLATION"
    command = (
        "g++ -std=c++17 -O2 " + macros
        + " -I/host-cuda-include -I/opt/nvidia/deepstream/deepstream/sources/includes"
        + " $(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " /src/main.cpp -o /work/bin/" + name
        + " $(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " -L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -lnvds_meta"
        + " -lnvbufsurface -lnvbufsurftransform -lnvds_infer -lnvinfer"
        + " -lnvinfer_plugin -L/usr/local/cuda/lib64 -lcudart -ldl -pthread"
        + " -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib"
    )
    subprocess.run([
        "docker", "run", "--rm", "--pull=never", "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{SOURCE}:/src:ro", "-v", f"{OUT}:/work",
        "-v", "/usr/local/cuda/include:/host-cuda-include:ro", IMAGE,
        "bash", "-lc", command,
    ], cwd=ROOT, check=True)


def main() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {OUT}")
    parent_manifest = json.loads((PARENT / "build-manifest.json").read_text())
    verifier_manifest = json.loads((VERIFIER / "build-manifest.json").read_text())
    if parent_manifest.get("deepstream_image") != IMAGE or verifier_manifest.get("deepstream_image") != IMAGE:
        raise RuntimeError("parent candidates do not use the pinned DeepStream 9.1 image")
    for filename, field in (
        ("cam-six-yolo-local6-posttracker", "binary_sha256"),
        ("cam-six-yolo-local6-direct-gatea", "direct_file_binary_sha256"),
    ):
        if sha256(PARENT / "bin" / filename) != parent_manifest.get(field):
            raise RuntimeError(f"parent binary does not match manifest: {filename}")
    profile = parent_manifest.get("detector_profile", {})
    if (profile.get("primary_engine_sha256") !=
            "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16" or
            profile.get("primary_parser_sha256") !=
            "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19" or
            profile.get("aux_interval") != 2 or profile.get("normal_nvdcf_input_threshold") != 0.25):
        raise RuntimeError("r12 parent detector semantics differ from the accepted candidate")
    v13 = ROOT / "config/deepstream/config_tracker_NvDCF_yolo26m_retention_v13_probation0.yml"
    expected_v13 = "5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834"
    if sha256(v13) != expected_v13:
        raise RuntimeError("frozen V13 tracker hash changed")
    if verifier_manifest.get("yolo_context_verifier", {}).get("engine_sha256") != sha256(
            VERIFIER / "models" / CONTEXT_ENGINE):
        raise RuntimeError("reviewed 448 verifier engine differs from its manifest")

    shutil.copytree(PARENT, OUT)
    for source, target in (
        (VERIFIER / "config" / CONTEXT_CONFIG, OUT / "config" / CONTEXT_CONFIG),
        (VERIFIER / "models" / CONTEXT_ENGINE, OUT / "models" / CONTEXT_ENGINE),
        (VERIFIER / "models" / CONTEXT_ONNX, OUT / "models" / CONTEXT_ONNX),
        (VERIFIER / "models" / CONTEXT_PARSER, OUT / "models" / CONTEXT_PARSER),
    ):
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target)

    compile_binary("cam-six-yolo-local6-posttracker", direct_file=False)
    compile_binary("cam-six-yolo-local6-direct-gatea", direct_file=True)

    manifest_path = OUT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    context_config_sha = sha256(OUT / "config" / CONTEXT_CONFIG)
    context_engine_sha = sha256(OUT / "models" / CONTEXT_ENGINE)
    context_onnx_sha = sha256(OUT / "models" / CONTEXT_ONNX)
    context_parser_sha = sha256(OUT / "models" / CONTEXT_PARSER)
    manifest["yolo_context_verifier"] = {
        **manifest.get("yolo_context_verifier", {}),
        "config": CONTEXT_CONFIG,
        "config_sha256": context_config_sha,
        "engine": CONTEXT_ENGINE,
        "engine_sha256": context_engine_sha,
        "parser": CONTEXT_PARSER,
        "parser_sha256": context_parser_sha,
        "runtime_batch_size": 2,
        "network_input": [3, 448, 448],
        "threshold": 0.20,
        "nms_iou": 0.45,
        "use": (
            "exact-frame context verification for low-confidence boxes and new target births; "
            "a low-confidence box may alternatively use an appearance-verified native target "
            "only with a current continuous INACTIVE shadow and detector association support "
            "within five source frames; shadow geometry never becomes visible geometry"
        ),
        "scheduling": {
            "worker_contexts": 3,
            "camera_workers": 6,
            "queue_capacity_per_camera": 4,
            "global_pending_limit": 32,
            "policy": (
                "persistent FP16 448px batch-2 TensorRT contexts; exact frame/PTS; 26ms "
                "deadline for untrusted targets; late results fail closed; recent verified-track "
                "support does not wait for a redundant current-frame crop"
            ),
        },
    }
    manifest["yolo_context_verifier_variants"] = {
        CONTEXT_CONFIG: {
            "config": CONTEXT_CONFIG,
            "config_sha256": context_config_sha,
            "engine": CONTEXT_ENGINE,
            "engine_sha256": context_engine_sha,
            "engine_source": str((VERIFIER / "models" / CONTEXT_ENGINE).relative_to(ROOT)),
            "engine_profile": {"min": [1, 3, 448, 448], "opt": [2, 3, 448, 448],
                               "max": [2, 3, 448, 448]},
            "onnx": CONTEXT_ONNX,
            "onnx_sha256": context_onnx_sha,
            "parser": CONTEXT_PARSER,
            "parser_sha256": context_parser_sha,
            "input": [3, 448, 448],
            "batch": {"runtime": 2, "engine_min": 1, "engine_opt": 2, "engine_max": 2},
            "precision": "FP16",
            "threshold": 0.20,
            "nms_iou": 0.45,
            "maintain_aspect_ratio": 1,
            "symmetric_padding": 1,
            "purpose": "candidate-local appearance confirmation only; never enters NvDCF",
        }
    }
    manifest["source_sha256"] = source_hashes()
    manifest["config_sha256"] = {
        path.name: sha256(path) for path in sorted((OUT / "config").iterdir()) if path.is_file()
    }
    manifest["model_sha256"] = {
        str(path.relative_to(OUT / "models")): sha256(path)
        for path in sorted((OUT / "models").rglob("*")) if path.is_file()
    }
    manifest["binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-posttracker")
    manifest["direct_file_binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-direct-gatea")
    manifest["shadow_context_refresh_enabled"] = False
    manifest["recovery"] = {
        "required_environment": "V11_SOURCE_STALL_RECOVERY=1",
        "candidate_binary_sha256": manifest["binary_sha256"],
        "validated_cycles": 0,
        "status": "REQUIRES_REVALIDATION_ON_EXACT_BINARY",
    }
    manifest["candidate_label"] = (
        f"{OUT.name}: r12 640+896 cascade with recent appearance-verified target support "
        "for strictly matched low-confidence current PGIE boxes"
    )
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "parent_binary_sha256": parent_manifest["binary_sha256"],
        "verifier_candidate": str(VERIFIER.relative_to(ROOT)),
        "change": (
            "permit a current low-confidence PGIE rectangle only when it uniquely matches "
            "a continuous current INACTIVE shadow whose same native ID was appearance-verified "
            "and detector-associated within five source frames"
        ),
        "invariants": [
            "frozen 640 engine/parser, threshold/NMS/preprocessing and >=0.25 NvDCF input unchanged",
            "896 interval-2 supplemental branch unchanged",
            "NvDCF config and parameters unchanged",
            "low-confidence PGIE metadata remains removed before NvDCF",
            "current-frame PTS, inactive shadow, strict IoU/center/size/uniqueness gates remain required",
            "current PGIE bbox is the only visible fallback rectangle",
            "context-rejected native IDs cannot publish low-confidence fallback",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "BUILT_ISOLATED_CANDIDATE_REQUIRES_VALIDATION",
        "candidate": str(OUT.relative_to(ROOT)),
        "binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "primary_engine_sha256": profile["primary_engine_sha256"],
        "tracker_sha256": manifest["config_sha256"][manifest["tracker_config"]],
        "context_engine_sha256": context_engine_sha,
        "stall_recovery": "must run with V11_SOURCE_STALL_RECOVERY=1 and pass exact-binary cycles",
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
