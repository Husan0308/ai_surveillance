#!/usr/bin/env python3
"""Build an isolated r43 candidate for a tighter context-verifier crop A/B."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-final-20261008-r42-aux896-interval2-dedupe"
OUT = ROOT / ".runtime/sixcam-cam05-final-20261008-r43-candidate-centered-crop"
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
PRIMARY_ENGINE = "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16"
PRIMARY_PARSER = "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19"
LOCAL_TRACKER = "5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355"
V13_REFERENCE = "5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {path.name: sha256(path) for path in sorted(SOURCE.iterdir()) if path.is_file()}


def compile_binary(name: str, *, direct_file: bool) -> None:
    macros = "-DYOLO26_PERSON -DYOLO26_TRACKER -DYOLO26_AUXILIARY_DETECTOR"
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
        "docker", "run", "--rm", "--pull=never", "--network=none", "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{SOURCE}:/src:ro", "-v", f"{OUT}:/work",
        "-v", "/usr/local/cuda/include:/host-cuda-include:ro", IMAGE,
        "bash", "-lc", command,
    ], cwd=ROOT, check=True)


def main() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite candidate evidence: {OUT}")
    if not OUT.is_relative_to(ROOT / ".runtime"):
        raise ValueError("candidate must stay under .runtime")
    parent_manifest = json.loads((PARENT / "build-manifest.json").read_text())
    if parent_manifest.get("deepstream_image") != IMAGE:
        raise RuntimeError("parent candidate does not use the pinned DeepStream 9.1 image")
    profile = parent_manifest.get("detector_profile", {})
    if (profile.get("primary_engine_sha256") != PRIMARY_ENGINE
            or profile.get("primary_parser_sha256") != PRIMARY_PARSER
            or parent_manifest.get("tracker_config_sha256") != LOCAL_TRACKER
            or parent_manifest.get("frozen_v13_reference_sha256") != V13_REFERENCE
            or profile.get("aux_interval") != 2):
        raise RuntimeError("parent candidate differs from the protected r42 diagnostic profile")
    for binary, field in (
        ("cam-six-yolo-local6-posttracker", "binary_sha256"),
        ("cam-six-yolo-local6-direct-gatea", "direct_file_binary_sha256"),
    ):
        if sha256(PARENT / "bin" / binary) != parent_manifest.get(field):
            raise RuntimeError(f"parent binary provenance mismatch: {binary}")

    shutil.copytree(PARENT, OUT)
    compile_binary("cam-six-yolo-local6-posttracker", direct_file=False)
    compile_binary("cam-six-yolo-local6-direct-gatea", direct_file=True)

    manifest_path = OUT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["candidate_label"] = OUT.name + ": diagnostic candidate-centered context crop; no primary/tracker changes"
    manifest["source_sha256"] = source_hashes()
    manifest["binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-posttracker")
    manifest["direct_file_binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-direct-gatea")
    manifest["config_sha256"] = {
        path.name: sha256(path) for path in sorted((OUT / "config").iterdir()) if path.is_file()
    }
    manifest["model_sha256"] = {
        str(path.relative_to(OUT / "models")): sha256(path)
        for path in sorted((OUT / "models").rglob("*")) if path.is_file()
    }
    manifest["context_crop_diagnostic"] = {
        "environment": "V11_CONTEXT_CANDIDATE_CROP=1",
        "default": "legacy-scene-context-v1",
        "candidate": "candidate-centered-zoom-v1",
        "crop": {"context_scale": 1.8, "minimum_source_side": 256,
                 "maximum_source_side": 640, "target_must_be_fully_inside": True},
        "scope": "context-verifier pixels only; tracker input and published bbox unchanged",
        "diagnostic_only": True,
    }
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "change": "add opt-in candidate-centered crop for the existing persistent context verifier",
        "invariants": [
            "primary YOLO engine/parser/config/threshold/preprocessing unchanged",
            "normal >=0.25 primary proposals and NvDCF input unchanged",
            "auxiliary detector and cadence unchanged",
            "local NvDCF and frozen V13 hashes unchanged",
            "verifier still uses current frame and existing strict candidate-local decision gates",
            "no camera-specific ROI or coordinate policy",
        ],
    }
    (OUT / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "BUILT_ISOLATED_DIAGNOSTIC_CANDIDATE",
        "candidate": str(OUT.relative_to(ROOT)),
        "live_binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "primary_engine_sha256": profile["primary_engine_sha256"],
        "primary_parser_sha256": profile["primary_parser_sha256"],
        "tracker_sha256": LOCAL_TRACKER,
        "frozen_v13_sha256": V13_REFERENCE,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
