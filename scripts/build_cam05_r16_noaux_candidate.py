#!/usr/bin/env python3
"""Build a fresh no-auxiliary A/B from the exact r15 live candidate.

This removes only the full-frame 896 supplemental detector branch.  The frozen
640 primary, 0.23 low-confidence sidecar, 448 exact-frame context verifier,
NvDCF, canonical policy, preview, and source recovery source remain as staged in
r15.  The resulting candidate is diagnostic until its replay and live gates
pass.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r15-aux-sparse20"
OUT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r16-noaux"
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compile_binary(name: str, *, direct_file: bool) -> None:
    macros = "-DYOLO26_PERSON -DYOLO26_TRACKER"
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
        raise FileExistsError(f"refusing to overwrite isolated candidate: {OUT}")
    parent_manifest = json.loads((PARENT / "build-manifest.json").read_text())
    if parent_manifest.get("deepstream_image") != IMAGE:
        raise RuntimeError("parent candidate does not use pinned DeepStream 9.1")
    parent_binary = PARENT / "bin/cam-six-yolo-local6-posttracker"
    parent_direct = PARENT / "bin/cam-six-yolo-local6-direct-gatea"
    if sha256(parent_binary) != parent_manifest.get("binary_sha256"):
        raise RuntimeError("r15 live binary does not match its manifest")
    if sha256(parent_direct) != parent_manifest.get("direct_file_binary_sha256"):
        raise RuntimeError("r15 replay binary does not match its manifest")
    profile = parent_manifest.get("detector_profile", {})
    if (profile.get("primary_engine_sha256") !=
            "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16" or
            profile.get("primary_parser_sha256") !=
            "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19" or
            profile.get("normal_nvdcf_input_threshold") != 0.25 or
            profile.get("aux_interval") != 19):
        raise RuntimeError("r15 does not match the pinned primary/auxiliary A/B parent contract")

    shutil.copytree(PARENT, OUT)
    compile_binary("cam-six-yolo-local6-posttracker", direct_file=False)
    compile_binary("cam-six-yolo-local6-direct-gatea", direct_file=True)

    manifest_path = OUT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["detector_profile"] = {
        "name": "yolo26m-fp16-640-primary-context-verified-noaux",
        "candidate_only": True,
        "context_verifier_mode": "yolo26m-current-frame-crop-confirmation",
        "primary_config": profile["primary_config"],
        "primary_engine_sha256": profile["primary_engine_sha256"],
        "primary_parser_sha256": profile["primary_parser_sha256"],
        "primary_threshold": profile["primary_threshold"],
        "normal_nvdcf_input_threshold": profile["normal_nvdcf_input_threshold"],
        "batch_size": 6,
        "auxiliary_branch_compiled": False,
        "candidate_reason": (
            "remove the 896 full-frame branch and its every-frame NVMM conversion; "
            "retain the low-confidence sidecar and exact-frame 448 verifier"
        ),
    }
    manifest["runtime"] = (
        "DeepStream 9.1; frozen YOLO26m FP16 640x640 batch=6 primary unchanged; "
        "persistent exact-frame 448 context verifier; local NvDCF; no supplemental "
        "full-frame detector; no MV3DT/calibration"
    )
    manifest["binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-posttracker")
    manifest["direct_file_binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-direct-gatea")
    manifest["source_sha256"] = {
        path.name: sha256(path) for path in sorted(SOURCE.iterdir()) if path.is_file()
    }
    manifest["build_provenance"] = {
        **manifest.get("build_provenance", {}),
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "change": (
            "compile the exact r15 native sources without the auxiliary 896 branch; "
            "preserve frozen 640 primary, verifier, tracker, canonical, preview, and recovery code"
        ),
        "primary_engine_sha256": profile["primary_engine_sha256"],
        "nvdcf_input_contract_changed": False,
        "context_verifier_changed": False,
        "tracker_changed": False,
        "auxiliary_branch_compiled": False,
    }
    manifest["recovery"] = {
        "required_environment": "V11_SOURCE_STALL_RECOVERY=1",
        "candidate_binary_sha256": manifest["binary_sha256"],
        "validated_cycles": 0,
        "status": "REQUIRES_REVALIDATION_ON_EXACT_BINARY",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "BUILT_DIAGNOSTIC_CANDIDATE",
        "candidate": str(OUT.relative_to(ROOT)),
        "live_binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "primary_engine_sha256": profile["primary_engine_sha256"],
        "tracker_config_sha256": manifest["tracker_config_sha256"],
        "verifier_engine_sha256": manifest["yolo_context_verifier"]["engine_sha256"],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
