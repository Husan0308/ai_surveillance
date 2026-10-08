#!/usr/bin/env python3
"""Rebuild the isolated six-camera r3 detector/tracker profile from current source."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r6-verifier-admission"
DEFAULT_OUT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r7-current-source"
def configured_path(env_name: str, default: Path) -> Path:
    raw = os.environ.get(env_name)
    if not raw:
        return default
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path

PARENT = configured_path("SIXCAM_CURRENT_SOURCE_PARENT", DEFAULT_PARENT)
OUT = configured_path("SIXCAM_CURRENT_SOURCE_OUT", DEFAULT_OUT)
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
CUDA_HEADERS = Path("/usr/local/cuda-12.6/targets/x86_64-linux/include")
EXPECTED = {
    "primary_engine": "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16",
    "primary_parser": "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19",
    "local_tracker": "5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355",
    "frozen_v13": "5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834",
}


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
    compile_script = (
        "g++ -std=c++17 -O2 " + macros
        + " -I/cuda-host/include -I/opt/nvidia/deepstream/deepstream/sources/includes"
        + " $(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " /src/main.cpp -o /work/bin/" + name
        + " $(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " -L/opt/nvidia/deepstream/deepstream/lib -L/usr/local/cuda/targets/x86_64-linux/lib"
        + " -lnvdsgst_meta -lnvds_meta -lnvbufsurface -lnvbufsurftransform"
        + " -lnvds_infer -lnvinfer -lnvinfer_plugin -lcudart -ldl -pthread"
        + " -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib"
    )
    subprocess.run([
        "docker", "run", "--rm", "--pull=never", "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--mount", f"type=bind,src={CUDA_HEADERS},dst=/cuda-host/include,readonly",
        "--mount", f"type=bind,src={SOURCE},dst=/src,readonly",
        "--mount", f"type=bind,src={OUT},dst=/work",
        IMAGE, "bash", "-lc", compile_script,
    ], cwd=ROOT, check=True)


def main() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite isolated candidate: {OUT}")
    parent_manifest = json.loads((PARENT / "build-manifest.json").read_text())
    if parent_manifest.get("deepstream_image") != IMAGE:
        raise RuntimeError("parent candidate does not use the pinned DeepStream 9.1 image")
    profile = parent_manifest.get("detector_profile", {})
    if (profile.get("primary_engine_sha256") != EXPECTED["primary_engine"]
            or profile.get("primary_parser_sha256") != EXPECTED["primary_parser"]
            or parent_manifest.get("tracker_config_sha256") != EXPECTED["local_tracker"]
            or parent_manifest.get("frozen_v13_reference_sha256") != EXPECTED["frozen_v13"]):
        raise RuntimeError("candidate parent does not match the frozen detector/tracker contract")
    for rel, expected in (
        ("models/yolo26m_raw_otm_b6_fp16.engine", EXPECTED["primary_engine"]),
        ("models/libyolo26rawperson.so", EXPECTED["primary_parser"]),
        ("config/config_tracker_NvDCF_v13_local6.yml", EXPECTED["local_tracker"]),
    ):
        if sha256(PARENT / rel) != expected:
            raise RuntimeError(f"parent protected artifact hash mismatch: {rel}")
    if sha256(ROOT / "config/deepstream/config_tracker_NvDCF_yolo26m_retention_v13_probation0.yml") != EXPECTED["frozen_v13"]:
        raise RuntimeError("frozen V13 source artifact hash mismatch")
    if not (SOURCE / "person_proposal_dedupe.hpp").is_file():
        raise RuntimeError("current source is missing the reviewed generic nested-proposal duplicate guard")

    shutil.copytree(PARENT, OUT)
    compile_binary("cam-six-yolo-local6-posttracker", direct_file=False)
    compile_binary("cam-six-yolo-local6-direct-gatea", direct_file=True)

    manifest_path = OUT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_sha256"] = source_hashes()
    manifest["binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-posttracker")
    manifest["direct_file_binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-direct-gatea")
    manifest["candidate_label"] = OUT.name + ": 640+896 profile rebuilt from exact current source"
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "parent_binary_sha256": parent_manifest["binary_sha256"],
        "change": "recompile both isolated runtime and direct-file binaries from current native source; preserve detector, auxiliary, verifier, tracker, and UI assets",
        "current_source_sha256": manifest["source_sha256"],
        "protected_settings_unchanged": True,
        "diagnostic_only": True,
    }
    manifest["recovery"] = {
        **manifest.get("recovery", {}),
        "candidate_binary_sha256": manifest["binary_sha256"],
        "validated_cycles": 0,
        "status": "REQUIRES_REVALIDATION_ON_EXACT_BINARY_AND_PROFILE",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "BUILT_ISOLATED_CURRENT_SOURCE_CANDIDATE",
        "candidate": str(OUT.relative_to(ROOT)),
        "runtime_binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "primary_engine_sha256": EXPECTED["primary_engine"],
        "primary_parser_sha256": EXPECTED["primary_parser"],
        "local_tracker_sha256": EXPECTED["local_tracker"],
        "frozen_v13_sha256": EXPECTED["frozen_v13"],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
