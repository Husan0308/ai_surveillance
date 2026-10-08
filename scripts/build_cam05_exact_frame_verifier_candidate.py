"""Build an isolated CAM-05 verifier candidate with pre-NvDCF crop scheduling."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-aux448-no-shadow-refresh-20261008-r40"
OUT = ROOT / ".runtime/sixcam-cam05-aux448-pretracker-verify-20261008-r41b"
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {path.name: sha256(path) for path in sorted(SOURCE.iterdir()) if path.is_file()}


def build_binary(target: Path, direct_file: bool) -> None:
    macros = "-DYOLO26_PERSON -DYOLO26_TRACKER -DYOLO26_AUXILIARY_896"
    if direct_file:
        macros += " -DPGIE_ISOLATION"
    compile_script = (
        "g++ -std=c++17 -O2 " + macros
        + " -I/opt/nvidia/deepstream/deepstream/sources/includes"
        + " $(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " /src/main.cpp -o /work/bin/" + target.name
        + " $(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " -L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -lnvds_meta -ldl -pthread"
        + " -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib"
    )
    command = [
        "docker", "run", "--rm", "--pull=never", "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v", f"{SOURCE}:/src:ro", "-v", f"{OUT}:/work", IMAGE,
        "bash", "-lc", compile_script,
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite isolated candidate: {OUT}")
    parent_manifest_path = PARENT / "build-manifest.json"
    manifest = json.loads(parent_manifest_path.read_text())
    parent_live = PARENT / "bin/cam-six-yolo-local6-posttracker"
    parent_direct = PARENT / "bin/cam-six-yolo-local6-direct-gatea"
    if sha256(parent_live) != manifest.get("binary_sha256"):
        raise RuntimeError("parent live binary does not match its manifest")
    if sha256(parent_direct) != manifest.get("direct_file_binary_sha256"):
        raise RuntimeError("parent direct-file binary does not match its manifest")
    if manifest.get("recovery", {}).get("required_environment") != "V11_SOURCE_STALL_RECOVERY=1":
        raise RuntimeError("parent recovery contract is not enabled for final acceptance")
    shutil.copytree(PARENT, OUT)

    live_binary = OUT / "bin/cam-six-yolo-local6-posttracker"
    direct_binary = OUT / "bin/cam-six-yolo-local6-direct-gatea"
    build_binary(live_binary, direct_file=False)
    build_binary(direct_binary, direct_file=True)

    source_manifest = source_hashes()
    live_hash, direct_hash = sha256(live_binary), sha256(direct_binary)
    manifest["candidate_label"] = (
        f"{OUT.name}: r40 detector/tracker artifacts with pre-NvDCF exact-frame context request join"
    )
    manifest["source_sha256"] = source_manifest
    manifest["binary_sha256"] = live_hash
    manifest["direct_file_binary_sha256"] = direct_hash
    manifest["recovery"] = {
        "required_environment": "V11_SOURCE_STALL_RECOVERY=1",
        "candidate_binary_sha256": live_hash,
        "validated_cycles": 0,
        "status": "REQUIRES_REVALIDATION_ON_EXACT_BINARY",
    }
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "parent_live_binary_sha256": sha256(parent_live),
        "parent_direct_file_binary_sha256": sha256(parent_direct),
        "source_sha256": source_manifest,
        "binary_sha256": live_hash,
        "direct_file_binary_sha256": direct_hash,
        "change": (
            "active-track context crop is submitted at the NvDCF sink, joined after association by "
            "source/frame/PTS/detector index/native ID, and the per-camera canonical publisher "
            "waits only to the existing bounded verifier deadline for that exact result"
        ),
        "invariants": [
            "primary >=0.25 detector object metadata is unchanged before NvDCF",
            "low-confidence and auxiliary proposals preserve the existing candidate policy",
            "late verifier results fail closed for their source frame",
            "YOLO/V13/tracker/config/calibration artifacts copied byte-identically from r40",
        ],
    }
    (OUT / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "BUILT_EXPERIMENTAL",
        "candidate": str(OUT.relative_to(ROOT)),
        "live_binary_sha256": live_hash,
        "direct_file_binary_sha256": direct_hash,
        "source_sha256": source_manifest,
        "stall_recovery_required": "V11_SOURCE_STALL_RECOVERY=1",
        "v13_sha256": manifest["tracker_config_sha256"],
        "primary_engine_sha256": manifest["detector_profile"]["primary_engine_sha256"],
        "primary_parser_sha256": manifest["detector_profile"]["primary_parser_sha256"],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
