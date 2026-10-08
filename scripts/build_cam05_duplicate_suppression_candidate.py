#!/usr/bin/env python3
"""Build an isolated r4 candidate with conservative same-GIE proposal dedupe."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r3-coordinate-contract"
OUT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r4-proposal-dedupe-r2"
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
    parent_manifest_path = PARENT / "build-manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text())
    if parent_manifest.get("deepstream_image") != IMAGE:
        raise RuntimeError("r3 parent is not pinned to the expected DeepStream 9.1 image")
    for filename, field in (
        ("cam-six-yolo-local6-posttracker", "binary_sha256"),
        ("cam-six-yolo-local6-direct-gatea", "direct_file_binary_sha256"),
    ):
        if sha256(PARENT / "bin" / filename) != parent_manifest.get(field):
            raise RuntimeError(f"r3 parent binary does not match its manifest: {filename}")

    protected = {
        "primary_engine": (PARENT / "models/yolo26m_raw_otm_b6_fp16.engine",
            "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16"),
        "primary_parser": (PARENT / "models/libyolo26rawperson.so",
            "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19"),
        "tracker_local6": (PARENT / "config/config_tracker_NvDCF_v13_local6.yml",
            "5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355"),
        "tracker_v13_reference": (ROOT / "config/deepstream/config_tracker_NvDCF_yolo26m_retention_v13_probation0.yml",
            "5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834"),
    }
    for label, (path, expected) in protected.items():
        if sha256(path) != expected:
            raise RuntimeError(f"protected candidate asset changed: {label}")

    shutil.copytree(PARENT, OUT)
    compile_binary("cam-six-yolo-local6-posttracker", direct_file=False)
    compile_binary("cam-six-yolo-local6-direct-gatea", direct_file=True)

    manifest_path = OUT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_sha256"] = source_hashes()
    manifest["binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-posttracker")
    manifest["direct_file_binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-direct-gatea")
    manifest["same_component_nested_duplicate_policy"] = {
        "enabled": True,
        "placement": "after DeepStream per-GIE NMS and before NvDCF input",
        "component_scope": "same class-0 unique_component_id only",
        "iou_min": 0.30,
        "smaller_box_containment_min": 0.95,
        "center_distance_over_max_height_max": 0.25,
        "width_ratio_max": 2.0,
        "height_ratio_max": 2.0,
        "confidence_margin_min": 0.05,
        "unique_higher_confidence_match_required": True,
        "ambiguous_match_policy": "retain all candidates",
        "retained_proposal": "higher detector confidence",
        "cross_component_policy": "unchanged existing primary/auxiliary filter",
    }
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "parent_binary_sha256": parent_manifest["binary_sha256"],
        "change": (
            "after per-GIE DeepStream NMS, suppress only a lower-confidence same-GIE "
            "person proposal that is a unique, strictly nested spatial duplicate; "
            "does not change NMS settings or NvDCF thresholds"
        ),
        "invariants": [
            "YOLO engine, parser, thresholds, NMS, resolution, preprocessing, and batch remain unchanged",
            "NvDCF profile/settings and six-camera source ordering remain unchanged",
            "cross-component primary/auxiliary suppression remains unchanged",
            ">=0.25 NvDCF input changes only by removing evidence-matched duplicate proposals",
            "ambiguous matches fail open",
            "r3 and prior evidence directories are immutable",
        ],
    }
    manifest["recovery"] = {
        **manifest.get("recovery", {}),
        "candidate_binary_sha256": manifest["binary_sha256"],
        "status": "REQUIRES_REVALIDATION_ON_EXACT_BINARY",
        "validated_cycles": 0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "BUILT_ISOLATED_CANDIDATE_REQUIRES_VALIDATION",
        "candidate": str(OUT.relative_to(ROOT)),
        "binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "primary_engine_sha256": protected["primary_engine"][1],
        "primary_parser_sha256": protected["primary_parser"][1],
        "tracker_sha256": protected["tracker_local6"][1],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
