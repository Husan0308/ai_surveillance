#!/usr/bin/env python3
"""Build an isolated r3-derived candidate with an on-demand source-tee recorder."""
from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
PARENT = ROOT / ".runtime/sixcam-cam05-verified-track-20261009-active-birth-geometry-recheck-dedupe-r3"
OUT = ROOT / ".runtime/sixcam-calibration-recording-20261009-r1"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
EXPECTED_PARENT_BINARY = "11b713f98bebcebdfac96afe912bc59ebc530db9e49b4cfc9590eb5e0833c414"
PROTECTED = {
    "primary_engine": ("models/yolo26m_raw_otm_b6_fp16.engine", "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16"),
    "primary_parser": ("models/libyolo26rawperson.so", "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19"),
    "local_tracker": ("config/config_tracker_NvDCF_v13_local6.yml", "5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355"),
}
FROZEN_V13 = ROOT / "config/deepstream/config_tracker_NvDCF_yolo26m_retention_v13_probation0.yml"
FROZEN_V13_SHA256 = "5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {str(path.relative_to(SOURCE)): sha256(path)
            for path in sorted(SOURCE.rglob("*")) if path.is_file()}


def compile_binary(name: str, *, direct_file: bool, output: Path = OUT) -> None:
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
        "-v", f"{SOURCE}:/src:ro", "-v", f"{output}:/work",
        "-v", "/usr/local/cuda/include:/host-cuda-include:ro", IMAGE,
        "bash", "-lc", command,
    ], cwd=ROOT, check=True)


def main(parent: Path = PARENT, output: Path = OUT) -> dict:
    parent = parent.resolve()
    output = output.resolve()
    if not parent.is_relative_to(ROOT / ".runtime") or not output.is_relative_to(ROOT / ".runtime"):
        raise ValueError("candidate parent/output must remain under .runtime")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {output}")
    if sha256(FROZEN_V13) != FROZEN_V13_SHA256:
        raise RuntimeError("frozen V13 tracker hash changed")
    parent_binary = parent / "bin/cam-six-yolo-local6-posttracker"
    if sha256(parent_binary) != EXPECTED_PARENT_BINARY:
        raise RuntimeError("the selected r3 parent executable no longer matches its recorded SHA256")
    for name, (relative, expected) in PROTECTED.items():
        if sha256(parent / relative) != expected:
            raise RuntimeError(f"protected parent asset changed: {name}")
    if json.loads((parent / "build-manifest.json").read_text()).get("deepstream_image") != IMAGE:
        raise RuntimeError("selected parent is not built for the pinned DeepStream image")

    shutil.copytree(parent, output)
    compile_binary("cam-six-yolo-local6-posttracker", direct_file=False, output=output)
    compile_binary("cam-six-yolo-local6-direct-gatea", direct_file=True, output=output)
    manifest_path = output / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    live = output / "bin/cam-six-yolo-local6-posttracker"
    direct = output / "bin/cam-six-yolo-local6-direct-gatea"
    manifest["candidate_label"] = "r3 local-NvDCF candidate with optional native-resolution 120s/180s calibration tee recording"
    manifest["source_sha256"] = source_hashes()
    manifest["binary_sha256"] = sha256(live)
    manifest["direct_file_binary_sha256"] = sha256(direct)
    manifest["calibration_recording"] = {
        "enabled_by_default": False,
        "activation": "local control file after READY; select 120s or 180s; optional tee branches on decoded per-camera NVMM before nvstreammux",
        "supported_duration_sec": [120, 180],
        "room_camera_pairs": {"entrance": ["CAM-02", "CAM-05"], "main_room": ["CAM-03", "CAM-06"]},
        "encoding": "NVENC H.264, native source width/height, no resize, no B frames, MP4 qtmux",
        "analytics_path_changed": False,
        "rtsp_owners_added": 0,
        "cross_camera_identity": "not provided by current local-only candidate",
    }
    manifest["build_provenance"] = {
        **manifest.get("build_provenance", {}),
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "parent_binary_sha256": EXPECTED_PARENT_BINARY,
        "changes": [
            "optional on-demand native-resolution recording branch on selected existing decoded source tees",
            "host-side controller starts only after operator READY and verifies both encoded streams before movement prompts",
            "shared-memory human evidence recorder supports camera filtering and opens no RTSP connection",
        ],
        "invariants": [
            "YOLO models, configs, parser, thresholds and detector cadence copied unchanged",
            "local NvDCF and frozen V13 artifacts copied unchanged",
            "calibration assets untouched",
            "recording branch disabled unless explicitly enabled and commanded",
            "one RTSP source owner per camera",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "BUILT_ISOLATED_RECORDER_CANDIDATE",
        "candidate": str(output.relative_to(ROOT)),
        "binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "parent_binary_sha256": EXPECTED_PARENT_BINARY,
        **{name: expected for name, (_, expected) in PROTECTED.items()},
        "frozen_v13_sha256": FROZEN_V13_SHA256,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--output", type=Path, default=OUT)
    arguments = parser.parse_args()
    print(json.dumps(main(arguments.parent, arguments.output), indent=2))
