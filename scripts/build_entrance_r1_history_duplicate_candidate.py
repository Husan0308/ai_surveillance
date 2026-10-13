#!/usr/bin/env python3
"""Build isolated file/live binaries from the verified Entrance r1 profile."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
PARENT = ROOT / ".runtime/sixcam-entrance-repairs-20261009-r1"
OUT = ROOT / ".runtime/sixcam-entrance-r1-history-duplicate-throttled-20261009"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
EXPECTED = {
    "live_binary": "430aa23a49138b8b916abb350d1831c74abe16bbdb4572fa68759fe94b457d1f",
    "direct_file_binary": "ed254bb9fbf83680de62d8b13360eb38cf4edf398209eb990955fbb8be880b59",
    "primary_engine": "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16",
    "primary_parser": "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19",
    "tracker": "5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355",
    "frozen_v13": "5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834",
}
PROTECTED = {
    "primary_engine": "models/yolo26m_raw_otm_b6_fp16.engine",
    "primary_parser": "models/libyolo26rawperson.so",
    "local_tracker": "config/config_tracker_NvDCF_v13_local6.yml",
}
FROZEN_V13 = ROOT / "config/deepstream/config_tracker_NvDCF_yolo26m_retention_v13_probation0.yml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(path: Path) -> dict[str, str]:
    return {str(file.relative_to(path)): sha256(file)
            for file in sorted(path.rglob("*")) if file.is_file()}


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
        "docker", "run", "--rm", "--pull=never", "--network=none", "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{SOURCE}:/src:ro", "-v", f"{OUT}:/work",
        "-v", "/usr/local/cuda/include:/host-cuda-include:ro", IMAGE,
        "bash", "-lc", command,
    ], cwd=ROOT, check=True)


def main() -> dict[str, object]:
    if not PARENT.is_relative_to(ROOT / ".runtime") or not OUT.is_relative_to(ROOT / ".runtime"):
        raise ValueError("candidate paths must remain under .runtime")
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {OUT}")
    parent_manifest = json.loads((PARENT / "build-manifest.json").read_text())
    parent_live = sha256(PARENT / "bin/cam-six-yolo-local6-posttracker")
    parent_file = sha256(PARENT / "bin/cam-six-yolo-local6-direct-gatea")
    if parent_live != EXPECTED["live_binary"] or parent_file != EXPECTED["direct_file_binary"]:
        raise RuntimeError("r1 candidate binaries do not match their approved hashes")
    if parent_manifest.get("deepstream_image") != IMAGE:
        raise RuntimeError("r1 candidate does not use the pinned DeepStream image")
    for name, rel in PROTECTED.items():
        expected = EXPECTED["tracker" if name == "local_tracker" else name]
        if sha256(PARENT / rel) != expected:
            raise RuntimeError(f"protected r1 artifact mismatch: {name}")
    if sha256(FROZEN_V13) != EXPECTED["frozen_v13"]:
        raise RuntimeError("frozen V13 hash changed")

    config_hashes = tree_hashes(PARENT / "config")
    model_hashes = tree_hashes(PARENT / "models")
    shutil.copytree(PARENT, OUT)
    compile_binary("cam-six-yolo-local6-posttracker", direct_file=False)
    compile_binary("cam-six-yolo-local6-direct-gatea", direct_file=True)
    if config_hashes != tree_hashes(OUT / "config") or model_hashes != tree_hashes(OUT / "models"):
        raise RuntimeError("offline candidate build modified model/config assets")

    manifest_path = OUT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_sha256"] = {
        str(file.relative_to(SOURCE)): sha256(file)
        for file in sorted(SOURCE.rglob("*")) if file.is_file()
    }
    manifest["binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-posttracker")
    manifest["direct_file_binary_sha256"] = sha256(OUT / "bin/cam-six-yolo-local6-direct-gatea")
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "parent_live_binary_sha256": parent_live,
        "parent_direct_file_binary_sha256": parent_file,
        "deepstream_image": IMAGE,
        "change": "fail-closed pre-NvDCF duplicate proposal suppression only when exactly two >=0.25 proposals independently match one immediately previous active native target, with bounded current overlap and >=0.15 previous-geometry IoU margin; otherwise fail open",
        "direct_file_test_control": "PGIE_ISOLATION_SLEEP_PER_FRAME_US enables bounded identity sleep before the decoded tee; no live build path uses it; source PTS and frame order remain unchanged",
        "runtime_deployment": "NOT_DEPLOYED; live r1 owner preserved",
        "protected_hashes": {
            "primary_engine": EXPECTED["primary_engine"],
            "primary_parser": EXPECTED["primary_parser"],
            "local_tracker": EXPECTED["tracker"],
            "frozen_v13": EXPECTED["frozen_v13"],
        },
        "calibration_modified": False,
    }
    manifest["history_supported_duplicate_policy"] = {
        "placement": "post-NMS PGIE output, before low-confidence split and NvDCF input",
        "threshold": 0.25,
        "history": "same source epoch, immediately previous source frame, active target geometry",
        "association": "one-to-one unique proposal-to-target match; same target for exactly two proposals",
        "current_pair": {
            "iou_min": 0.25,
            "containment_min": 0.75,
            "center_over_max_height_max": 0.35,
            "center_over_max_width_max": 0.22,
            "width_ratio_max": 2.20,
            "height_ratio_max": 3.20,
        },
        "history_iou_margin_min": 0.15,
        "lowconf_sidecar_modified": False,
        "ambiguous_or_stale_history": "fail open",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "ISOLATED_CANDIDATE_BUILT_NOT_DEPLOYED",
        "candidate": str(OUT.relative_to(ROOT)),
        "live_binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "protected_hashes": manifest["build_provenance"]["protected_hashes"],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
