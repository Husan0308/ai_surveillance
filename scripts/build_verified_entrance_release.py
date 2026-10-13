#!/usr/bin/env python3
"""Build a reproducible release from an embedded, immutable native snapshot.

This packages the preserved r1 models/runtime profile and compiles all native
entry points from a source copy inside the release directory. It never starts
the production stack.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
PARENT = ROOT / ".runtime/sixcam-entrance-repairs-20261009-r1"
BASE_RELEASE = ROOT / ".runtime/sixcam-verified-release-20261010-r1"
OUT = ROOT / ".runtime/sixcam-verified-release-20261010-r2"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
SOURCE_OVERLAY = {"lowconf_verifier.hpp", "track_birth_verification.hpp"}
EXPECTED = {
    "r1_live_binary": "430aa23a49138b8b916abb350d1831c74abe16bbdb4572fa68759fe94b457d1f",
    "r1_direct_binary": "ed254bb9fbf83680de62d8b13360eb38cf4edf398209eb990955fbb8be880b59",
    "primary_engine": "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16",
    "primary_parser": "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19",
    "auxiliary_engine": "84832420e6e3ad68e1664b2ceb04b38474459960990c00d76987ca8bbda153be",
    "local_tracker": "5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355",
    "frozen_v13": "5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834",
    "diagnostic_0503": "0503ee1c4a588281614fabf3dcd64a0ffc5aea7f0625fd3d05d2280d429c7ef7",
    "base_release_live": "e2fbfa2b5c7cacfde702c8e25a0917171919847627a219e13d7110a49d422a92",
    "base_release_direct": "0503ee1c4a588281614fabf3dcd64a0ffc5aea7f0625fd3d05d2280d429c7ef7",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(path: Path) -> dict[str, str]:
    return {file.relative_to(path).as_posix(): sha256(file)
            for file in sorted(path.rglob("*")) if file.is_file()}


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def run_logged(command: list[str], log: Path, *, timeout: int = 900) -> str:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            timeout=timeout, check=True)
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    return result.stdout.strip()


def docker_command(output_path: str, defines: list[str]) -> list[str]:
    source_dir = OUT / "source/native"
    define_args = " ".join(f"-D{name}" for name in defines)
    command = (
        f"g++ -std=c++17 -O2 {define_args} "
        "-I/host-cuda-include -I/opt/nvidia/deepstream/deepstream/sources/includes "
        "$(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0) "
        f"/src/main.cpp -o /work/{output_path} "
        "$(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0) "
        "-L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -lnvds_meta "
        "-lnvbufsurface -lnvbufsurftransform -lnvds_infer -lnvinfer "
        "-lnvinfer_plugin -L/usr/local/cuda/lib64 -lcudart -ldl -pthread "
        "-Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib"
    )
    return [
        "docker", "run", "--rm", "--pull=never", "--network=none", "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{source_dir}:/src:ro", "-v", f"{OUT}:/work",
        "-v", "/usr/local/cuda/include:/host-cuda-include:ro", IMAGE,
        "bash", "-lc", command,
    ]


def compile_binary(name: str, defines: list[str], log_name: str) -> str:
    command = docker_command(name, defines)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                              timeout=900, check=True)
    (OUT / "build-logs" / log_name).write_text(
        json.dumps({"argv": command, "stdout": completed.stdout,
                    "stderr": completed.stderr}, indent=2) + "\n", encoding="utf-8")
    return sha256(OUT / name)


def main() -> dict:
    if not OUT.is_relative_to(ROOT / ".runtime"):
        raise ValueError("release path must remain under .runtime")
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite release candidate: {OUT}")
    if not SOURCE.is_dir() or not PARENT.is_dir() or not BASE_RELEASE.is_dir():
        raise FileNotFoundError("native sources or preserved r1 release bundle are missing")

    base_manifest = json.loads((BASE_RELEASE / "build-manifest.json").read_text(encoding="utf-8"))
    base_snapshot = BASE_RELEASE / base_manifest["source_snapshot"]["directory"]
    base_snapshot_manifest_path = BASE_RELEASE / base_manifest["source_snapshot"]["manifest"]
    base_snapshot_manifest = json.loads(base_snapshot_manifest_path.read_text(encoding="utf-8"))
    if sha256(BASE_RELEASE / "bin/cam-six-yolo-local6-posttracker") != EXPECTED["base_release_live"]:
        raise RuntimeError("preserved verified r1 base release live binary changed")
    if sha256(BASE_RELEASE / "bin/cam-six-yolo-local6-direct-gatea") != EXPECTED["base_release_direct"]:
        raise RuntimeError("preserved verified r1 base release direct binary changed")
    if sha256(base_snapshot_manifest_path) != base_manifest["source_snapshot"]["manifest_sha256"]:
        raise RuntimeError("preserved r1 source manifest hash mismatch")
    base_source_hashes = tree_hashes(base_snapshot)
    if base_source_hashes != base_snapshot_manifest.get("source_sha256") or \
            canonical_hash(base_source_hashes) != base_manifest["source_snapshot"]["tree_sha256"]:
        raise RuntimeError("preserved r1 embedded source snapshot failed integrity verification")
    current_source_hashes = tree_hashes(SOURCE)
    if set(current_source_hashes) != set(base_source_hashes):
        raise RuntimeError("native source file set drifted from r1; refusing an unreviewed release overlay")
    unexpected_source_changes = [relative for relative in current_source_hashes
        if relative not in SOURCE_OVERLAY and current_source_hashes[relative] != base_source_hashes[relative]]
    if unexpected_source_changes:
        raise RuntimeError("native source changes outside reviewed overlay: " + ", ".join(unexpected_source_changes))
    missing_overlay_changes = [relative for relative in SOURCE_OVERLAY
        if current_source_hashes.get(relative) == base_source_hashes.get(relative)]
    if missing_overlay_changes:
        raise RuntimeError("expected active-context geometry source overlay absent: " + ", ".join(missing_overlay_changes))

    parent_manifest = json.loads((PARENT / "build-manifest.json").read_text(encoding="utf-8"))
    if sha256(PARENT / "bin/cam-six-yolo-local6-posttracker") != EXPECTED["r1_live_binary"]:
        raise RuntimeError("preserved r1 live binary does not match its known SHA256")
    if sha256(PARENT / "bin/cam-six-yolo-local6-direct-gatea") != EXPECTED["r1_direct_binary"]:
        raise RuntimeError("preserved r1 direct-file binary does not match its known SHA256")
    if parent_manifest.get("deepstream_image") != IMAGE:
        raise RuntimeError("preserved r1 bundle is not tied to the pinned DeepStream image")
    if parent_manifest.get("detector_profile", {}).get("aux_interval") != 19:
        raise RuntimeError("preserved r1 profile is not the requested sparse-768/interval-19 profile")
    if parent_manifest.get("recovery", {}).get("required_environment") != "V11_SOURCE_STALL_RECOVERY=1":
        raise RuntimeError("r1 profile does not require the intended source-stall recovery mode")
    protected_paths = {
        "primary_engine": "models/yolo26m_raw_otm_b6_fp16.engine",
        "primary_parser": "models/libyolo26rawperson.so",
        "auxiliary_engine": "models/yolo26m_raw_otm_768_b6_fp16.engine",
        "local_tracker": "config/config_tracker_NvDCF_v13_local6.yml",
    }
    for key, relative in protected_paths.items():
        if sha256(PARENT / relative) != EXPECTED[key]:
            raise RuntimeError(f"preserved r1 protected artifact changed: {key}")

    started = time.monotonic()
    shutil.copytree(PARENT, OUT)
    (OUT / "build-logs").mkdir()
    (OUT / "verification").mkdir()
    snapshot = OUT / "source/native"
    shutil.copytree(base_snapshot, snapshot)
    for relative in sorted(SOURCE_OVERLAY):
        shutil.copy2(SOURCE / relative, snapshot / relative)
    source_hashes = tree_hashes(snapshot)
    if not source_hashes:
        raise RuntimeError("native source snapshot is empty")
    if source_hashes != tree_hashes(SOURCE):
        raise RuntimeError("development source changed while it was being snapshotted")
    source_tree_hash = canonical_hash(source_hashes)
    source_manifest = {
        "schema": "six-camera-native-source-snapshot-v1",
        "origin": "verified r1 immutable source snapshot plus only the reviewed active-context geometry safeguard overlay",
        "base_source_tree_sha256": base_manifest["source_snapshot"]["tree_sha256"],
        "explicit_source_overlay": {relative: source_hashes[relative] for relative in sorted(SOURCE_OVERLAY)},
        "source_sha256": source_hashes,
        "tree_sha256": source_tree_hash,
    }
    source_manifest_path = OUT / "source/source-manifest.json"
    source_manifest_path.write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
    source_manifest_sha = sha256(source_manifest_path)
    recipe = ROOT / "scripts/build_verified_entrance_release.py"
    (OUT / "build-logs/build-recipe.py").write_bytes(recipe.read_bytes())
    recipe_sha = sha256(recipe)

    image_id = subprocess.run([
        "docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    toolchain_command = [
        "docker", "run", "--rm", "--pull=never", "--network=none", "--gpus", "all",
        "--entrypoint", "bash", IMAGE, "-lc",
        "g++ --version; pkg-config --modversion gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0; "
        "nvcc --version | tail -n 4",
    ]
    toolchain = run_logged(toolchain_command, OUT / "build-logs/toolchain.txt")

    defines = ["YOLO26_PERSON", "YOLO26_TRACKER", "YOLO26_AUXILIARY_896"]
    live_sha = compile_binary("bin/cam-six-yolo-local6-posttracker", defines,
                              "compile-live.json")
    direct_sha = compile_binary("bin/cam-six-yolo-local6-direct-gatea",
                                [*defines, "PGIE_ISOLATION"], "compile-direct-file.json")
    direct_repro_sha = compile_binary(
        "verification/direct-repro-build",
        [*defines, "PGIE_ISOLATION"], "compile-direct-reproduction.json")
    if direct_sha != direct_repro_sha:
        raise RuntimeError("two direct-file builds from the candidate source snapshot are not byte-identical")

    repro_live_sha = compile_binary("verification/live-repro-build", defines,
                                    "compile-live-repro.json")
    reproducible = repro_live_sha == live_sha
    diagnostic_match = direct_sha == EXPECTED["diagnostic_0503"]

    before_config = {key: value for key, value in tree_hashes(PARENT / "config").items()}
    after_config = tree_hashes(OUT / "config")
    before_models = tree_hashes(PARENT / "models")
    after_models = tree_hashes(OUT / "models")
    if before_config != after_config or before_models != after_models:
        raise RuntimeError("build changed the preserved r1 config/model assets")
    if sha256(OUT / "config/config_tracker_NvDCF_v13_local6.yml") != EXPECTED["local_tracker"]:
        raise RuntimeError("local tracker config changed during release build")

    manifest = parent_manifest
    manifest.update({
        "candidate_label": "verified source-snapshot release r2; r1 sparse-768 interval-19 profile plus reviewed lifecycle/canonical fixes and active-context geometry safeguard",
        "binary_sha256": live_sha,
        "direct_file_binary_sha256": direct_sha,
        "source_sha256": source_hashes,
        "source_snapshot": {
            "directory": "source/native",
            "manifest": "source/source-manifest.json",
            "manifest_sha256": source_manifest_sha,
            "tree_sha256": source_tree_hash,
        },
        "frozen_v13_reference_sha256": EXPECTED["frozen_v13"],
        "tracker_config_sha256": EXPECTED["local_tracker"],
        "build_provenance": {
            "parent_candidate": str(PARENT.relative_to(ROOT)),
            "parent_live_binary_sha256": EXPECTED["r1_live_binary"],
            "parent_direct_file_binary_sha256": EXPECTED["r1_direct_binary"],
            "source_origin": "verified r1 immutable snapshot with a strict two-file, human-reviewed source overlay; builds use only candidate/source/native",
            "base_release": str(BASE_RELEASE.relative_to(ROOT)),
            "base_release_live_binary_sha256": EXPECTED["base_release_live"],
            "base_release_direct_binary_sha256": EXPECTED["base_release_direct"],
            "source_snapshot_tree_sha256": source_tree_hash,
            "build_recipe_sha256": recipe_sha,
            "deepstream_image": IMAGE,
            "deepstream_image_id": image_id,
            "toolchain": toolchain,
            "compile_defines_live": defines,
            "compile_defines_direct_file": [*defines, "PGIE_ISOLATION"],
            "runtime_profile_unchanged": True,
            "fixes_included": [
                "same-native-ID canonical duplicate protection",
                "reviewed cross-ID context ownership continuation suppression; ambiguous links fail open",
                "source epoch-aware tracker/verifier joins and reconnect telemetry lifecycle",
                "finite, positive, in-frame canonical geometry enforcement",
                "active-track context confirmation requires candidate center containment and IoU >= 0.25, preventing an adjacent context-crop person from confirming static garment/background geometry",
            ],
            "excluded_artifacts": ["0503 remains a diagnostic provenance comparison, not a production release input"],
            "compile_reproducibility": {
                "second_live_build_sha256": repro_live_sha,
                "byte_identical": reproducible,
                "direct_reproduction_binary_sha256": direct_repro_sha,
                "direct_file_builds_byte_identical": direct_sha == direct_repro_sha,
                "prior_0503_binary_sha256": EXPECTED["diagnostic_0503"],
                "diagnostic_0503_hash_reproduced": diagnostic_match,
            },
            "protected_hashes": {
                "primary_engine": EXPECTED["primary_engine"],
                "primary_parser": EXPECTED["primary_parser"],
                "auxiliary_engine": EXPECTED["auxiliary_engine"],
                "local_tracker": EXPECTED["local_tracker"],
                "frozen_v13": EXPECTED["frozen_v13"],
            },
            "calibration_modified": False,
            "recovery_validation": "NOT_RUN_ON_THIS_BINARY",
        },
        "recovery": {
            **parent_manifest.get("recovery", {}),
            "candidate_binary_sha256": live_sha,
            "validated_cycles": 0,
            "status": "REQUIRES_VALIDATION_ON_EXACT_BINARY",
            "required_environment": "V11_SOURCE_STALL_RECOVERY=1",
        },
        "release_build_duration_sec": round(time.monotonic() - started, 3),
    })
    manifest_path = OUT / "build-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    (OUT / "dependency-manifest.json").write_text(json.dumps({
        "schema": "six-camera-dependency-manifest-v1",
        "deepstream_image": IMAGE,
        "deepstream_image_id": image_id,
        "toolchain": toolchain,
        "runtime_libraries": "DeepStream 9.1 container libraries; pinned by image digest",
        "host_gpu_minimum_verified_today": {"model": "NVIDIA GeForce RTX 3060", "driver": "595.99.02"},
        "python_runtime": "repository full-stack virtual environment; production preflight resolves via scripts/runtime_python.sh",
        "source_snapshot_tree_sha256": source_tree_hash,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    deployment = OUT / "deployment"
    deployment.mkdir(exist_ok=True)
    (deployment / "runtime-profile.json").write_text(json.dumps({
        "candidate": OUT.name,
        "binary_sha256": live_sha,
        "camera_order": [f"CAM-{index:02d}" for index in range(1, 7)],
        "unique_rtsp_owners_per_camera": 1,
        "deepstream_batch_size": 6,
        "detector_profile": parent_manifest["detector_profile"],
        "source_stall_recovery": "V11_SOURCE_STALL_RECOVERY=1",
        "ports": {"ml": 8101, "api": 8100, "automagic_calib": 8000},
        "production_start_authorized": False,
        "calibration_deployed": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (deployment / "STARTUP-AND-RECOVERY.md").write_text(
        "# Verified release startup and recovery\n\n"
        "This release is not authorized for production startup until the operator explicitly approves its live binary SHA.\n\n"
        "After approval, run the standard project launcher with this exact candidate path and confirm its source-snapshot validation succeeds before any owner starts. Use `scripts/start_full_live_stack.sh --preflight` first. Start through the existing single-owner supervisor only; never launch a second native owner.\n\n"
        "For failure: request graceful shutdown through the supervisor, verify process exit and RTSP socket release, preserve native/service logs, and do not automatically fall back to the old r1 binary because its current-tree source validation is unresolved. AMC on port 8000 must remain untouched.\n",
        encoding="utf-8")

    return {
        "status": "BUILT_NOT_DEPLOYED",
        "candidate": str(OUT.relative_to(ROOT)),
        "binary_sha256": live_sha,
        "direct_file_binary_sha256": direct_sha,
        "source_tree_sha256": source_tree_hash,
        "source_file_count": len(source_hashes),
        "reproducible_build": reproducible,
        "direct_build_reproducible": direct_sha == direct_repro_sha,
        "prior_0503_binary_matches_new_release": diagnostic_match,
        "production_start_authorized": False,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
