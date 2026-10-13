#!/usr/bin/env python3
"""Build a reproducible offline release over the preserved r2 runtime profile.

No service is started. The native binaries are compiled only from the
candidate's embedded read-only source snapshot in the pinned DeepStream image.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PARENT = Path(os.environ.get(
    "SIXCAM_RELEASE_PARENT", str(ROOT / ".runtime/sixcam-verified-release-20261010-r2")
)).resolve()
OUT = Path(os.environ.get(
    "SIXCAM_RELEASE_OUT", str(ROOT / ".runtime/sixcam-reliability-release-20261010-r1")
)).resolve()
NATIVE_SOURCE = Path(os.environ.get(
    "SIXCAM_NATIVE_SOURCE", str(ROOT / "services/camera_v11/native/six_camera_v13")
)).resolve()
APPLICATION_SOURCE_ROOT = Path(os.environ.get(
    "SIXCAM_APPLICATION_SOURCE_ROOT", str(ROOT)
)).resolve()
VALIDATION_ATTESTATION = os.environ.get("SIXCAM_VALIDATION_ATTESTATION", "").strip()
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
EXPECTED_PARENT_BINARY = "5b423df4db0a8f4c3b73eba0164691953c6452c872d6b17ba28d09e7f7d6b1b9"
APP_FILES = (
    ".env.example",
    "services/shared/deployment.py",
    "services/api_service/app/config.py",
    "services/api_service/app/main.py",
    "services/api_service/app/ml_client.py",
    "services/ml_service/app/main.py",
    "services/camera_v11/monitoring_telemetry_ipc_v1.py",
    "services/frontend/app/api_client.py",
    "services/frontend/sentinel_v1/monitoring_client_v1.py",
    "scripts/run_six_camera_v13_wall_candidate.py",
    "scripts/run_verified_candidate_synthetic_six.py",
    "scripts/build_six_camera_v13_wall_candidate.py",
    "tests/test_app_contract.py",
    "tests/conftest.py",
    "tests/test_monitoring_services_v1.py",
    "tests/test_six_camera_v13_wall_candidate.py",
    "tests/test_camera_recovery_policy.py",
    "tests/test_source_epoch_meta.py",
    "tests/test_source_reconnect_telemetry.py",
    "tests/test_source_message_generation_contract.py",
    "tests/test_tracker_output_dedupe.py",
    "tests/test_cross_id_context_duplicate.py",
    "tests/test_calibration_recording_contract.py",
    "tests/native/test_camera_recovery_policy.cpp",
    "tests/native/test_source_epoch_meta.cpp",
    "tests/native/test_source_reconnect_telemetry.cpp",
    "tests/native/test_source_message_generation.cpp",
    "tests/native/test_calibration_recording_failures.cpp",
    "tests/native/test_tracker_output_dedupe.cpp",
    "tests/native/test_cross_id_context_duplicate.cpp",
    "scripts/test_calibration_recording_offline.py",
    "scripts/validate_calibration_recording.py",
    "scripts/run_verified_candidate_two_source_replay.py",
    "tests/test_verified_candidate_two_source_replay.py",
    "docs/SURVEILLANCE_APPLICATION_RUNTIME.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"release snapshot cannot contain a symlink: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha256(path)
    return result


def canonical_hash(value: dict[str, str]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_provenance() -> dict:
    """Record the actual checkout baseline and dirty inventory without changing it."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
        text=True, check=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=ROOT, capture_output=True,
        text=True, check=True,
    ).stdout.strip()
    raw = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=ROOT, capture_output=True, check=True,
    ).stdout
    parts = raw.split(b"\0")
    inventory = []
    index = 0
    while index < len(parts) and parts[index]:
        record = os.fsdecode(parts[index])
        status = record[:2]
        path = record[3:]
        # Release evidence is deliberately excluded from the source inventory.
        if path != ".runtime" and not path.startswith(".runtime/"):
            entry = {"status": status, "path": path}
            if "R" in status or "C" in status:
                index += 1
                if index < len(parts) and parts[index]:
                    entry["previous_path"] = os.fsdecode(parts[index])
            inventory.append(entry)
        index += 1
    return {"baseline_commit": head, "branch": branch, "working_tree_files": inventory}


def make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def compile_command(source: Path, output: Path, defines: list[str]) -> list[str]:
    define_args = " ".join(f"-D{name}" for name in defines)
    shell = (
        f"g++ -std=c++17 -O2 {define_args} "
        "-I/host-cuda-include -I/opt/nvidia/deepstream/deepstream/sources/includes "
        "$(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0) "
        f"/src/main.cpp -o /work/{output.as_posix()} "
        "$(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0) "
        "-L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -lnvds_meta "
        "-lnvbufsurface -lnvbufsurftransform -lnvds_infer -lnvinfer "
        "-lnvinfer_plugin -L/usr/local/cuda/lib64 -lcudart -ldl -pthread "
        "-Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib"
    )
    return [
        "docker", "run", "--rm", "--pull=never", "--network=none", "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{source}:/src:ro", "-v", f"{OUT}:/work",
        "-v", "/usr/local/cuda/include:/host-cuda-include:ro",
        IMAGE, "bash", "-lc", shell,
    ]


def compile_one(source: Path, output_relative: str, defines: list[str], log_name: str) -> str:
    output = OUT / output_relative
    output.parent.mkdir(parents=True, exist_ok=True)
    command = compile_command(source, Path(output_relative), defines)
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            timeout=900, check=True)
    (OUT / "build-logs" / log_name).write_text(
        json.dumps({"argv": command, "stdout": result.stdout, "stderr": result.stderr},
                   indent=2) + "\n", encoding="utf-8")
    return sha256(output)


def copy_application_snapshot() -> tuple[dict[str, str], str]:
    target_root = OUT / "source/application"
    target_root.mkdir(parents=True)
    expected: dict[str, str] = {}
    for relative in APP_FILES:
        source = APPLICATION_SOURCE_ROOT / relative
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"required release source is missing or unsafe: {source}")
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        expected[relative] = sha256(source)
        if sha256(target) != expected[relative]:
            raise RuntimeError(f"application source changed while snapshotting: {relative}")
    if expected != tree_hashes(target_root):
        raise RuntimeError("application source snapshot is not an exact file-set copy")
    tree_digest = canonical_hash(expected)
    manifest_path = OUT / "source/application-manifest.json"
    manifest_path.write_text(json.dumps({
        "schema": "six-camera-application-source-snapshot-v1",
        "files_sha256": expected,
        "tree_sha256": tree_digest,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return expected, tree_digest


def main() -> dict:
    if not OUT.is_relative_to(ROOT / ".runtime"):
        raise ValueError("release output must be under .runtime")
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite release output: {OUT}")
    if not NATIVE_SOURCE.is_dir() or not APPLICATION_SOURCE_ROOT.is_dir():
        raise FileNotFoundError("native source or application source root is missing")
    validation_path = None
    validation_payload = None
    if VALIDATION_ATTESTATION:
        validation_path = Path(VALIDATION_ATTESTATION).resolve()
        if not validation_path.is_relative_to(ROOT / ".runtime") or not validation_path.is_file():
            raise ValueError("offline validation attestation must be an existing file under .runtime")
        validation_payload = json.loads(validation_path.read_text(encoding="utf-8"))
        if (validation_payload.get("schema_version") != 1
                or not isinstance(validation_payload.get("evidence"), list)
                or not validation_payload["evidence"]):
            raise RuntimeError("offline validation attestation schema is invalid")
        if validation_payload.get("status") not in {
                "OFFLINE_VALIDATED", "OFFLINE_VALIDATED_WITH_LIMITATIONS"}:
            raise RuntimeError("offline validation attestation status is invalid")
        unresolved = validation_payload.get("unresolved_cases", [])
        if (not isinstance(unresolved, list)
                or any(not isinstance(item, str) or not item.strip() for item in unresolved)):
            raise RuntimeError("offline validation unresolved cases must be a list of non-empty strings")
        seen_evidence_paths = set()
        for row in validation_payload["evidence"]:
            relative = row.get("path") if isinstance(row, dict) else None
            expected = row.get("sha256") if isinstance(row, dict) else None
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise RuntimeError("offline validation evidence entry lacks path/hash")
            if relative in seen_evidence_paths:
                raise RuntimeError(f"duplicate offline validation evidence path: {relative}")
            seen_evidence_paths.add(relative)
            evidence_path = (ROOT / relative).resolve()
            if not evidence_path.is_relative_to(ROOT / ".runtime") or not evidence_path.is_file():
                raise RuntimeError(f"offline validation evidence is missing or outside .runtime: {relative}")
            if sha256(evidence_path) != expected:
                raise RuntimeError(f"offline validation evidence hash mismatch: {relative}")
    if not Path("/usr/local/cuda/include/cuda_runtime_api.h").is_file():
        raise RuntimeError("host CUDA development headers are unavailable for the pinned DS9.1 build")

    from scripts.run_six_camera_v13_wall_candidate import validate_candidate

    parent_manifest = validate_candidate(PARENT)
    parent_binary = PARENT / "bin/cam-six-yolo-local6-posttracker"
    if sha256(parent_binary) != EXPECTED_PARENT_BINARY:
        raise RuntimeError("preserved production r2 binary hash differs from the authorized release")
    if parent_manifest.get("deepstream_image") != IMAGE:
        raise RuntimeError("parent candidate does not use the pinned DeepStream 9.1 image")

    started = time.monotonic()
    OUT.mkdir(parents=True)
    for directory in ("bin", "config", "models", "build-logs", "verification"):
        shutil.copytree(PARENT / directory, OUT / directory)
    native_target = OUT / "source/native"
    shutil.copytree(NATIVE_SOURCE, native_target)
    native_hashes = tree_hashes(native_target)
    if not native_hashes or native_hashes != tree_hashes(NATIVE_SOURCE):
        raise RuntimeError("native source changed while taking the release snapshot")
    native_tree_digest = canonical_hash(native_hashes)
    native_manifest_path = OUT / "source/source-manifest.json"
    native_manifest_path.write_text(json.dumps({
        "schema": "six-camera-native-source-snapshot-v1",
        "baseline_release": str(PARENT.relative_to(ROOT)),
        "baseline_binary_sha256": EXPECTED_PARENT_BINARY,
        "source_sha256": native_hashes,
        "tree_sha256": native_tree_digest,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    app_hashes, app_tree_digest = copy_application_snapshot()
    shutil.copyfile(Path(__file__), OUT / "source/reliability-build-recipe.py")
    recipe_hash = sha256(OUT / "source/reliability-build-recipe.py")

    base_config_hashes = tree_hashes(PARENT / "config")
    base_model_hashes = tree_hashes(PARENT / "models")
    if tree_hashes(OUT / "config") != base_config_hashes or tree_hashes(OUT / "models") != base_model_hashes:
        raise RuntimeError("candidate copy changed detector/tracker/model artifacts")

    provenance = parent_manifest.get("build_provenance", {})
    defines = provenance.get("compile_defines_live")
    direct_defines = provenance.get("compile_defines_direct_file")
    if not isinstance(defines, list) or not isinstance(direct_defines, list):
        raise RuntimeError("parent manifest lacks its exact native compile definitions")
    make_read_only(native_target)
    make_read_only(OUT / "source/application")

    image_id = subprocess.run(
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if image_id != IMAGE.removeprefix("nvcr.io/nvidia/deepstream@"):
        raise RuntimeError("local DS9.1 build image ID differs from the pinned digest")

    live_relative = "bin/cam-six-yolo-local6-posttracker"
    direct_relative = "bin/cam-six-yolo-local6-direct-gatea"
    live_hash = compile_one(native_target, live_relative, defines, "compile-live.json")
    direct_hash = compile_one(native_target, direct_relative, direct_defines, "compile-direct-file.json")
    live_repro_hash = compile_one(native_target, "verification/live-repro-build", defines,
                                  "compile-live-repro.json")
    direct_repro_hash = compile_one(native_target, "verification/direct-repro-build", direct_defines,
                                    "compile-direct-repro.json")
    if live_hash != live_repro_hash or direct_hash != direct_repro_hash:
        raise RuntimeError("two builds from the immutable source snapshot were not byte-identical")

    offline_validation = None
    if validation_payload is not None and validation_path is not None:
        expected_binding = {
            "candidate_binary_sha256": live_hash,
            "candidate_direct_file_binary_sha256": direct_hash,
            "native_source_tree_sha256": native_tree_digest,
            "application_source_tree_sha256": app_tree_digest,
        }
        if any(validation_payload.get(key) != value for key, value in expected_binding.items()):
            raise RuntimeError("offline validation attestation is not bound to the exact candidate sources/binaries")
        attestation_copy = OUT / "verification/offline-validation-attestation.json"
        shutil.copyfile(validation_path, attestation_copy)
        offline_validation = {
            "status": validation_payload.get("status"),
            "attestation": "verification/offline-validation-attestation.json",
            "attestation_sha256": sha256(attestation_copy),
            "evidence_file_count": len(validation_payload["evidence"]),
            "unresolved_cases": validation_payload.get("unresolved_cases", []),
        }

    native_manifest_hash = sha256(native_manifest_path)
    app_manifest_path = OUT / "source/application-manifest.json"
    app_manifest_hash = sha256(app_manifest_path)
    release_source_hashes = {
        **{f"native/{key}": value for key, value in native_hashes.items()},
        **{f"application/{key}": value for key, value in app_hashes.items()},
        "reliability-build-recipe.py": recipe_hash,
    }
    build_manifest = dict(parent_manifest)
    build_manifest.update({
        "candidate_label": "R13 offline reliability candidate over verified r2 profile; per-source recovery, generation-tagged source messages, immutable epochs, asynchronous recorder metadata, truthful readiness, and source-PTS-bounded recorder duration",
        "binary_sha256": live_hash,
        "direct_file_binary_sha256": direct_hash,
        "source_sha256": native_hashes,
        "source_snapshot": {
            "directory": "source/native",
            "manifest": "source/source-manifest.json",
            "manifest_sha256": native_manifest_hash,
            "tree_sha256": native_tree_digest,
        },
        "application_snapshot": {
            "directory": "source/application",
            "manifest": "source/application-manifest.json",
            "manifest_sha256": app_manifest_hash,
            "files_sha256": app_hashes,
            "tree_sha256": app_tree_digest,
        },
        "release_source_sha256": release_source_hashes,
        "release_source_tree_sha256": canonical_hash(release_source_hashes),
        "git_provenance": git_provenance(),
        "dependency_manifest": {
            "deepstream_image": IMAGE,
            "deepstream_image_id": image_id,
            "toolchain": provenance.get("toolchain"),
            "runtime_dependencies": parent_manifest.get("runtime", {}),
            "native_link_libraries": [
                "gstreamer-1.0", "gstreamer-app-1.0", "gstreamer-video-1.0",
                "nvdsgst_meta", "nvds_meta", "nvbufsurface",
                "nvbufsurftransform", "nvds_infer", "nvinfer",
                "nvinfer_plugin", "cudart", "dl", "pthread",
            ],
            "detector_config_sha256": parent_manifest.get("config_sha256", {}),
            "model_and_parser_sha256": parent_manifest.get("model_sha256", {}),
        },
        "offline_validation": offline_validation,
        "recovery": {
            **parent_manifest.get("recovery", {}),
            "candidate_binary_sha256": live_hash,
            "status": "OFFLINE_CANDIDATE_NOT_LIVE_VALIDATED",
            "validated_cycles": 0,
            "required_environment": "V11_SOURCE_STALL_RECOVERY=1",
        },
        "build_provenance": {
            **provenance,
            "parent_candidate": str(PARENT.relative_to(ROOT)),
            "parent_binary_sha256": EXPECTED_PARENT_BINARY,
            "deepstream_image": IMAGE,
            "deepstream_image_id": image_id,
            "build_recipe_sha256": recipe_hash,
            "source_snapshot_tree_sha256": native_tree_digest,
            "application_snapshot_tree_sha256": app_tree_digest,
            "compile_reproducibility": {
                "live_binary_sha256": live_hash,
                "live_reproduction_sha256": live_repro_hash,
                "live_byte_identical": live_hash == live_repro_hash,
                "direct_binary_sha256": direct_hash,
                "direct_reproduction_sha256": direct_repro_hash,
                "direct_byte_identical": direct_hash == direct_repro_hash,
            },
            "runtime_profile_unchanged": True,
            "changes": [
                "camera-local serialized recovery with bounded backoff and auth quarantine; process-level fatal remains separate",
            "generation-tagged source-child bus messages; retired-source errors and EOS cannot become shared-pipeline fatal events",
            "effective nvurisrcbin reconnect intervals read back at runtime; retry count recorded but not treated as the disable switch",
                "per-decoded-buffer immutable source generation/tracking epoch metadata; stale cross-epoch PGIE/tracker publications fail closed",
                "bounded dedicated recorder metadata writer; saturation, disk failure, or row loss marks evidence REVIEW_REQUIRED",
                "recording branches stop at the earlier of shared wall-duration or per-camera source-PTS duration, preserving included source PTS",
                "loopback API/ML defaults and bearer authentication for explicitly configured non-loopback access",
                "health distinguishes process liveness from per-source and six-camera readiness",
            ],
            "production_startup_authorized": False,
        },
    })
    manifest_path = OUT / "build-manifest.json"
    manifest_path.write_text(json.dumps(build_manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    final_hash = sha256(OUT / live_relative)
    direct_final_hash = sha256(OUT / direct_relative)
    if final_hash != build_manifest["binary_sha256"] or direct_final_hash != build_manifest["direct_file_binary_sha256"]:
        raise RuntimeError("candidate binary differs from release manifest")
    return {
        "status": "OFFLINE_BUILD_REPRODUCIBLE",
        "candidate": str(OUT.relative_to(ROOT)),
        "binary_sha256": final_hash,
        "direct_file_binary_sha256": direct_final_hash,
        "source_tree_sha256": native_tree_digest,
        "application_tree_sha256": app_tree_digest,
        "manifest_sha256": sha256(manifest_path),
        "parent_binary_sha256": EXPECTED_PARENT_BINARY,
        "deepstream_image": IMAGE,
        "runtime_profile_unchanged": True,
        "live_reproducible": live_hash == live_repro_hash,
        "direct_file_reproducible": direct_hash == direct_repro_hash,
        "elapsed_sec": round(time.monotonic() - started, 3),
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
