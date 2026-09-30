#!/usr/bin/env python3
"""Reproduce the pinned Gate 3J candidate without installing any production asset."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

try:
    from .verify_validated_assets import load_staging_profile, sha256
except ImportError:
    from verify_validated_assets import load_staging_profile, sha256

ROOT = Path(__file__).resolve().parents[2]
BINARY_NAME = "deepstream-test5-pn263-global-id-proto"


def input_hashes(repo: Path, sdk: Path) -> dict[str, str]:
    """Inventory repo-native units and external SDK source/header inputs."""
    roots = {
        "native": repo / "services/mv3dt_room/native",
        "sdk-common": sdk / "src/apps/common",
        "sdk-app": sdk / "src/apps/sample_apps/deepstream-app",
        "sdk-includes": sdk / "includes",
    }
    hashes = {}
    for label, root in roots.items():
        if not root.is_dir():
            raise ValueError(f"missing read-only build input: {root}")
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".c", ".cpp", ".h", ".hpp"}:
                hashes[f"{label}/{path.relative_to(root)}"] = sha256(path)
    hashes["sdk/deepstream_utc.c"] = sha256(sdk / "src/apps/sample_apps/deepstream-test5/deepstream_utc.c")
    hashes["build_native_binary.sh"] = sha256(repo / "scripts/dev_room_mv3dt/build_native_binary.sh")
    return hashes


def command(repo: Path, sdk: Path, output: Path, image: str) -> list[str]:
    if not re.fullmatch(r"nvcr\.io/nvidia/deepstream@sha256:[0-9a-f]{64}", image):
        raise ValueError("the DeepStream build image must be digest-pinned")
    mounts = (
        (repo / "services/mv3dt_room/native", "/workspace/room-native", True),
        (sdk / "src/apps/sample_apps/deepstream-test5", "/workspace/old-source", True),
        (sdk / "src", "/workspace/ds-src", True),
        (sdk / "includes", "/workspace/ds-includes", True),
        (repo / "scripts/dev_room_mv3dt/build_native_binary.sh", "/workspace/build-native.sh", True),
        (output, "/workspace/bin", False),
    )
    result = ["docker", "run", "--rm", "--pull=never", "--network=none"]
    for host, guest, readonly in mounts:
        result += ["-v", f"{host}:{guest}" + (":ro" if readonly else "")]
    return result + ["--entrypoint", "bash", image, "/workspace/build-native.sh"]


def build(repo: Path, sdk: Path, output: Path, profile_path: Path) -> dict:
    repo, sdk, output = repo.resolve(), sdk.resolve(), output.resolve()
    if not output.is_relative_to(repo / ".runtime"):
        raise ValueError("candidate output must be a fresh directory under the worktree .runtime")
    staging = load_staging_profile(repo, profile_path)
    source = repo / "services/mv3dt_room/native/deepstream_test5_app_main.c"
    if sha256(source) != staging["source_sha256"]:
        raise ValueError("source does not match the pinned validated candidate")
    # Never overwrite a candidate, backup, or installed binary, even on retry.
    output.mkdir(parents=True, exist_ok=False)
    inputs = input_hashes(repo, sdk)
    image = staging["toolchain_image"]
    image_metadata = json.loads(subprocess.check_output(
        ["docker", "image", "inspect", image], text=True))[0]
    toolchain = subprocess.check_output(
        ["docker", "run", "--rm", "--pull=never", "--network=none", "--entrypoint", "bash", image,
         "-c", "gcc --version; g++ --version; pkg-config --modversion gstreamer-1.0 gstreamer-video-1.0 x11 json-glib-1.0"],
        text=True,
    )
    build_command = command(repo, sdk, output, image)
    with (output / "build.log").open("w") as handle:
        subprocess.run(build_command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    if input_hashes(repo, sdk) != inputs:
        raise RuntimeError("build inputs changed during compilation")
    binary = output / BINARY_NAME
    binary_hash = sha256(binary)
    if binary_hash != staging["binary_sha256"]:
        raise RuntimeError(f"candidate is not byte-identical: {binary_hash}; do not promote")
    notes = subprocess.check_output(["readelf", "-n", str(binary)], text=True)
    build_id = re.search(r"Build ID: ([0-9a-f]+)", notes).group(1)
    if build_id != staging["elf_build_id"]:
        raise RuntimeError("candidate ELF build ID changed")
    git = lambda *args: subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    result = {
        "status": "REPRODUCED_NOT_PROMOTED", "source_sha256": sha256(source),
        "binary_sha256": binary_hash, "binary": str(binary), "elf_build_id": build_id,
        "revision": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
        "worktree_status": git("status", "--short"),
        "validation_revision": staging["validation_revision"],
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "toolchain_image": image, "toolchain_image_id": image_metadata["Id"],
        "toolchain_versions": toolchain, "command": build_command,
        "input_sha256": inputs, "production_accepted": False,
    }
    (output / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--sdk", type=Path, default=Path("/home/apsidal/nvidia/DeepStream"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staging-profile", type=Path,
                        default=Path("config/mv3dt_dev_room/gate3j_candidate.json"))
    args = parser.parse_args()
    result = build(args.repo, args.sdk, args.output, args.staging_profile)
    print(json.dumps({key: value for key, value in result.items() if key != "input_sha256"}, indent=2))


if __name__ == "__main__":
    main()
