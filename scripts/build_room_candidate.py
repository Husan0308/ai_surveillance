"""Build a separate hash-pinned native candidate, never an installed asset.

Reuses the existing DeepStream 9.1 build script and its read-only SDK inputs.
Staged diagnostics permit detector-only operation and log unchanged rectangles;
the tracker-enabled path executes the existing bbox probe attachment otherwise.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from scripts.dev_room_mv3dt.verify_validated_assets import sha256

ROOT = Path(__file__).resolve().parents[1]
SDK = Path("/home/apsidal/nvidia/DeepStream")
IMAGE = "nvcr.io/nvidia/deepstream@sha256:fd31f5b44ababdbdee8cd397a375e888191b49e402ac237254a4cdc239130f5b"
BINARY = "deepstream-test5-pn263-global-id-proto"
OLD = "    if (!pn263_bbox_correction_attach ("
NEW = "    if (appCtx[i]->config.tracker_config.enable && !pn263_bbox_correction_attach ("
AUDIT_OLD = '''          obj_meta->confidence, obj_meta->tracker_confidence);
'''
AUDIT_NEW = '''          obj_meta->confidence, obj_meta->tracker_confidence,
          obj_meta->rect_params.left, obj_meta->rect_params.top,
          obj_meta->rect_params.width, obj_meta->rect_params.height);
'''


def detector_only_source(source: str) -> str:
    if source.count(OLD) != 1:
        raise ValueError("detector-only staging guard no longer matches exactly one call")
    return source.replace(OLD, NEW, 1)


def diagnostic_source(source: str) -> str:
    source = detector_only_source(source)
    old = '\\"tracker_confidence\\":%.6f}",'
    new = '\\"tracker_confidence\\":%.6f,\\"bbox\\":[%.3f,%.3f,%.3f,%.3f]}",'
    if source.count(old) != 1 or source.count(AUDIT_OLD) != 1:
        raise ValueError("bbox audit diagnostic must match exactly one frame logger")
    return source.replace(old, new, 1).replace(AUDIT_OLD, AUDIT_NEW, 1)


def input_hashes(native: Path) -> dict[str, str]:
    roots = {"native": native, "sdk-common": SDK / "src/apps/common",
             "sdk-app": SDK / "src/apps/sample_apps/deepstream-app", "sdk-includes": SDK / "includes"}
    result = {}
    for label, root in roots.items():
        if not root.is_dir():
            raise ValueError(f"missing read-only input {root}")
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".c", ".cpp", ".h", ".hpp"}:
                result[f"{label}/{path.relative_to(root)}"] = sha256(path)
    result["sdk/deepstream_utc.c"] = sha256(SDK / "src/apps/sample_apps/deepstream-test5/deepstream_utc.c")
    result["build_native_binary.sh"] = sha256(ROOT / "scripts/dev_room_mv3dt/build_native_binary.sh")
    return result


def build(output: Path) -> dict:
    output = output.resolve()
    output.relative_to(ROOT / ".runtime")
    output.mkdir(parents=True, exist_ok=False)
    base = input_hashes(ROOT / "services/mv3dt_room/native")
    native = output / "native"
    shutil.copytree(ROOT / "services/mv3dt_room/native", native)
    main = native / "deepstream_test5_app_main.c"
    main.write_text(diagnostic_source(main.read_text()))
    inputs = input_hashes(native)
    if [k for k in base if base[k] != inputs[k]] != ["native/deepstream_test5_app_main.c"]:
        raise RuntimeError("unexpected staged native delta")
    binary_dir = output / "bin"
    binary_dir.mkdir()
    mounts = [(native, "/workspace/room-native", True),
        (SDK / "src/apps/sample_apps/deepstream-test5", "/workspace/old-source", True),
        (SDK / "src", "/workspace/ds-src", True), (SDK / "includes", "/workspace/ds-includes", True),
        (ROOT / "scripts/dev_room_mv3dt/build_native_binary.sh", "/workspace/build-native.sh", True),
        (binary_dir, "/workspace/bin", False)]
    command = ["docker", "run", "--rm", "--pull=never", "--network=none", "--user", f"{os.getuid()}:{os.getgid()}"]
    for source, destination, readonly in mounts:
        command.extend(["-v", f"{source}:{destination}" + (":ro" if readonly else "")])
    command.extend(["--entrypoint", "bash", IMAGE, "/workspace/build-native.sh"])
    with (output / "build.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    if input_hashes(native) != inputs or input_hashes(ROOT / "services/mv3dt_room/native") != base:
        raise RuntimeError("build inputs changed during compilation")
    binary = binary_dir / BINARY
    notes = subprocess.check_output(["readelf", "-n", str(binary)], text=True)
    result = {"status": "EXPERIMENTAL_NOT_ACCEPTED", "built_at": datetime.now(timezone.utc).isoformat(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "base_source_sha256": base["native/deepstream_test5_app_main.c"], "source_sha256": sha256(main),
        "binary": str(binary), "binary_sha256": sha256(binary),
        "elf_build_id": re.search(r"Build ID: ([0-9a-f]+)", notes).group(1),
        "toolchain_image": IMAGE, "base_inputs": base, "input_sha256": inputs,
        "command": command, "production_accepted": False,
        "staged_source_delta": {"before": OLD, "after": NEW,
            "additional": "environment-gated frame audit now includes unmodified object rect_params"},
        "diagnostic_source_override": True}
    (output / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify_build(path: Path) -> dict:
    record = json.loads(path.read_text())
    if record["status"] != "EXPERIMENTAL_NOT_ACCEPTED" or record["production_accepted"]:
        raise ValueError("candidate must remain explicitly experimental")
    if input_hashes(ROOT / "services/mv3dt_room/native") != record["base_inputs"]:
        raise ValueError("candidate no longer matches the frozen checkout/SDK")
    if input_hashes(path.parent / "native") != record["input_sha256"]:
        raise ValueError("staged compile inputs changed")
    if sha256(Path(record["binary"])) != record["binary_sha256"]:
        raise ValueError("candidate binary hash mismatch")
    if sha256(path.parent / "native/deepstream_test5_app_main.c") != record["source_sha256"]:
        raise ValueError("candidate source hash mismatch")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.output)
    print(json.dumps({k: result[k] for k in ("status", "base_source_sha256", "source_sha256", "binary_sha256", "elf_build_id")}), flush=True)


if __name__ == "__main__":
    main()
