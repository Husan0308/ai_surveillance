"""Isolate a live-only EOS teardown experiment; frozen sources stay untouched."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from scripts.build_room_candidate import ROOT, SDK, IMAGE, BINARY, input_hashes, diagnostic_source, verify_build
from scripts.dev_room_mv3dt.verify_validated_assets import sha256

EOS = "  gst_element_send_event(appCtx->pipeline.pipeline, gst_event_new_eos());\n  sleep (1);"
LIVE_CLOSE = """  /* Live headless teardown is cancellation, not file finalization. Sending
   * per-source EOS while MV3DT still processes the last uneven batch can strand
   * its peer association worker. Stop both streams through the parent NULL
   * transition instead. URI replay retains the original EOS finalization. */
  gboolean live_close = FALSE;
  for (i = 0; i < config->num_source_sub_bins; i++)
    if (config->multi_source_config[i].type == NV_DS_SOURCE_RTSP)
      live_close = TRUE;
  if (!live_close) {
    gst_element_send_event(appCtx->pipeline.pipeline, gst_event_new_eos());
    sleep (1);
  } else {
    g_print ("F5 experimental live teardown: coordinated NULL without injected EOS\\n");
  }"""


def shutdown_source(source):
    if source.count(EOS) != 1:
        raise ValueError("exact SDK teardown call required")
    return source.replace(EOS, LIVE_CLOSE, 1)


def verify_candidate(path):
    record = verify_build(path)
    override = record.get("sdk_override")
    if override:
        current = SDK / "src/apps/sample_apps/deepstream-app/deepstream_app.c"
        staged = Path(override["path"])
        if sha256(current) != override["original_sha256"] or sha256(staged) != override["sha256"]:
            raise ValueError("SDK teardown override hash mismatch")
        if staged.read_text() != shutdown_source(current.read_text()):
            raise ValueError("only the live shutdown experiment is permitted")
    return record


def build(output):
    output = output.resolve()
    output.relative_to(ROOT / ".runtime")
    output.mkdir(parents=True, exist_ok=False)
    base = input_hashes(ROOT / "services/mv3dt_room/native")
    native = output / "native"
    shutil.copytree(ROOT / "services/mv3dt_room/native", native)
    main = native / "deepstream_test5_app_main.c"
    main.write_text(diagnostic_source(main.read_text()))
    inputs = input_hashes(native)
    original = SDK / "src/apps/sample_apps/deepstream-app/deepstream_app.c"
    override = output / "deepstream_app.c"
    override.write_text(shutdown_source(original.read_text()))
    binary_dir = output / "bin"
    binary_dir.mkdir()
    mounts = [(native, "/workspace/room-native", True),
        (SDK / "src/apps/sample_apps/deepstream-test5", "/workspace/old-source", True),
        (SDK / "src", "/workspace/ds-src", True), (SDK / "includes", "/workspace/ds-includes", True),
        (override, "/workspace/ds-src/apps/sample_apps/deepstream-app/deepstream_app.c", True),
        (ROOT / "scripts/dev_room_mv3dt/build_native_binary.sh", "/workspace/build-native.sh", True),
        (binary_dir, "/workspace/bin", False)]
    command = ["docker", "run", "--rm", "--pull=never", "--network=none", "--user", f"{os.getuid()}:{os.getgid()}"]
    for source, destination, readonly in mounts:
        command.extend(["-v", f"{source}:{destination}" + (":ro" if readonly else "")])
    command.extend(["--entrypoint", "bash", IMAGE, "/workspace/build-native.sh"])
    with (output / "build.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    if input_hashes(native) != inputs or input_hashes(ROOT / "services/mv3dt_room/native") != base:
        raise RuntimeError("compile inputs changed")
    binary = binary_dir / BINARY
    notes = subprocess.check_output(["readelf", "-n", str(binary)], text=True)
    record = {"status": "EXPERIMENTAL_NOT_ACCEPTED", "built_at": datetime.now(timezone.utc).isoformat(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "base_source_sha256": base["native/deepstream_test5_app_main.c"], "source_sha256": sha256(main),
        "binary": str(binary), "binary_sha256": sha256(binary),
        "elf_build_id": re.search(r"Build ID: ([0-9a-f]+)", notes).group(1), "toolchain_image": IMAGE,
        "base_inputs": base, "input_sha256": inputs, "command": command, "production_accepted": False,
        "diagnostic_source_override": True, "sdk_override": {"path": str(override),
            "original_sha256": sha256(original), "sha256": sha256(override),
            "purpose": "live-only coordinated NULL teardown; no injected per-source EOS"}}
    path = output / "build.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return verify_candidate(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = build(args.output)
    print(json.dumps({k: record[k] for k in ("source_sha256", "binary_sha256", "sdk_override")}))


if __name__ == "__main__":
    main()
