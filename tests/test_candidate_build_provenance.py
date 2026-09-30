from pathlib import Path

import pytest

from scripts.dev_room_mv3dt.build_staging_candidate import build, command, input_hashes


def test_build_command_has_read_only_inputs_no_network_or_production_output(tmp_path):
    repo, sdk, output = tmp_path / "repo", tmp_path / "sdk", tmp_path / "candidate"
    image = "nvcr.io/nvidia/deepstream@sha256:" + "a" * 64
    cmd = command(repo, sdk, output, image)
    assert "--network=none" in cmd
    assert "--pull=never" in cmd
    assert "--privileged" not in cmd
    mounts = [cmd[i + 1] for i, token in enumerate(cmd) if token == "-v"]
    assert len(mounts) == 6
    assert all(mount.endswith(":ro") for mount in mounts[:-1])
    assert mounts[-1] == f"{output}:/workspace/bin"
    assert "/experiments/deepstream/" not in " ".join(cmd)


def test_unpinned_image_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        command(tmp_path, tmp_path, tmp_path, "nvcr.io/nvidia/deepstream:latest")


def test_non_runtime_output_is_rejected_before_any_build_or_write(tmp_path):
    output = tmp_path / "installed-production"
    with pytest.raises(ValueError):
        build(tmp_path, tmp_path, output, Path("not-read.json"))
    assert not output.exists()


def test_inventory_detects_header_changes_and_ignores_generated_binaries(tmp_path):
    repo, sdk = tmp_path / "repo", tmp_path / "sdk"
    for path in (repo / "services/mv3dt_room/native", sdk / "src/apps/common",
                 sdk / "src/apps/sample_apps/deepstream-app", sdk / "includes",
                 sdk / "src/apps/sample_apps/deepstream-test5", repo / "scripts/dev_room_mv3dt"):
        path.mkdir(parents=True)
    header = sdk / "includes/nvdsmeta.h"
    header.write_text("header v1")
    (sdk / "src/apps/sample_apps/deepstream-test5/deepstream_utc.c").write_text("source")
    (repo / "scripts/dev_room_mv3dt/build_native_binary.sh").write_text("build")
    before = input_hashes(repo, sdk)
    (repo / "services/mv3dt_room/native/generated.engine").write_bytes(b"not a compile input")
    assert input_hashes(repo, sdk) == before
    header.write_text("header v2")
    assert input_hashes(repo, sdk) != before
