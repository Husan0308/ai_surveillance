"""Exact staging pins must not turn into production asset acceptance."""
import json
from pathlib import Path
import shutil

import pytest
import yaml

from scripts.dev_room_mv3dt import run_room_pair as runner
from scripts.dev_room_mv3dt.verify_validated_assets import load_staging_profile, sha256, verify


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def asset_repo(tmp_path):
    profile = tmp_path / "config/mv3dt_dev_room"
    shutil.copytree(ROOT / "config/mv3dt_dev_room", profile)
    manifest = json.loads((ROOT / "services/mv3dt_room/asset_manifest.json").read_text())
    for relative in manifest["validated_assets"]:
        target = tmp_path / "services/mv3dt_room" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "services/mv3dt_room" / relative, target)
        manifest["validated_assets"][relative] = sha256(target)
    binary = tmp_path / "accepted-binary"
    binary.write_bytes(b"accepted native binary fixture")
    manifest["accepted_binary_sha256"] = sha256(binary)
    (tmp_path / "services/mv3dt_room/asset_manifest.json").write_text(json.dumps(manifest))
    runtime = yaml.safe_load((profile / "runtime.yaml").read_text())
    runtime["runtime"]["binary"] = "accepted-binary"
    (profile / "runtime.yaml").write_text(yaml.safe_dump(runtime))
    return tmp_path


def candidate(repo):
    native = repo / "services/mv3dt_room/native/deepstream_test5_app_main.c"
    native.write_text(native.read_text() + "\n/* candidate source fixture */\n")
    binary = repo / "candidate-binary"
    binary.write_bytes(b"different staged native binary fixture")
    return binary, {"candidate_source_sha256": sha256(native),
                    "candidate_binary_sha256": sha256(binary)}


def test_bare_validator_checks_runtime_binary_by_default(asset_repo):
    assert verify(asset_repo)["production_accepted"] is True
    (asset_repo / "accepted-binary").write_bytes(b"unrelated installed binary")
    result = verify(asset_repo)
    assert result["ok"] is False
    assert any("binary hash mismatch" in error for error in result["errors"])


def test_exact_staging_is_checked_but_never_accepted(asset_repo):
    binary, pins = candidate(asset_repo)
    manifest = asset_repo / "services/mv3dt_room/asset_manifest.json"
    before = manifest.read_bytes()
    result = verify(asset_repo, binary, **pins)
    assert result["ok"] is True
    assert result["staging_hash_override"] is True
    assert result["production_accepted"] is False
    assert result["asset_role"] == "staging_pending_live_recall"
    assert manifest.read_bytes() == before
    assert verify(asset_repo, binary)["ok"] is False


@pytest.mark.parametrize("bad", ["", "*", "6e960", "g" * 64, "A" * 64])
def test_staging_pins_cannot_be_empty_wildcard_or_malformed(asset_repo, bad):
    binary, pins = candidate(asset_repo)
    with pytest.raises(ValueError):
        verify(asset_repo, binary, **{**pins, "candidate_source_sha256": bad})


def test_staging_requires_both_pins_and_explicit_binary(asset_repo):
    binary, pins = candidate(asset_repo)
    with pytest.raises(ValueError):
        verify(asset_repo, binary, candidate_source_sha256=pins["candidate_source_sha256"])
    with pytest.raises(ValueError):
        verify(asset_repo, None, **pins)


@pytest.mark.parametrize("field", ["candidate_source_sha256", "candidate_binary_sha256"])
def test_wrong_exact_pin_fails(asset_repo, field):
    binary, pins = candidate(asset_repo)
    result = verify(asset_repo, binary, **{**pins, field: "0" * 64})
    assert result["ok"] is False


@pytest.mark.parametrize("path", ["services/mv3dt_room/global_identity_manager.py",
                                 "services/mv3dt_room/native/bbox_correction.c",
                                 "config/mv3dt_dev_room/config_tracker.yml",
                                 "config/mv3dt_dev_room/config_pgie.txt"])
def test_staging_cannot_bypass_other_frozen_assets(asset_repo, path):
    binary, pins = candidate(asset_repo)
    target = asset_repo / path
    target.write_text(target.read_text() + "\n# deliberate unaccepted change\n")
    assert verify(asset_repo, binary, **pins)["ok"] is False


def test_diagnostics_do_not_bypass_unpinned_source_or_binary(asset_repo, monkeypatch):
    binary, pins = candidate(asset_repo)
    monkeypatch.setattr(runner, "ROOT", asset_repo)
    monkeypatch.setenv("MV3DT_BINARY", str(binary))
    monkeypatch.setenv("MV3DT_FRAME_AUDIT_LOG", "1")
    monkeypatch.setenv("MV3DT_SOURCE_HEALTH_DIR", "1")
    monkeypatch.delenv("MV3DT_DIAGNOSTIC_SOURCE_SHA256", raising=False)
    monkeypatch.delenv("MV3DT_DIAGNOSTIC_BINARY_SHA256", raising=False)
    assert runner.check_runtime_assets({"runtime": {"binary": "accepted-binary"}})["ok"] is False
    monkeypatch.setenv("MV3DT_DIAGNOSTIC_SOURCE_SHA256", pins["candidate_source_sha256"])
    monkeypatch.setenv("MV3DT_DIAGNOSTIC_BINARY_SHA256", pins["candidate_binary_sha256"])
    assert runner.check_runtime_assets({"runtime": {"binary": "accepted-binary"}})["ok"] is True


def test_invalid_assets_fail_before_starting_sources(asset_repo, monkeypatch):
    binary, _ = candidate(asset_repo)
    monkeypatch.setattr(runner, "ROOT", asset_repo)
    monkeypatch.setattr(runner, "load_profile", lambda: {"runtime": {"binary": str(binary)}})
    monkeypatch.delenv("MV3DT_BINARY", raising=False)
    monkeypatch.delenv("MV3DT_DIAGNOSTIC_SOURCE_SHA256", raising=False)
    monkeypatch.delenv("MV3DT_DIAGNOSTIC_BINARY_SHA256", raising=False)
    monkeypatch.setattr(runner, "camera_uris", lambda: pytest.fail("opened sources before asset validation"))
    with pytest.raises(SystemExit):
        runner.run("live", 10, True)
    assert not (asset_repo / ".runtime").exists()


def test_staging_profile_explicitly_remains_pending(asset_repo, monkeypatch):
    binary, pins = candidate(asset_repo)
    path = asset_repo / "staging.json"
    data = {"binary": binary.name, "source_sha256": pins["candidate_source_sha256"],
            "binary_sha256": pins["candidate_binary_sha256"], "status": "staged_pending_live_recall",
            "toolchain_image": "nvcr.io/nvidia/deepstream@sha256:" + "a" * 64}
    path.write_text(json.dumps(data))
    monkeypatch.setattr(runner, "ROOT", asset_repo)
    for name in ("MV3DT_BINARY", "MV3DT_DIAGNOSTIC_SOURCE_SHA256", "MV3DT_DIAGNOSTIC_BINARY_SHA256"):
        monkeypatch.delenv(name, raising=False)
    assert runner.check_runtime_assets({}, path)["ok"] is True
    assert runner.check_runtime_assets({}, path)["deepstream_image"] == data["toolchain_image"]
    monkeypatch.setenv("MV3DT_BINARY", str(binary))
    with pytest.raises(ValueError):
        runner.check_runtime_assets({}, path)
    data["status"] = "production_accepted"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_staging_profile(asset_repo, path)


def test_repository_candidate_is_not_confused_with_previous_candidates():
    data = json.loads((ROOT / "config/mv3dt_dev_room/gate3j_candidate.json").read_text())
    assert data["status"] == "staged_pending_live_recall"
    assert data["validation_revision"] == "8a4769b8f9f2ddbc70f449b8dbd23bc0b511ee54"
    assert data["source_sha256"] == sha256(ROOT / "services/mv3dt_room/native/deepstream_test5_app_main.c")
    assert data["binary_sha256"] == "6e9604d1aa31e2473303edcc2ef05f3e7812989361b8aad98de703a56a3c8817"


def test_staging_profile_pins_additional_inputs_without_rebaselining(asset_repo):
    binary, pins = candidate(asset_repo)
    target = asset_repo / "config/mv3dt_dev_room/config_pgie.txt"
    path = asset_repo / "candidate.json"
    path.write_text(json.dumps({"status": "staged_pending_live_recall", "binary": str(binary),
        "source_sha256": pins["candidate_source_sha256"], "binary_sha256": pins["candidate_binary_sha256"],
        "toolchain_image": "nvcr.io/nvidia/deepstream@sha256:" + "a" * 64,
        "frozen_inputs_sha256": {"config/mv3dt_dev_room/config_pgie.txt": sha256(target)}}))
    assert load_staging_profile(asset_repo, path)["binary"] == binary
    target.write_text(target.read_text() + "\n# unvalidated input\n")
    with pytest.raises(ValueError, match="frozen input hash mismatch"):
        load_staging_profile(asset_repo, path)


@pytest.mark.parametrize("input_path", ["../private.env", "/etc/passwd"])
def test_staging_input_inventory_cannot_escape_repo(asset_repo, input_path):
    binary, pins = candidate(asset_repo)
    path = asset_repo / "candidate.json"
    path.write_text(json.dumps({"status": "staged_pending_live_recall", "binary": str(binary),
        "source_sha256": pins["candidate_source_sha256"], "binary_sha256": pins["candidate_binary_sha256"],
        "toolchain_image": "nvcr.io/nvidia/deepstream@sha256:" + "a" * 64,
        "frozen_inputs_sha256": {input_path: "a" * 64}}))
    with pytest.raises(ValueError, match="repository config/service paths"):
        load_staging_profile(asset_repo, path)
