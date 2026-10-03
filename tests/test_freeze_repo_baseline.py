import pytest

from scripts.freeze_repo_baseline import file_record, verify


def test_secret_contents_are_never_in_file_record(tmp_path):
    path = tmp_path / ".env"
    path.write_text("CAMERA_PASSWORD=private-value\n")
    row = file_record(path)
    assert set(row) == {"path", "exists", "sha256", "bytes"}
    assert "private-value" not in str(row)
    assert len(row["sha256"]) == 64


def test_verification_fails_on_protected_hash_drift(tmp_path, monkeypatch):
    from scripts import freeze_repo_baseline as audit
    monkeypatch.setattr(audit, "freeze_refs", lambda root: "unchanged-ref")
    path = tmp_path / "protected.yml"
    path.write_text("original")
    report = {"root": str(tmp_path), "freeze_refs": "unchanged-ref",
              "protected_hashes": {"protected.yml": file_record(path)}, "artifact_hashes": {}}
    assert verify(report)["ok"]
    path.write_text("different")
    result = verify(report)
    assert not result["ok"]
    assert result["changed"][0]["file"] == "protected.yml"
    assert not result["production_asset_gate_pass_claimed"]


def test_missing_candidate_cannot_appear_without_detected_drift(tmp_path, monkeypatch):
    from scripts import freeze_repo_baseline as audit
    monkeypatch.setattr(audit, "freeze_refs", lambda root: "")
    path = tmp_path / "candidate.bin"
    report = {"root": str(tmp_path), "freeze_refs": "", "protected_hashes": {},
              "artifact_hashes": {str(path): file_record(path)}}
    assert verify(report)["ok"]
    path.write_bytes(b"new candidate")
    assert not verify(report)["ok"]


def test_existing_freeze_directory_is_never_overwritten(tmp_path):
    from scripts.freeze_repo_baseline import capture
    output = tmp_path / "old-evidence"
    output.mkdir()
    saved = output / "baseline.json"
    saved.write_text("existing evidence")
    with pytest.raises(FileExistsError):
        capture(tmp_path, output, [])
    assert saved.read_text() == "existing evidence"


def test_frozen_ref_change_is_a_failure(tmp_path, monkeypatch):
    from scripts import freeze_repo_baseline as audit
    monkeypatch.setattr(audit, "freeze_refs", lambda root: "new-tip")
    report = {"root": str(tmp_path), "freeze_refs": "old-tip",
              "protected_hashes": {}, "artifact_hashes": {}}
    assert not verify(report)["ok"]
