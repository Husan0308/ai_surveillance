from pathlib import Path

from scripts import run_entrance_calibration_r6 as runner


def test_r6_runner_uses_the_built_eos_fix_and_protected_profile():
    manifest = runner.verify_candidate()
    assert manifest["binary_sha256"] == runner.R6_SHA256
    assert manifest["model_sha256"]["yolo26m_raw_otm_b6_fp16.engine"] == runner.ENGINE_SHA256
    assert manifest["tracker_config_sha256"] == runner.TRACKER_SHA256
    assert manifest["frozen_v13_reference_sha256"] == runner.V13_SHA256
    assert Path(runner.CANDIDATE).is_relative_to(runner.ROOT / ".runtime")


def test_runtime_command_does_not_modify_amc_or_add_rtsp_sources():
    source = (runner.ROOT / "scripts/run_entrance_calibration_r6.py").read_text()
    assert "port_listener_signature(8000)" in source
    assert 'environment["V11_CALIBRATION_RECORDING_ENABLED"] = "1"' in source
    assert "scripts/record_entrance_calibration.sh" in source
    assert "nvurisrcbin" not in source
    assert "SIGKILL" not in source


def test_old_owner_restart_is_pinned_to_known_r5_provenance():
    assert runner.R5_SHA256 == "54edebdb4c2764b819d57014bafab17549ed94f080e0fa43c7bb3cd5cf080604"
    source = (runner.ROOT / "scripts/run_entrance_calibration_r6.py").read_text()
    assert "active stack is not the known r5 owner; refusing to stop it" in source
    assert "no forced kill was attempted" in source
