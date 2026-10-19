"""CPU-only gate-1 checks for isolated YOLO26m staging.

No camera connections, Docker startup, GPU inference, or production file writes.
"""
import hashlib
from pathlib import Path
import sys

import pytest

from services.mv3dt_staging.live_tracking.yolo_candidate import (
    ASSETS, DEFAULT_MODEL_DIR, FROZEN_PGIE, ROOT, render_batch_two_config, stage_yolo_batch_two,
    validate_yolo_assets,
)
from services.mv3dt_staging.live_tracking import live_bridge


def _fake_tree(tmp_path):
    root = tmp_path / "checkout"
    assets = root / ".runtime" / "models"
    assets.mkdir(parents=True)
    config = root / "config/deepstream/config_infer_primary_yolo26m_raw_otm.txt"
    config.parent.mkdir(parents=True)
    config.write_bytes(FROZEN_PGIE.read_bytes())
    expected = {
        "config_infer_primary_yolo26m_raw_otm.txt": hashlib.sha256(config.read_bytes()).hexdigest()
    }
    for name in ASSETS:
        target = assets / name
        target.write_bytes(("synthetic-" + name).encode())
        expected[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    return root, assets, expected


def test_batch_two_preserves_detector_contract():
    original = FROZEN_PGIE.read_text()
    edited = render_batch_two_config(original)
    assert edited == original.replace("batch-size=6", "batch-size=2", 1)
    assert edited.count("batch-size=2") == 1
    assert "interval=0" in edited
    assert "parse-bbox-func-name=NvDsInferParseYolo26RawPerson" in edited


def test_default_artifact_directory_matches_verified_frozen_profile():
    assert DEFAULT_MODEL_DIR == ROOT / ".runtime/sixcam-approved-bbox-20261006-r12-profile/models"


def test_rejects_nonfrozen_detector_settings():
    baseline = FROZEN_PGIE.read_text()
    with pytest.raises(ValueError, match="contract"):
        render_batch_two_config(baseline.replace("interval=0", "interval=2"))


def test_static_candidate_in_private_output_and_no_source_modification(tmp_path):
    root, assets, expected = _fake_tree(tmp_path)
    source_before = (root / "config/deepstream/config_infer_primary_yolo26m_raw_otm.txt").read_bytes()
    output = root / ".runtime" / "gate1"
    result = stage_yolo_batch_two(assets, output, expected=expected, root=root)
    assert result["batch_size"] == 2
    assert result["model_asset_sha256"] == {k: expected[k] for k in ASSETS}
    assert Path(result["config_file"]).is_file()
    assert Path(result["config_file"]).stat().st_mode & 0o777 == 0o600
    assert (root / "config/deepstream/config_infer_primary_yolo26m_raw_otm.txt").read_bytes() == source_before
    assert "NOT TESTED" in result["validated"]


def test_changed_model_engine_hash_rejected(tmp_path):
    root, assets, expected = _fake_tree(tmp_path)
    (assets / "yolo26m_raw_otm_b6_fp16.engine").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        validate_yolo_assets(assets, expected=expected, root=root)


def test_external_model_directory_rejected(tmp_path):
    root, assets, expected = _fake_tree(tmp_path)
    external = tmp_path / "outside"
    external.mkdir()
    with pytest.raises(ValueError, match=r"\.runtime"):
        validate_yolo_assets(external, expected=expected, root=root)


def test_yolo_mode_rejects_missing_explicit_artifacts():
    with pytest.raises(ValueError, match="yolo-artifacts"):
        live_bridge.docker_command(mux_only=False, output_root=Path("/tmp"), detector="yolo26m")


def test_offline_cli_refuses_yolo_without_inference_or_artifacts(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["live_bridge", "mux-smoke", "--cam02-video", "fake1.mp4",
                                      "--cam05-video", "fake2.mp4", "--detector", "yolo26m"])
    with pytest.raises(SystemExit) as exc:
        live_bridge.main()
    assert exc.value.code == 2

    monkeypatch.setattr(sys, "argv", ["live_bridge", "mux-smoke", "--cam02-video", "fake1.mp4",
                                      "--cam05-video", "fake2.mp4", "--detector", "yolo26m", "--inference"])
    with pytest.raises(SystemExit) as exc:
        live_bridge.main()
    assert exc.value.code == 2


def test_yolo_candidate_never_starts_an_rtsp_or_mqtt_network_session(tmp_path, monkeypatch):
    from services.mv3dt_staging.live_tracking import yolo_candidate
    root, assets, expected = _fake_tree(tmp_path)
    staged = stage_yolo_batch_two(assets, root / ".runtime" / "stage", expected=expected, root=root)
    monkeypatch.setattr(yolo_candidate, "stage_yolo_batch_two", lambda model_dir, output_root: staged)
    cmd = live_bridge.docker_command(
        mux_only=False, output_root=root / ".runtime" / "stage",
        detector="yolo26m", yolo_artifacts=assets,
    )
    assert "--network" in cmd and cmd[cmd.index("--network") + 1] == "none"
    assert "--pull=never" in cmd
    assert "--gpus" in cmd
    assert live_bridge.YOLO_DS_IMAGE in cmd
    assert any("target=/models,readonly" in item for item in cmd)
    assert any("pgie-yolo26m-b2.txt" in item for item in cmd)
    assert not any("PeopleNetTransformer" in item or "config_pgie_pnt" in item for item in cmd)
    assert "rtsp://" not in " ".join(cmd)


def test_default_peoplenet_command_keeps_its_existing_image_and_mount(monkeypatch, tmp_path):
    model_dir = tmp_path / "PeopleNetTransformer"
    model_dir.mkdir()
    monkeypatch.setattr(live_bridge, "MODEL_DIR", model_dir)
    cmd = live_bridge.docker_command(mux_only=False, output_root=tmp_path / "out")
    assert "--pull=never" not in cmd
    assert live_bridge.DS_IMAGE in cmd
    assert "/workspace/app/config_pgie_pnt_batch2.txt" in cmd[-1]
    assert any(
        f"source={model_dir},target=/workspace/models/PeopleNetTransformer,readonly" in item
        for item in cmd
    )
