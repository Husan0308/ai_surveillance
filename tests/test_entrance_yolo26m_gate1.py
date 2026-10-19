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


def test_engine_only_config_fails_closed_without_onnx_build_fallback():
    original = FROZEN_PGIE.read_text()
    edited = render_batch_two_config(original, engine_only=True)
    assert "onnx-file=" not in edited
    assert "model-engine-file=/models/yolo26m_raw_otm_b6_fp16.engine" in edited
    assert "custom-lib-path=/models/libyolo26rawperson.so" in edited
    assert "batch-size=2" in edited
    assert "batch-size=6" not in edited
    assert "interval=0" in edited
    assert "pre-cluster-threshold=0.25" in edited
    assert "nms-iou-threshold=0.45" in edited


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


def test_engine_only_stage_has_distinct_config_and_no_onnx(tmp_path):
    root, assets, expected = _fake_tree(tmp_path)
    result = stage_yolo_batch_two(
        assets, root / ".runtime" / "stage", expected=expected, root=root, engine_only=True
    )
    config = Path(result["config_file"]).read_text()
    assert Path(result["config_file"]).name == "pgie-yolo26m-b2-engine-only.txt"
    assert result["engine_only"] is True
    assert "onnx-file=" not in config
    assert "model-engine-file=/models/yolo26m_raw_otm_b6_fp16.engine" in config
    runtime_models = Path(result["model_dir"])
    assert {item.name for item in runtime_models.iterdir()} == {
        "yolo26m_raw_otm_b6_fp16.engine", "libyolo26rawperson.so", "labels.txt"
    }
    assert not (runtime_models / "yolo26m_raw_otm.onnx").exists()
    assert {name: hashlib.sha256((runtime_models / name).read_bytes()).hexdigest()
            for name in ("yolo26m_raw_otm_b6_fp16.engine", "libyolo26rawperson.so", "labels.txt")} == {
        name: expected[name]
        for name in ("yolo26m_raw_otm_b6_fp16.engine", "libyolo26rawperson.so", "labels.txt")
    }


def test_engine_only_stage_refuses_to_replace_changed_staged_plan(tmp_path):
    root, assets, expected = _fake_tree(tmp_path)
    output = root / ".runtime" / "stage"
    result = stage_yolo_batch_two(assets, output, expected=expected, root=root, engine_only=True)
    plan = Path(result["model_dir"]) / "yolo26m_raw_otm_b6_fp16.engine"
    plan.write_bytes(b"unexpected replacement")
    with pytest.raises(ValueError, match="do not overwrite"):
        stage_yolo_batch_two(assets, output, expected=expected, root=root, engine_only=True)


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


def test_offline_frame_crosswalk_requires_complete_ordered_frames_and_pts():
    rows = [
        {"source_id": sid, "input_frame_index": idx, "source_pts_ns": idx * 33_333_333}
        for idx in range(3) for sid in (0, 1)
    ]
    live_bridge.validate_offline_stage_crosswalk(rows, 3, "MUX")
    missing = [row for row in rows if not (row["source_id"] == 1 and row["input_frame_index"] == 1)]
    with pytest.raises(RuntimeError, match="frame crosswalk mismatch"):
        live_bridge.validate_offline_stage_crosswalk(missing, 3, "MUX")
    bad_pts = [dict(row) for row in rows]
    next(row for row in bad_pts if row["source_id"] == 0 and row["input_frame_index"] == 2)["source_pts_ns"] = 0
    with pytest.raises(RuntimeError, match="source PTS not strictly increasing"):
        live_bridge.validate_offline_stage_crosswalk(bad_pts, 3, "PGIE")


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
    staged = stage_yolo_batch_two(
        assets, root / ".runtime" / "stage", expected=expected, root=root, engine_only=True
    )
    calls = []
    def stage_for_runtime(model_dir, output_root, *, engine_only=False):
        calls.append(engine_only)
        return staged
    monkeypatch.setattr(yolo_candidate, "stage_yolo_batch_two", stage_for_runtime)
    cmd = live_bridge.docker_command(
        mux_only=False, output_root=root / ".runtime" / "stage",
        detector="yolo26m", yolo_artifacts=assets, lossless_input=True,
    )
    assert "--network" in cmd and cmd[cmd.index("--network") + 1] == "none"
    assert "--pull=never" in cmd
    assert "--gpus" in cmd
    assert live_bridge.YOLO_DS_IMAGE in cmd
    assert live_bridge.YOLO_DS_IMAGE == (
        "nvcr.io/nvidia/deepstream@sha256:"
        "10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
    )
    assert live_bridge.YOLO_DS_IMAGE_ID == "sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
    assert calls == [True]
    assert "--lossless-input" in cmd[-1]
    assert any("target=/models,readonly" in item for item in cmd)
    assert any("pgie-yolo26m-b2-engine-only.txt" in item for item in cmd)
    assert not any("PeopleNetTransformer" in item or "config_pgie_pnt" in item for item in cmd)
    assert "rtsp://" not in " ".join(cmd)


def test_default_peoplenet_command_keeps_its_existing_image_and_mount(monkeypatch, tmp_path):
    model_dir = tmp_path / "PeopleNetTransformer"
    model_dir.mkdir()
    monkeypatch.setattr(live_bridge, "MODEL_DIR", model_dir)
    cmd = live_bridge.docker_command(mux_only=False, output_root=tmp_path / "out")
    assert "--pull=never" not in cmd
    assert live_bridge.DS_IMAGE in cmd
    assert live_bridge.DS_IMAGE == "nvcr.io/nvidia/deepstream:9.1-triton-multiarch"
    assert live_bridge.YOLO_DS_IMAGE not in cmd
    assert "--lossless-input" not in cmd[-1]
    assert "/workspace/app/config_pgie_pnt_batch2.txt" in cmd[-1]
    assert any(
        f"source={model_dir},target=/workspace/models/PeopleNetTransformer,readonly" in item
        for item in cmd
    )
