"""CPU-only safety contracts for the bounded offline detector comparison."""
import hashlib
from pathlib import Path
import sys

import pytest

from services.mv3dt_staging.live_tracking import live_bridge


PNT_CONFIG = """[property]
onnx-file=/workspace/models/PeopleNetTransformer/peoplenet_transformer_model_op17.onnx
model-engine-file=/workspace/models/PeopleNetTransformer/peoplenet_transformer_model_op17.onnx_b2_gpu0_fp16.engine
batch-size=2
interval=0
infer-dims=3;544;960
pre-cluster-threshold=0.3
"""


def _pnt_fixture(tmp_path):
    model = tmp_path / "models"
    model.mkdir()
    config = tmp_path / "config.txt"
    config.write_text(PNT_CONFIG)
    expected = {}
    for relative in live_bridge.PNT_ASSET_SHA256:
        path = model / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("fixture:" + relative).encode())
        expected[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return model, config, expected


def test_pnt_engine_only_staging_removes_only_onnx_and_preserves_assets(tmp_path):
    model, config, expected = _pnt_fixture(tmp_path)
    source_before = config.read_bytes()
    staged = live_bridge.stage_peoplenet_engine_only(
        model, config, tmp_path / "out", expected_assets=expected,
        expected_config_sha256=hashlib.sha256(source_before).hexdigest(),
    )
    staged_config = Path(staged["config_file"])
    assert staged_config.read_text() == PNT_CONFIG.replace(
        "onnx-file=/workspace/models/PeopleNetTransformer/peoplenet_transformer_model_op17.onnx\n", "", 1
    )
    assert "onnx-file=" not in staged_config.read_text()
    assert config.read_bytes() == source_before
    staged_models = Path(staged["model_dir"])
    assert not (staged_models / "peoplenet_transformer_model_op17.onnx").exists()
    assert hashlib.sha256((staged_models / "peoplenet_transformer_model_op17.onnx_b2_gpu0_fp16.engine").read_bytes()).hexdigest() == expected[
        "peoplenet_transformer_model_op17.onnx_b2_gpu0_fp16.engine"
    ]
    assert staged["engine_only"] is True
    assert staged_config.stat().st_mode & 0o777 == 0o600


def test_pnt_engine_only_staging_rejects_changed_model_asset(tmp_path):
    model, config, expected = _pnt_fixture(tmp_path)
    (model / "detector_labels.txt").write_text("changed")
    with pytest.raises(ValueError, match="asset missing/changed"):
        live_bridge.stage_peoplenet_engine_only(
            model, config, tmp_path / "out", expected_assets=expected,
            expected_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        )


def test_pnt_engine_only_staging_rejects_config_change_and_unsafe_output(tmp_path):
    model, config, expected = _pnt_fixture(tmp_path)
    with pytest.raises(ValueError, match="config SHA256"):
        live_bridge.stage_peoplenet_engine_only(
            model, config, tmp_path / "out", expected_assets=expected,
            expected_config_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="must remain under /tmp"):
        live_bridge.stage_peoplenet_engine_only(
            model, config, Path("/var/tmp/entrance-gate3-unsafe-test-output"), expected_assets=expected,
            expected_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        )


def test_comparison_runtime_is_offline_peoplenet_only(monkeypatch, tmp_path):
    monkeypatch.setattr(live_bridge, "MODEL_DIR", tmp_path / "unused-models")
    monkeypatch.setattr(live_bridge, "stage_peoplenet_engine_only", lambda *a, **k: {
        "model_dir": str(tmp_path / "pnt-models"), "config_file": str(tmp_path / "pnt.txt")
    })
    command = live_bridge.docker_command(
        mux_only=False, output_root=tmp_path / "out", detector="peoplenet", comparison_runtime=True
    )
    assert live_bridge.YOLO_DS_IMAGE in command
    assert "--pull=never" in command
    assert "pgie-pnt-b2-engine-only.txt" in command[-1]
    assert any("target=/workspace/models/PeopleNetTransformer,readonly" in item for item in command)
    assert "rtsp://" not in " ".join(command)
    with pytest.raises(ValueError, match="restricted"):
        live_bridge.docker_command(
            mux_only=False, output_root=tmp_path / "out", detector="yolo26m", comparison_runtime=True
        )
    with pytest.raises(ValueError, match="restricted"):
        live_bridge.docker_command(
            mux_only=True, output_root=tmp_path / "out", detector="peoplenet", comparison_runtime=True
        )


def test_pnt_engine_only_uses_original_default_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(live_bridge, "MODEL_DIR", tmp_path / "unused-models")
    monkeypatch.setattr(live_bridge, "stage_peoplenet_engine_only", lambda *a, **k: {
        "model_dir": str(tmp_path / "pnt-models"), "config_file": str(tmp_path / "pnt.txt")
    })
    command = live_bridge.docker_command(
        mux_only=False, output_root=tmp_path / "out", detector="peoplenet", pnt_engine_only=True
    )
    assert live_bridge.DS_IMAGE in command
    assert live_bridge.YOLO_DS_IMAGE not in command
    assert "--pull=never" in command
    assert "pgie-pnt-b2-engine-only.txt" in command[-1]


def test_default_peoplenet_path_still_uses_its_original_image_and_config(monkeypatch, tmp_path):
    model_dir = tmp_path / "PeopleNetTransformer"
    model_dir.mkdir()
    monkeypatch.setattr(live_bridge, "MODEL_DIR", model_dir)
    command = live_bridge.docker_command(mux_only=False, output_root=tmp_path / "out")
    assert live_bridge.DS_IMAGE in command
    assert live_bridge.YOLO_DS_IMAGE not in command
    assert "--pull=never" not in command
    assert "/workspace/app/config_pgie_pnt_batch2.txt" in command[-1]


def test_comparison_smoke_limit_is_bounded_at_600_frames(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["live_bridge", "mux-smoke", "--cam02-video", "a", "--cam05-video", "b", "--frames", "300"])
    monkeypatch.setattr(live_bridge, "run_mux_smoke", lambda args: 0)
    assert live_bridge.main() == 0
    monkeypatch.setattr(sys, "argv", ["live_bridge", "mux-smoke", "--cam02-video", "a", "--cam05-video", "b", "--frames", "601"])
    with pytest.raises(SystemExit) as exc:
        live_bridge.main()
    assert exc.value.code == 2


def test_appsrc_clock_and_timing_probe_contract_are_explicit():
    source = (live_bridge.APP_DIR / "deepstream_appsrc.cpp").read_text()
    assert '"framerate", GST_TYPE_FRACTION, 20, 1' in source
    assert "pgie_pad_latency_ns" in source
    assert "tracker_pad_latency_ns" in source
    assert "host_input_to_stage_ns" in source
    assert "stage_input_probe" in source
