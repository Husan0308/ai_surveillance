from pathlib import Path

from scripts.dev_room_mv3dt.run_room_pair import (
    configured_camera_decoder_settings,
    deepstream_command,
)


def test_room_pair_uses_validated_decoder_profile() -> None:
    low_latency, extra_surfaces = configured_camera_decoder_settings(
        {"CAM-01", "CAM-04"}
    )
    assert low_latency == ("CAM-01", "CAM-04")
    assert extra_surfaces == {"CAM-04": 8}


def test_live_command_forwards_room_pair_decoder_profile(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MV3DT_UI_PREVIEW_DIAGNOSTICS", "1")
    (tmp_path / "config_deepstream.txt").write_text(
        "[source0]\nenable=1\n[source1]\nenable=1\n"
    )
    (tmp_path / "config_msgconv.txt").write_text(
        "[sensor0]\nid=CAM-01\n[sensor1]\nid=CAM-04\n"
    )
    command = deepstream_command(
        tmp_path, tmp_path / "binary", "ds-image", "live", "preview-test"
    )
    assert "MV3DT_DECODER_LOW_LATENCY_CAMERAS=CAM-01,CAM-04" in command
    assert "MV3DT_DECODER_EXTRA_SURFACES_BY_CAMERA=CAM-04=8" in command
    assert "MV3DT_UI_PREVIEW_DIAGNOSTICS=/workspace/experiments/logs/probe/preview_diagnostics.jsonl" in command
