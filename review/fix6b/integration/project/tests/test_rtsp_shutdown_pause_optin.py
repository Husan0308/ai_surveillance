"""Default-off launcher handoff tests for the isolated RTSP shutdown guard."""

from pathlib import Path

import pytest

from services.gpu.runtime import live_launcher as live


def _mosaic_args(tmp_path: Path, *, suppress: bool = False) -> list[str]:
    return live._docker_args(
        "pinned-image", tmp_path / "models", "bridge", None, None, "test-owned",
        mosaic_viewer_binary=tmp_path / "candidate" / "deepstream-app",
        mosaic_slot_path=tmp_path / "ipc" / "preview.slot",
        suppress_shutdown_pause=suppress,
    )


def test_shutdown_pause_handoff_is_off_by_default(tmp_path):
    args = _mosaic_args(tmp_path)
    assert "SIX_LIVE_SUPPRESS_SHUTDOWN_PAUSE=1" not in args


def test_shutdown_pause_handoff_is_explicit_and_mosaic_only(tmp_path):
    args = _mosaic_args(tmp_path, suppress=True)
    assert args.count("SIX_LIVE_SUPPRESS_SHUTDOWN_PAUSE=1") == 1

    with pytest.raises(ValueError, match="isolated mosaic viewer"):
        live._docker_args(
            "pinned-image", tmp_path / "models", "bridge", ":0", tmp_path / "xauth",
            "test-owned", suppress_shutdown_pause=True,
        )


def test_environment_opt_in_fails_closed_without_live_mosaic_run(monkeypatch):
    monkeypatch.setenv("SIX_LIVE_SUPPRESS_SHUTDOWN_PAUSE", "1")
    with pytest.raises(SystemExit):
        live.main(["--check", "--infer-config", "unused"])


def test_invalid_environment_value_fails_closed(monkeypatch):
    monkeypatch.setenv("SIX_LIVE_SUPPRESS_SHUTDOWN_PAUSE", "yes")
    with pytest.raises(SystemExit):
        live.main(["--check", "--infer-config", "unused"])
