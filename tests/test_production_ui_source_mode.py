from __future__ import annotations

import json
from pathlib import Path

from scripts.dev_room_mv3dt.run_room_pair import make_crop_socket_alias, replay_config
from services.mv3dt_room.room_pair_state import RoomPairState


def _write_state(root: Path, mode: str, people: list[dict], running: bool) -> None:
    (root / "identity-live").mkdir(parents=True)
    (root / "source_mode.json").write_text(json.dumps({"source_mode": mode, "session_id": root.name}))
    (root / "identity-live/current_state.json").write_text(json.dumps({
        "people": people,
        "active_frame_by_camera": {"CAM-01": 10, "CAM-04": 10},
    }))
    (root / ("running" if running else "done")).touch()


def test_room_pair_state_does_not_leak_replay_people_into_live(tmp_path: Path) -> None:
    replay = tmp_path / "replay-1"
    _write_state(replay, "replay", [{
        "camera_id": "CAM-01", "application_id": "Person_01", "global_person_id": "Person_01",
        "native_track_id": 7, "frame": 10, "bbox": [1, 2, 3, 4], "world": [5, 6],
    }], running=False)
    state = RoomPairState(tmp_path)
    first = state.snapshot()
    assert first["source_mode"] == "replay"
    assert [row["application_id"] for row in first["people"]] == ["Person_01"]

    live = tmp_path / "live-2"
    _write_state(live, "live", [], running=True)
    second = state.snapshot()
    assert second["source_mode"] == "live"
    assert second["session_id"] == "live-2"
    assert second["people"] == []
    assert second["presence"]["no_marker_without_active_camera_observation"] is True


def test_room_pair_state_ignores_stale_running_marker_when_newer_session_exists(tmp_path: Path) -> None:
    stale = tmp_path / "live-stale"
    _write_state(stale, "live", [{
        "camera_id": "CAM-01", "application_id": "Person_01", "global_person_id": "Person_01",
        "native_track_id": 7, "frame": 10, "bbox": [1, 2, 3, 4], "world": [5, 6],
    }], running=True)

    current = tmp_path / "live-current"
    _write_state(current, "live", [], running=False)

    snapshot = RoomPairState(tmp_path).snapshot()
    assert snapshot["session_id"] == "live-current"
    assert snapshot["status"] == "complete"
    assert snapshot["people"] == []


def test_replay_config_changes_only_runtime_copy_to_clock_paced(tmp_path: Path) -> None:
    config = tmp_path / "config_deepstream.txt"
    config.write_text("""[source0]\ntype=3\nuri=file:///cam_00.mp4\n\n[sink0]\nenable=1\ntype=1\nsync=0\n\n[sink3]\nenable=1\ntype=6\nsync=0\n""")
    replay_config(tmp_path)
    text = config.read_text()
    assert "uri=file:///cam_00.mp4" in text
    assert "[sink0]\nenable=1\ntype=1\nsync=1" in text
    assert "[sink3]\nenable=1\ntype=6\nsync=0" in text


def test_frontend_declares_all_six_camera_tiles() -> None:
    text = Path("services/frontend/app/main.py").read_text()
    assert 'ALL_CAMERAS = tuple(f"CAM-{index:02d}" for index in range(1, 7))' in text
    assert "self.camera_wall.set_cameras(list(ALL_CAMERAS))" in text


def test_preview_only_runtime_owns_all_ui_preview_cameras() -> None:
    from services.camera_v11.preview_only_runtime import DEFAULT_CAMERAS
    assert set(DEFAULT_CAMERAS) == {f"CAM-{index:02d}" for index in range(1, 7)}


def test_crop_socket_host_alias_stays_within_af_unix_limit(tmp_path: Path) -> None:
    stage = tmp_path / ("descriptive-runtime-segment-" * 4) / "run"
    (stage / "logs").mkdir(parents=True)
    alias, socket_path = make_crop_socket_alias(stage, process_id=987654)
    try:
        assert len(str(socket_path).encode()) < 108
        assert alias.resolve() == (stage / "logs").resolve()
    finally:
        alias.unlink(missing_ok=True)


def test_frontend_scales_dev_room_overlays_from_per_camera_source_dimensions(monkeypatch) -> None:
    from services.frontend.app.camera_wall import overlay_source_dimensions, scale_overlay_bbox

    monkeypatch.setenv("MV3DT_OVERLAY_WIDTH_CAM01", "2560")
    monkeypatch.setenv("MV3DT_OVERLAY_HEIGHT_CAM01", "1440")
    monkeypatch.setenv("MV3DT_OVERLAY_WIDTH_CAM04", "3200")
    monkeypatch.setenv("MV3DT_OVERLAY_HEIGHT_CAM04", "1800")
    assert overlay_source_dimensions("CAM-01") == (2560, 1440)
    assert overlay_source_dimensions("CAM-04") == (3200, 1800)
    assert overlay_source_dimensions("CAM-01", "replay") == (1920, 1080)
    assert overlay_source_dimensions("CAM-04", "replay") == (1920, 1080)
    assert scale_overlay_bbox((1280, 720, 2560, 1440), 1920, 1080, 2560, 1440) == (
        960, 540, 1920, 1080
    )
    assert scale_overlay_bbox((1600, 900, 3200, 1800), 1920, 1080, 3200, 1800) == (
        960, 540, 1920, 1080
    )

    text = Path("services/frontend/app/camera_wall.py").read_text()
    assert "MV3DT_OVERLAY_WIDTH" in text
    assert "overlay_source_dimensions(camera_id)" in text
    assert "self.camera_id, self.source_mode" in text


def test_preview_reader_reopens_when_deepstream_recreates_file(tmp_path: Path) -> None:
    from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameReader, PreviewFrameWriter

    path = tmp_path / "v11_ui_preview_cam01_v1.bin"
    payload = bytes([13] * (16 * 8 * 4))
    first = PreviewFrameWriter(str(path), width=16, height=8)
    reader = PreviewFrameReader(str(path))
    try:
        first.publish(payload, object_count=1)
        assert reader.read_latest().payload == payload
        first.close()
        second_payload = bytes([47] * len(payload))
        second = PreviewFrameWriter(str(path), width=16, height=8)
        try:
            second.publish(second_payload, object_count=2)
            fresh = reader.read_latest()
            assert fresh is not None
            assert fresh.payload == second_payload
            assert fresh.object_count == 2
        finally:
            second.close()
    finally:
        reader.close()
