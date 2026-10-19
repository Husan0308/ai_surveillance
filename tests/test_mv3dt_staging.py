from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fastapi.testclient import TestClient
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from services.mv3dt_staging.api import create_app
from services.mv3dt_staging.adoption_aliases import AdoptionAliasError, AdoptionAliasIndex, build_bound_manifest
from services.mv3dt_staging.event_store import CAMERA_MAPS, import_event_log, sha256_file
from services.mv3dt_staging.presentation import display_state, group_for_display, update_active_observations, world_to_map_pixel
from services.mv3dt_staging.qt_ui import MainWindow


def _message(sensor: str, frame: int, object_id: str, *, timestamp: str | None = None,
             world: str = "5.0,-6.0", confidence: float | None = 0.83) -> dict:
    object_row = {
        "id": object_id,
        "type": "Person",
        "bbox": {"leftX": 100, "topY": 120, "rightX": 200, "bottomY": 300},
        "info": {"footLocation2D": "150,290", "footLocation3D": world},
    }
    if confidence is not None:
        object_row["confidence"] = confidence
    return {"version": "4.0", "id": str(frame),
            "timestamp": timestamp or f"2026-10-17T19:05:{46 + frame:02d}.000Z",
            "sensorId": sensor, "objects": [object_row]}


def _write_log(path: Path, messages: list[dict]) -> None:
    path.write_text("Starting fixture\n" + "".join(
        "JSON Data:\n" + json.dumps(message) + "\n--------------------------------------------------\n" for message in messages
    ), encoding="utf-8")


def _qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


def test_offline_import_mapping_stable_ids_deduplicates_and_timestamp_order(tmp_path: Path) -> None:
    source = tmp_path / "events.log"
    first = _message("Camera1", 0, "7")
    second = _message("Camera2", 0, "7")
    third = _message("Camera1", 1, "7", timestamp="2026-10-17T19:05:49.000Z", confidence=None)
    _write_log(source, [first, second, third, first])
    database = tmp_path / "staging.sqlite3"
    result = import_event_log(source, database, expected_frames=2)

    assert result["inserted_messages"] == 3
    assert result["inserted_observations"] == 3
    assert result["duplicate_messages"] == 1
    assert result["duplicate_observations"] == 1
    assert database.stat().st_mode & 0o777 == 0o600

    _qapp()
    map_path = tmp_path / "map.png"
    map_image = QImage(10, 10, QImage.Format.Format_RGB32)
    map_image.fill(QColor("#fff"))
    assert map_image.save(str(map_path))
    transforms_path = tmp_path / "transforms.yml"
    transforms_path.write_text(
        "T_ov2px:\n  - 100\n  - 0\n  - 0\n  - 0\n  - -100\n  - 0\n  - 0\n  - 0\n  - 1\n",
        encoding="utf-8",
    )
    with TestClient(create_app(database, map_path, transforms_path)) as client:
        summary = client.get("/api/v1/summary").json()
        assert summary["camera_mapping"][0]["camera_id"] == "CAM-01"
        assert summary["camera_mapping"][0]["source_id"] == 0
        assert summary["camera_mapping"][1]["camera_id"] == "CAM-04"
        assert summary["camera_mapping"][1]["source_id"] == 1
        assert summary["distinct_mv3dt_output_ids_union"] == 1
        assert summary["shared_output_ids"] == ["7"]
        assert summary["source_pts_in_event_log"] is False
        assert summary["map"]["width"] == 10 and summary["map"]["height"] == 10
        assert summary["map"]["projected_observation_counts"] == {
            "inside_map": 0, "outside_map": 3, "missing_world_position": 0
        }
        rows = client.get("/api/v1/observations", params={"limit": 20}).json()["observations"]
        assert [(row["frame_id"], row["camera_id"]) for row in rows] == [
            (0, "CAM-01"), (0, "CAM-04"), (1, "CAM-01")
        ]
        assert rows[-1]["mv3dt_object_confidence"] is None
        assert rows[0]["media_time_seconds_nominal"] == 0.0
        assert rows[0]["world_position_unverified_units"] == [5.0, -6.0]
        assert client.get("/api/v1/observations", params={"camera_id": "CAM-02"}).status_code == 400


def test_conflicting_duplicate_observation_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "conflict.log"
    left = _message("Camera1", 2, "5", confidence=0.5)
    right = _message("Camera1", 2, "5", confidence=0.9)
    _write_log(source, [left, right])
    try:
        import_event_log(source, tmp_path / "conflict.sqlite3")
    except ValueError as error:
        assert "conflicting duplicate" in str(error)
    else:
        raise AssertionError("conflicting duplicate should not be silently resolved")


def test_websocket_replay_sends_frame_order_and_camera_scoped_observations(tmp_path: Path) -> None:
    source = tmp_path / "events.log"
    _write_log(source, [
        _message("Camera2", 0, "3"),
        _message("Camera1", 0, "3"),
        _message("Camera1", 1, "8", timestamp="2026-10-17T19:05:49.000Z"),
    ])
    database = tmp_path / "staging.sqlite3"
    import_event_log(source, database, expected_frames=2)
    with TestClient(create_app(database)) as client:
        with client.websocket_connect("/ws/replay?speed=100&start_frame=0&end_frame=1") as socket:
            start = socket.receive_json()
            assert start["type"] == "start"
            assert start["source_pts_in_event_log"] is False
            frame0 = socket.receive_json()
            frame1 = socket.receive_json()
            complete = socket.receive_json()
    assert [frame0["frame_id"], frame1["frame_id"]] == [0, 1]
    assert [row["camera_id"] for row in frame0["observations"]] == ["CAM-01", "CAM-04"]
    assert frame0["observations"][0]["mv3dt_global_id"] == frame0["observations"][1]["mv3dt_global_id"] == "3"
    assert frame0["nominal_media_time_seconds"] <= frame1["nominal_media_time_seconds"]
    assert complete == {"type": "complete", "last_frame_id": 1, "frames_sent": 2}


def test_active_map_rows_keep_camera_scoped_ids_and_expire_without_fusion() -> None:
    base = {"mv3dt_global_id": "9", "frame_id": 10, "world_position_unverified_units": [5, -6]}
    rows = update_active_observations([], [
        {**base, "camera_id": "CAM-01"}, {**base, "camera_id": "CAM-04"}
    ], current_frame=10, expiry_frames=30)
    assert len(rows) == 2
    assert {row["camera_id"] for row in rows} == {"CAM-01", "CAM-04"}
    groups = group_for_display(rows)
    assert len(groups) == 1
    assert groups[0]["cameras"] == ["CAM-01", "CAM-04"]
    moved = update_active_observations(rows, [
        {**base, "camera_id": "CAM-01", "frame_id": 11, "world_position_unverified_units": [5.5, -6]}
    ], current_frame=11, expiry_frames=30)
    cam01 = next(row for row in moved if row["camera_id"] == "CAM-01")
    assert cam01["frame_id"] == 11 and cam01["world_position_unverified_units"] == [5.5, -6]
    assert update_active_observations(moved, [], current_frame=42, expiry_frames=30) == []
    px, py = world_to_map_pixel(5.0, -6.0, [100.020558, 0, 0, 0, -100.020558, 0, 0, 0, 1])
    assert math.isclose(px, 500.10279, abs_tol=1e-9)
    assert math.isclose(py, 600.123348, abs_tol=1e-9)


def test_qt_map_client_renders_two_camera_rows_for_same_mv3dt_id(tmp_path: Path) -> None:
    app = _qapp()
    image_path = tmp_path / "map.png"
    image = QImage(1200, 896, QImage.Format.Format_RGB32)
    image.fill(QColor("#eeeeee"))
    assert image.save(str(image_path))
    window = MainWindow(image_path, "ws://127.0.0.1:9/ws/replay?speed=100", auto_start=False)
    window._on_message(json.dumps({"type": "start", "transform_world_to_pixel_row_major": [
        100.020558, 0, 0, 0, -100.020558, 0, 0, 0, 1
    ]}))
    common = {"mv3dt_global_id": "4", "frame_id": 10,
              "world_position_unverified_units": [5.0, -6.0]}
    window._on_message(json.dumps({"type": "frame", "frame_id": 10,
        "frame_messages": [{"camera_id": "CAM-01", "source_id": 0}, {"camera_id": "CAM-04", "source_id": 1}],
        "observations": [{**common, "camera_id": "CAM-01"}, {**common, "camera_id": "CAM-04"}]}))
    app.processEvents()
    assert window.table.rowCount() == 2
    assert {window.table.item(row, 0).text() for row in range(2)} == {"CAM-01", "CAM-04"}
    assert {window.table.item(row, 2).text() for row in range(2)} == {"4"}
    assert {window.table.item(row, 3).text() for row in range(2)} == {"4"}
    assert len(window.map_canvas.observations) == 2
    assert len(window.map_canvas.primary_markers) == 1
    assert "Camera observations: 2" in window.counters.text()
    assert "Unique global IDs: 1" in window.counters.text()
    assert "Confirmed visible people: UNAVAILABLE" in window.counters.text()
    assert window.received_frame_batches == 1
    assert window.received_observations == 2
    rendered = QImage(window.map_canvas.size(), QImage.Format.Format_ARGB32)
    rendered.fill(Qt.GlobalColor.transparent)
    painter = QPainter(rendered)
    window.map_canvas.render(painter, QPoint(0, 0))
    painter.end()
    assert not rendered.isNull()
    window.close()


def test_qt_client_rejects_non_loopback_websocket(tmp_path: Path) -> None:
    _qapp()
    image_path = tmp_path / "map.png"
    image = QImage(32, 32, QImage.Format.Format_RGB32)
    image.fill(QColor("#fff"))
    assert image.save(str(image_path))
    try:
        MainWindow(image_path, "ws://example.invalid/ws/replay", auto_start=False)
    except ValueError as error:
        assert "loopback" in str(error)
    else:
        raise AssertionError("staging Qt client must reject non-loopback endpoints")


def _duplicate_frame_rows(frame: int = 3599) -> list[dict]:
    return [{"camera_id": camera, "mv3dt_global_id": output_id, "frame_id": frame,
             "world_position_unverified_units": world} for camera, output_id, world in [
        ("CAM-01", "1", [4.7223, -7.273937]),
        ("CAM-01", "10", [2.352619, -2.201956]),
        ("CAM-01", "5", [3.480657, -2.161604]),
        ("CAM-04", "5", [2.806136, -2.587836]),
    ]]


def _exact_frame_review(frame: int = 3599) -> dict:
    return {"frame_id": frame, "verified_for_source": True,
            "people_groups": [{"primary_global_id": "1", "member_global_ids": ["1"]},
                              {"primary_global_id": "5", "member_global_ids": ["5", "10"]}],
            "note": "Test exact-frame visual review: IDs 5 and 10 cover one person."}


def test_global_id_markers_preserve_distinct_ids_and_all_camera_evidence() -> None:
    rows = _duplicate_frame_rows()
    ordinary = display_state(rows, current_frame=3599)
    assert ordinary["camera_observations"] == 4
    assert ordinary["unique_global_ids"] == 3
    assert ordinary["confirmed_visible_people"] is None
    assert len(ordinary["primary_markers"]) == 3
    # Same world position is not evidence that different IDs represent one person.
    close_people = [dict(rows[0]), {**rows[0], "mv3dt_global_id": "other-person"}]
    assert len(display_state(close_people, current_frame=3599)["primary_markers"]) == 2
    reviewed = display_state(rows, current_frame=3599, visual_review=_exact_frame_review())
    assert reviewed["camera_observations"] == 4 and reviewed["unique_global_ids"] == 3
    assert reviewed["confirmed_visible_people"] == 2
    assert {row["mv3dt_global_id"] for row in reviewed["primary_markers"]} == {"1", "5"}
    assert reviewed["duplicate_global_ids"] == {"10": "5"}
    assert len(reviewed["current_observations"]) == 4
    id5 = next(row for row in reviewed["primary_markers"] if row["mv3dt_global_id"] == "5")
    assert id5["observing_cameras"] == ["CAM-01", "CAM-04"]
    assert id5["reviewed_duplicate_ids"] == ["10"]
    assert id5["world_position_unverified_units"] == rows[2]["world_position_unverified_units"]


def test_recent_rows_and_review_of_other_frame_cannot_create_confirmed_people() -> None:
    rows = _duplicate_frame_rows(frame=3598)
    state = display_state(rows, current_frame=3599, visual_review=_exact_frame_review())
    assert state["camera_observations"] == state["unique_global_ids"] == 0
    assert state["recent_observations"] == 4 and state["confirmed_visible_people"] is None
    assert state["primary_markers"] == []
    current = display_state(_duplicate_frame_rows(), current_frame=3599,
                            visual_review=_exact_frame_review(frame=3598))
    assert len(current["primary_markers"]) == 3 and current["confirmed_visible_people"] is None


def test_qt_reviewed_frame_shows_two_markers_but_keeps_four_observations(tmp_path: Path) -> None:
    app = _qapp()
    map_path = tmp_path / "map.png"
    image = QImage(1200, 896, QImage.Format.Format_RGB32)
    image.fill(QColor("#fff"))
    assert image.save(str(map_path))
    window = MainWindow(map_path, "ws://127.0.0.1:9/ws/replay", auto_start=False)
    window._on_message(json.dumps({"type": "frame", "frame_id": 3599,
                                  "observations": _duplicate_frame_rows(),
                                  "visual_review": _exact_frame_review()}))
    app.processEvents()
    assert window.table.rowCount() == 4
    assert len(window.map_canvas.observations) == 4
    assert len(window.map_canvas.primary_markers) == 2
    assert "Camera observations: 4" in window.counters.text()
    assert "Unique global IDs: 3" in window.counters.text()
    assert "Confirmed visible people: 2" in window.counters.text()
    assert window.table.item(1, 2).text() == "10"
    assert window.table.item(1, 6).text() == "visually reviewed duplicate of raw ID 5"
    window.observation_toggle.setChecked(True)
    assert window.map_canvas.show_camera_observations is True
    window._on_message(json.dumps({"type": "complete", "last_frame_id": 3599}))
    window._on_disconnected()
    window._on_socket_error(None)
    assert "Replay complete at frame 3599" in window.status.text()
    window._on_message(json.dumps({"type": "frame", "frame_id": 3600, "observations": []}))
    assert "Confirmed visible people: UNAVAILABLE" in window.counters.text()
    assert len(window.map_canvas.primary_markers) == 0
    window.close()


def test_source_bound_frame_review_is_delivered_by_rest_and_websocket(tmp_path: Path) -> None:
    source = tmp_path / "events.log"
    first = _message("Camera1", 9, "1")
    first["objects"] += [_message("Camera1", 9, output_id)["objects"][0] for output_id in ("5", "10")]
    second = _message("Camera2", 9, "5")
    _write_log(source, [first, second])
    database = tmp_path / "staging.sqlite3"
    imported = import_event_log(source, database, expected_frames=10)
    document = {"schema_version": 1, "database_sha256": sha256_file(database),
                "expected_metadata": {"source_log_sha256": imported["source_log_sha256"]},
                "frames": {"9": {"expected_observations": [["CAM-01", "1"], ["CAM-01", "5"],
                                                            ["CAM-01", "10"], ["CAM-04", "5"]],
                                  "people_groups": _exact_frame_review(9)["people_groups"],
                                  "note": "Disposable fixture review."}}}
    review_path = tmp_path / "reviews.json"
    review_path.write_text(json.dumps(document), encoding="utf-8")
    before = sha256_file(database)
    with TestClient(create_app(database, review_path=review_path)) as client:
        frame = client.get("/api/v1/frames/9").json()
        assert len(frame["observations"]) == 4
        assert frame["visual_review"]["confirmed_visible_people"] == 2
        assert frame["visual_review"]["duplicate_global_ids"] == {"10": "5"}
        assert client.get("/api/v1/frames/8").json()["visual_review"] is None
        with client.websocket_connect("/ws/replay?speed=100&start_frame=9&end_frame=9") as socket:
            socket.receive_json()
            assert socket.receive_json()["visual_review"] == frame["visual_review"]
            socket.receive_json()
    assert sha256_file(database) == before
    document["database_sha256"] = "0" * 64
    review_path.write_text(json.dumps(document), encoding="utf-8")
    with TestClient(create_app(database, review_path=review_path)) as client:
        assert client.get("/api/v1/frames/9").json()["visual_review"] is None


def _adoption_alias(raw_id: str, display_id: str, *, camera_id: str = "CAM-02", source_id: int = 0,
                    old_birth: int = 100, target_camera: str = "CAM-05", target_source: int = 1,
                    target_birth: int = 200, start: int = 0, end: int = 5,
                    adoption_frame: int = 6, event_time: int = 300) -> dict:
    return {
        "camera_id": camera_id, "source_id": source_id, "raw_id": raw_id,
        "display_id": display_id, "valid_from_frame": start, "valid_until_frame": end,
        "adoption_frame": adoption_frame, "old_birth_timestamp_ns": old_birth,
        "target_camera_id": target_camera, "target_source_id": target_source,
        "target_birth_timestamp_ns": target_birth, "event_timestamp_ns": event_time,
        "event_type": "ADOPTED_ID",
        "raw_event": {"type": 3, "targetId": int(raw_id), "adoptedId": int(display_id),
                      "frameNum": adoption_frame, "targetIDTimeStamp": target_birth,
                      "messageTimeStamp": event_time},
    }


def _alias_index(rules: list[dict], *, source_hash: str = "a" * 64) -> AdoptionAliasIndex:
    return AdoptionAliasIndex(
        {"schema_version": 1, "scope": "offline_replay_only", "source_log_sha256": source_hash,
         "mqtt_raw_sha256": "b" * 64, "aliases": rules},
        expected_source_log_sha256=source_hash,
        camera_mapping={"Camera1": ("CAM-02", 0), "Camera2": ("CAM-05", 1)},
    )


def test_adoption_alias_is_camera_scoped_retrospective_and_preserves_raw_rows() -> None:
    index = _alias_index([_adoption_alias("1", "2")])
    before = index.resolve(camera_id="CAM-02", source_id=0, raw_id="1", frame_id=5)
    assert before["display_id"] == "2" and before["applied"] is True
    assert before["lineage"][0]["event_type"] == "ADOPTED_ID"
    # After adoption, the pipeline already emits the destination ID; do not alias it again.
    after = index.resolve(camera_id="CAM-02", source_id=0, raw_id="2", frame_id=6)
    assert after == {"display_id": "2", "applied": False, "lineage": []}
    # Same numeric ID on a different camera is not affected by this camera-scoped rule.
    peer = index.resolve(camera_id="CAM-05", source_id=1, raw_id="1", frame_id=5)
    assert peer["display_id"] == "1" and peer["applied"] is False


def test_adoption_alias_resolves_only_identity_linked_chains() -> None:
    first = _adoption_alias("1", "2", old_birth=100, target_camera="CAM-05", target_source=1,
                            target_birth=200, start=0, end=5, adoption_frame=6, event_time=300)
    second = _adoption_alias("2", "4", camera_id="CAM-05", source_id=1, old_birth=200,
                             target_camera="CAM-05", target_source=1, target_birth=400,
                             start=6, end=8, adoption_frame=9, event_time=500)
    result = _alias_index([first, second]).resolve(camera_id="CAM-02", source_id=0, raw_id="1", frame_id=1)
    assert result["display_id"] == "4"
    assert [event["display_id"] for event in result["lineage"]] == ["2", "4"]


def test_adoption_alias_rejects_cycles_conflicts_and_ambiguous_same_camera_merges() -> None:
    one_to_two = _adoption_alias("1", "2", old_birth=100, target_camera="CAM-05", target_source=1,
                                 target_birth=200)
    two_to_one = _adoption_alias("2", "1", camera_id="CAM-05", source_id=1, old_birth=200,
                                 target_camera="CAM-02", target_source=0, target_birth=100,
                                 start=6, end=8, adoption_frame=9, event_time=500)
    try:
        _alias_index([one_to_two, two_to_one])
    except AdoptionAliasError as error:
        assert "cycle" in str(error)
    else:
        raise AssertionError("cyclic adoption lineage must be rejected")

    conflicting = _adoption_alias("1", "3", old_birth=100, target_birth=300, event_time=400)
    try:
        _alias_index([one_to_two, conflicting])
    except AdoptionAliasError as error:
        assert "conflicting duplicate" in str(error)
    else:
        raise AssertionError("conflicting mappings for one raw track must be rejected")

    same_camera_merge = _adoption_alias("3", "2", old_birth=110, target_birth=200, event_time=401)
    try:
        _alias_index([one_to_two, same_camera_merge])
    except AdoptionAliasError as error:
        assert "merge distinct same-camera tracks" in str(error)
    else:
        raise AssertionError("overlapping distinct same-camera tracks must not be merged")

    cross_camera_merge = _adoption_alias("6", "2", camera_id="CAM-05", source_id=1,
        old_birth=120, target_camera="CAM-05", target_source=1, target_birth=200,
        event_time=402)
    try:
        _alias_index([one_to_two, cross_camera_merge])
    except AdoptionAliasError as error:
        assert "cross-camera identity merge" in str(error)
    else:
        raise AssertionError("ambiguous cross-camera convergences must be rejected")


def test_adoption_alias_builder_binds_real_adoption_and_peer_tracklet_evidence(tmp_path: Path) -> None:
    raw = {"type": 3, "targetId": 1, "adoptedId": 2, "frameNum": 6,
           "targetIDTimeStamp": 200, "messageTimeStamp": 300, "fromCamId": "broker;/trck/cam1"}
    old_track = {"type": 1, "targetId": 1, "frameNum": 5, "targetIDTimeStamp": 100,
                 "messageTimeStamp": 250, "fromCamId": "broker;/trck/cam1"}
    peer = {"type": 1, "targetId": 2, "frameNum": 5, "targetIDTimeStamp": 200,
            "messageTimeStamp": 250, "fromCamId": "broker;/trck/cam2"}
    alias_map = tmp_path / "alias-source.json"
    alias_map.write_text(json.dumps({"scope": "private offline replay presentation only", "aliases": [{
        "camera_id": "CAM-02", "source_id": 0, "mqtt_topic": "/trck/cam1",
        "raw_id_before_adoption": 1, "display_id_after_adoption": 2, "adoption_frame": 6,
        "display_alias_valid_from_frame": 0, "display_alias_valid_until_frame_inclusive": 5,
        "old_target_track_birth_timestamp_ns": 100, "event_message_timestamp_ns": 300,
        "event_targetIDTimeStamp_ns": 200, "adopted_peer_tracklet_matches": 1, "raw_event": raw,
    }]}), encoding="utf-8")
    mqtt_decoded = tmp_path / "mqtt.jsonl"
    mqtt_decoded.write_text(json.dumps(raw) + "\n" + json.dumps(old_track) + "\n" +
                            json.dumps(peer) + "\n", encoding="utf-8")
    event_log = tmp_path / "kafka.log"
    event_log.write_text("preserved source event log\n", encoding="utf-8")
    mqtt_raw = tmp_path / "mqtt.raw"
    mqtt_raw.write_bytes(b"private raw mqtt fixture")
    manifest = build_bound_manifest(alias_map, mqtt_decoded, event_log, mqtt_raw)
    assert manifest["scope"] == "offline_replay_only"
    assert manifest["source_log_sha256"] == sha256_file(event_log)
    assert manifest["aliases"][0]["target_camera_id"] == "CAM-05"
    assert manifest["aliases"][0]["raw_event"] == raw


def test_alias_manifest_hash_must_match_the_imported_database(tmp_path: Path) -> None:
    manifest_path = tmp_path / "aliases.json"
    manifest_path.write_text(json.dumps({"schema_version": 1, "scope": "offline_replay_only",
        "source_log_sha256": "f" * 64, "aliases": [_adoption_alias("1", "2")]}), encoding="utf-8")
    try:
        AdoptionAliasIndex.from_file(manifest_path, expected_source_log_sha256="a" * 64,
            camera_mapping={"Camera1": ("CAM-02", 0), "Camera2": ("CAM-05", 1)})
    except AdoptionAliasError as error:
        assert "hash does not match" in str(error)
    else:
        raise AssertionError("aliases from a different replay must not be applied")
    manifest = {"schema_version": 1, "scope": "live", "source_log_sha256": "a" * 64,
                "aliases": [_adoption_alias("1", "2")]}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        AdoptionAliasIndex.from_file(manifest_path, expected_source_log_sha256="a" * 64,
            camera_mapping={"Camera1": ("CAM-02", 0), "Camera2": ("CAM-05", 1)})
    except AdoptionAliasError as error:
        assert "offline_replay_only" in str(error)
    else:
        raise AssertionError("retrospective alias sidecars must be rejected in live mode")


def _entrance_messages() -> list[dict]:
    def message(sensor: str, frame: int, entries: list[tuple[str, int]]) -> dict:
        result = _message(sensor, frame, entries[0][0],
                          timestamp=f"2026-10-18T19:12:{49 + frame // 30:02d}.000Z")
        result["objects"] = []
        for output_id, left in entries:
            result["objects"].append({
                "id": output_id, "type": "Person", "confidence": 0.75,
                "bbox": {"leftX": left, "topY": 100, "rightX": left + 50, "bottomY": 240},
                "info": {"footLocation2D": f"{left + 25},240", "footLocation3D": f"{left / 100},-2.0"},
            })
        return result
    return [
        message("Camera1", 5, [("1", 100), ("3", 300), ("4", 500)]),
        message("Camera2", 5, [("0", 110), ("2", 310), ("5", 510)]),
        message("Camera1", 6, [("2", 101), ("5", 301), ("0", 501)]),
        message("Camera2", 6, [("0", 111), ("2", 311), ("5", 511)]),
    ]


def test_api_sends_hash_bound_raw_and_display_ids_without_mutating_observations(tmp_path: Path) -> None:
    source = tmp_path / "entrance-events.log"
    _write_log(source, _entrance_messages())
    database = tmp_path / "entrance.sqlite3"
    imported = import_event_log(source, database, fps=30, expected_frames=600,
                                camera_mapping=CAMERA_MAPS["entrance-room"])
    rules = [
        _adoption_alias("1", "2", old_birth=100, target_birth=200, event_time=300),
        _adoption_alias("3", "5", old_birth=110, target_birth=210, event_time=301),
        _adoption_alias("4", "0", old_birth=120, target_birth=220, event_time=302),
    ]
    manifest = {"schema_version": 1, "scope": "offline_replay_only",
                "source_log_sha256": imported["source_log_sha256"], "aliases": rules}
    alias_path = tmp_path / "aliases.json"
    alias_path.write_text(json.dumps(manifest), encoding="utf-8")
    db_before = sha256_file(database)
    app = _qapp()
    with TestClient(create_app(database, adoption_alias_path=alias_path)) as client:
        summary = client.get("/api/v1/summary").json()
        assert [(item["camera_id"], item["source_id"]) for item in summary["camera_mapping"]] == [
            ("CAM-02", 0), ("CAM-05", 1)
        ]
        assert summary["distinct_mv3dt_output_ids_union"] == 6
        assert summary["display_aliases"]["available"] is True
        assert client.get("/api/v1/observations", params={"camera_id": "CAM-02", "limit": 1}).status_code == 200
        frame5 = client.get("/api/v1/frames/5").json()
        cam02 = {row["raw_global_id"]: row for row in frame5["observations"] if row["camera_id"] == "CAM-02"}
        assert {key: value["display_global_id"] for key, value in cam02.items()} == {
            "1": "2", "3": "5", "4": "0"
        }
        assert all(row["display_alias_applied"] for row in cam02.values())
        assert cam02["1"]["bbox"] == {"leftX": 100.0, "topY": 100.0, "rightX": 150.0, "bottomY": 240.0}
        assert cam02["1"]["event_timestamp"] == frame5["observations"][0]["event_timestamp"]
        frame6 = client.get("/api/v1/frames/6").json()
        cam02_after = {row["raw_global_id"]: row for row in frame6["observations"] if row["camera_id"] == "CAM-02"}
        assert {key: value["display_global_id"] for key, value in cam02_after.items()} == {
            "2": "2", "5": "5", "0": "0"
        }
        assert not any(row["display_alias_applied"] for row in cam02_after.values())
        with client.websocket_connect("/ws/replay?speed=100&start_frame=5&end_frame=6") as socket:
            start = socket.receive_json()
            ws5, ws6, complete = [socket.receive_json() for _ in range(3)]
        assert start["display_aliasing_available"] is True
        assert {row["display_global_id"] for row in ws5["observations"] if row["camera_id"] == "CAM-02"} == {"0", "2", "5"}
        assert complete["last_frame_id"] == 6
    assert sha256_file(database) == db_before
    app.processEvents()


def test_qt_optional_alias_toggle_keeps_camera_rows_and_positions(tmp_path: Path) -> None:
    app = _qapp()
    map_path = tmp_path / "entrance.png"
    image = QImage(1200, 896, QImage.Format.Format_RGB32)
    image.fill(QColor("#e8e1d5"))
    assert image.save(str(map_path))
    window = MainWindow(map_path, "ws://127.0.0.1:9/ws/replay", auto_start=False,
                        room_title="Entrance Room")
    window._on_message(json.dumps({"type": "start", "display_aliasing_available": True,
                                  "display_alias_count": 3, "transform_world_to_pixel_row_major": [
                                      100, 0, 0, 0, -100, 0, 0, 0, 1
                                  ]}))
    assert window.alias_toggle.isEnabled() and window.alias_toggle.isChecked()
    observations = []
    for camera, source, rows in [
        ("CAM-02", 0, [("1", "2", 100), ("3", "5", 300), ("4", "0", 500)]),
        ("CAM-05", 1, [("0", "0", 110), ("2", "2", 310), ("5", "5", 510)]),
    ]:
        for raw_id, display_id, left in rows:
            observations.append({
                "camera_id": camera, "source_id": source, "frame_id": 5,
                "mv3dt_global_id": raw_id, "raw_global_id": raw_id,
                "display_global_id": display_id, "display_alias_applied": raw_id != display_id,
                "display_alias_lineage": ([{"event_type": "ADOPTED_ID", "adoption_frame": 6}]
                                           if raw_id != display_id else []),
                "event_timestamp": "2026-10-18T19:12:49.756Z",
                "media_time_seconds_nominal": 5 / 30,
                "bbox": {"leftX": left, "topY": 100, "rightX": left + 50, "bottomY": 240},
                "world_position_unverified_units": [left / 100, -2.0],
            })
    original_positions = [row["world_position_unverified_units"][:] for row in observations]
    window._on_message(json.dumps({"type": "frame", "frame_id": 5, "observations": observations,
        "frame_messages": [{"camera_id": "CAM-02", "source_id": 0}, {"camera_id": "CAM-05", "source_id": 1}]}))
    app.processEvents()
    assert window.table.rowCount() == 6
    assert len(window.map_canvas.observations) == 6
    assert {window.table.item(row, 0).text() for row in range(6)} == {"CAM-02", "CAM-05"}
    # Per camera, three visually distinct boxes remain three distinct display labels.
    for camera in ("CAM-02", "CAM-05"):
        camera_rows = [row for row in range(window.table.rowCount()) if window.table.item(row, 0).text() == camera]
        assert len({window.table.item(row, 3).text() for row in camera_rows}) == 3
    assert {window.table.item(row, 2).text() for row in range(6)} == {"0", "1", "2", "3", "4", "5"}
    assert {window.table.item(row, 3).text() for row in range(6)} == {"0", "2", "5"}
    aliased_state = display_state(window.active, current_frame=5, use_display_aliases=True)
    assert aliased_state["camera_observations"] == 6
    assert aliased_state["unique_global_ids"] == 6
    assert aliased_state["display_unique_ids"] == 3
    assert len(aliased_state["primary_markers"]) == 3
    assert [row["world_position_unverified_units"] for row in window.map_canvas.observations] == original_positions
    window.alias_toggle.setChecked(False)
    app.processEvents()
    raw_state = display_state(window.active, current_frame=5, use_display_aliases=False)
    assert len(raw_state["primary_markers"]) == 6
    assert window.table.item(0, 2).text() == window.table.item(0, 3).text()
    window.alias_toggle.setChecked(True)
    post_adoption = []
    for camera, source, rows in [
        ("CAM-02", 0, [("2", 2), ("5", 300), ("0", 500)]),
        ("CAM-05", 1, [("0", 110), ("2", 310), ("5", 510)]),
    ]:
        for raw_id, left in rows:
            post_adoption.append({
                "camera_id": camera, "source_id": source, "frame_id": 6,
                "mv3dt_global_id": raw_id, "raw_global_id": raw_id,
                "display_global_id": raw_id, "display_alias_applied": False,
                "display_alias_lineage": [], "event_timestamp": "2026-10-18T19:12:49.790Z",
                "media_time_seconds_nominal": 6 / 30,
                "bbox": {"leftX": left, "topY": 100, "rightX": left + 50, "bottomY": 240},
                "world_position_unverified_units": [left / 100, -2.0],
            })
    window._on_message(json.dumps({"type": "frame", "frame_id": 6, "observations": post_adoption,
        "frame_messages": [{"camera_id": "CAM-02", "source_id": 0}, {"camera_id": "CAM-05", "source_id": 1}]}))
    app.processEvents()
    # Superseded pre-adoption rows remain in the in-memory/raw store but are not
    # rendered as duplicate active identities beside the adopted current rows.
    assert len(window.active) == 9
    assert {row["raw_global_id"] for row in window.active if row["camera_id"] == "CAM-02"} == {
        "0", "1", "2", "3", "4", "5"
    }
    assert window._suppressed_adoption_rows == 3
    assert window.table.rowCount() == 6
    for camera in ("CAM-02", "CAM-05"):
        camera_rows = [row for row in range(window.table.rowCount()) if window.table.item(row, 0).text() == camera]
        assert len({window.table.item(row, 3).text() for row in camera_rows}) == 3
    assert "3 superseded pre-adoption raw row(s)" in window.review_status.text()
    window.close()


def test_qt_full_review_video_mode_embeds_player_and_raw_display_table(tmp_path: Path, monkeypatch) -> None:
    from services.mv3dt_staging import qt_ui

    app = _qapp()
    map_path = tmp_path / "entrance.png"
    image = QImage(1200, 896, QImage.Format.Format_RGB32)
    image.fill(QColor("#e8e1d5"))
    assert image.save(str(map_path))
    video_path = tmp_path / "three-panel.mp4"
    video_path.write_bytes(b"test media path fixture")

    class Signal:
        def connect(self, _callback) -> None:
            pass

    real_media_player = qt_ui.QMediaPlayer

    class MediaPlayerFixture:
        PlaybackState = real_media_player.PlaybackState

        def __init__(self, _parent) -> None:
            self.durationChanged = Signal()
            self.positionChanged = Signal()
            self.playbackStateChanged = Signal()
            self.errorOccurred = Signal()
            self._source = None

        def setAudioOutput(self, _output) -> None:
            pass

        def setVideoOutput(self, _widget) -> None:
            pass

        def setSource(self, source) -> None:
            self._source = source

        def source(self):
            return self._source

    monkeypatch.setattr(qt_ui, "QMediaPlayer", MediaPlayerFixture)
    window = MainWindow(map_path, "ws://127.0.0.1:9/ws/replay?speed=1", auto_start=False,
                        room_title="Entrance Room B", review_video_path=video_path)
    assert window.review_video_widget is not None
    assert window.video_player is not None
    assert window.video_player.source().toLocalFile() == str(video_path.resolve())
    assert window.video_play_button is not None and window.video_slider is not None
    assert window.table.horizontalHeaderItem(2).text() == "Raw Global ID"
    assert window.table.horizontalHeaderItem(3).text() == "Display ID"
    assert not window.alias_toggle.isEnabled()
    window.close()
    app.processEvents()

