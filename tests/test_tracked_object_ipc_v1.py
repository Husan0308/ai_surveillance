from __future__ import annotations

import json

from services.camera_v11.tracked_object_ipc_v1 import read_camera_snapshot, read_snapshot


def _snapshot(camera="CAM-01", frame=100, pts=123456789, **changes):
    row = {
        "schema_version": 1,
        "camera_id": camera,
        "generation": "test-session",
        "frame_num": frame,
        "pts_ns": pts,
        "timestamp_monotonic_ns": 9_000_000_000,
        "timestamp_epoch_ns": 10_000_000_000,
        "source_frame_width": 2560,
        "source_frame_height": 1440,
        "detector_proposals": [{
            "camera_id": camera, "frame_num": frame, "pts_ns": pts,
            "proposal_index": 0, "class_id": 0, "confidence": 0.83,
            "bbox": [20, 30, 120, 240], "source_frame_width": 2560,
            "source_frame_height": 1440,
        }],
        "objects": [{
            "camera_id": camera, "frame_num": frame, "pts_ns": pts,
            "native_track_id": 5, "bbox": [10, 20, 110, 220],
            "bbox_source": "PGIE_SHADOW_FALLBACK", "detector_confidence": 0.75,
            "tracker_confidence": 0.61, "generation": "test-session",
        }],
    }
    row.update(changes)
    return row


def test_current_valid_object_snapshot_is_read_from_atomic_latest_file(tmp_path, monkeypatch):
    monkeypatch.setenv("V11_TRACKED_OBJECT_FEED_DIR", str(tmp_path))
    (tmp_path / "v11_tracked_objects_cam01_v1.json").write_text(json.dumps(_snapshot()))
    row = read_camera_snapshot("CAM-01", now_monotonic_ns=9_100_000_000)
    assert row is not None
    assert row["objects"][0]["bbox_source"] == "PGIE_SHADOW_FALLBACK"
    assert row["detector_proposals"][0]["confidence"] == 0.83
    assert row["detector_proposals"][0]["source_frame_width"] == 2560
    assert row["objects"][0]["bbox_coordinate_width"] == 2560
    assert row["snapshot_age_ms"] == 100.0


def test_native_source_and_pipeline_bbox_dimensions_are_kept_distinct(tmp_path, monkeypatch):
    monkeypatch.setenv("V11_TRACKED_OBJECT_FEED_DIR", str(tmp_path))
    row = _snapshot(
        source_frame_width=3200, source_frame_height=1800,
        pipeline_width=2560, pipeline_height=1440,
        bbox_coordinate_space="pipeline", bbox_coordinate_width=2560,
        bbox_coordinate_height=1440,
    )
    row["objects"][0].update({"bbox": [1280, 720, 2560, 1440],
                              "bbox_coordinate_space": "pipeline",
                              "bbox_coordinate_width": 2560,
                              "bbox_coordinate_height": 1440})
    row["detector_proposals"][0].update({"bbox_coordinate_space": "pipeline",
                                         "bbox_coordinate_width": 2560,
                                         "bbox_coordinate_height": 1440})
    (tmp_path / "v11_tracked_objects_cam01_v1.json").write_text(json.dumps(row))
    clean = read_camera_snapshot("CAM-01", now_monotonic_ns=9_100_000_000)
    assert clean is not None
    assert (clean["source_frame_width"], clean["source_frame_height"]) == (3200, 1800)
    assert (clean["pipeline_width"], clean["pipeline_height"]) == (2560, 1440)
    assert clean["objects"][0]["bbox_coordinate_width"] == 2560
    assert clean["detector_proposals"][0]["bbox_coordinate_width"] == 2560


def test_tracked_bbox_is_validated_against_its_coordinate_space_not_native_source(tmp_path, monkeypatch):
    monkeypatch.setenv("V11_TRACKED_OBJECT_FEED_DIR", str(tmp_path))
    row = _snapshot(source_frame_width=3200, source_frame_height=1800,
                    bbox_coordinate_width=2560, bbox_coordinate_height=1440)
    row["objects"][0].update({"bbox": [2500, 20, 2600, 100],
                              "bbox_coordinate_width": 2560,
                              "bbox_coordinate_height": 1440})
    (tmp_path / "v11_tracked_objects_cam01_v1.json").write_text(json.dumps(row))
    assert read_camera_snapshot("CAM-01", now_monotonic_ns=9_100_000_000) is None


def test_pipeline_space_dimensions_cannot_be_mislabeled_as_native_source(tmp_path, monkeypatch):
    monkeypatch.setenv("V11_TRACKED_OBJECT_FEED_DIR", str(tmp_path))
    row = _snapshot(
        source_frame_width=3200, source_frame_height=1800,
        pipeline_width=2560, pipeline_height=1440,
        bbox_coordinate_space="pipeline",
        bbox_coordinate_width=3200, bbox_coordinate_height=1800,
    )
    row["objects"][0].update({
        "bbox_coordinate_space": "pipeline",
        "bbox_coordinate_width": 3200,
        "bbox_coordinate_height": 1800,
    })
    row["detector_proposals"][0].update({
        "bbox_coordinate_space": "pipeline",
        "bbox_coordinate_width": 3200,
        "bbox_coordinate_height": 1800,
    })
    (tmp_path / "v11_tracked_objects_cam01_v1.json").write_text(json.dumps(row))
    assert read_camera_snapshot("CAM-01", now_monotonic_ns=9_100_000_000) is None


def test_invalid_or_wrong_pts_diagnostic_proposal_is_omitted_without_hiding_tracker(tmp_path, monkeypatch):
    monkeypatch.setenv("V11_TRACKED_OBJECT_FEED_DIR", str(tmp_path))
    path = tmp_path / "v11_tracked_objects_cam01_v1.json"
    row = _snapshot()
    row["detector_proposals"] = [row["detector_proposals"][0] | {"pts_ns": 7}]
    path.write_text(json.dumps(row))
    clean = read_camera_snapshot("CAM-01", now_monotonic_ns=9_100_000_000)
    assert clean is not None
    assert clean["objects"][0] == row["objects"][0] | {
        "bbox_coordinate_width": 2560, "bbox_coordinate_height": 1440}
    assert clean["detector_proposals"] == []
    row = _snapshot()
    row["detector_proposals"] = [row["detector_proposals"][0] | {"bbox": [10, 10, float("nan"), 20]}]
    path.write_text(json.dumps(row))
    clean = read_camera_snapshot("CAM-01", now_monotonic_ns=9_100_000_000)
    assert clean is not None
    assert len(clean["objects"]) == len(row["objects"])
    assert clean["objects"][0]["native_track_id"] == row["objects"][0]["native_track_id"]
    assert clean["detector_proposals"] == []


def test_stale_wrong_camera_wrong_pts_and_duplicate_native_ids_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("V11_TRACKED_OBJECT_FEED_DIR", str(tmp_path))
    path = tmp_path / "v11_tracked_objects_cam01_v1.json"
    good = _snapshot()
    path.write_text(json.dumps(good))
    assert read_camera_snapshot("CAM-01", now_monotonic_ns=9_300_000_001) is None
    wrong = _snapshot(camera="CAM-04")
    path.write_text(json.dumps(wrong))
    assert read_camera_snapshot("CAM-01", now_monotonic_ns=9_100_000_000) is None
    mismatch = _snapshot(objects=[good["objects"][0] | {"pts_ns": 123}])
    path.write_text(json.dumps(mismatch))
    assert read_camera_snapshot("CAM-01", now_monotonic_ns=9_100_000_000) is None
    duplicate = _snapshot(objects=[good["objects"][0], good["objects"][0]])
    path.write_text(json.dumps(duplicate))
    assert read_camera_snapshot("CAM-01", now_monotonic_ns=9_100_000_000) is None


def test_snapshot_lists_latest_per_camera_and_marks_partial(tmp_path, monkeypatch):
    monkeypatch.setenv("V11_TRACKED_OBJECT_FEED_DIR", str(tmp_path))
    (tmp_path / "v11_tracked_objects_cam01_v1.json").write_text(json.dumps(_snapshot()))
    payload = read_snapshot(camera_ids=("CAM-01", "CAM-04"), max_age_sec=0.5,
                            now_monotonic_ns=9_100_000_000)
    assert payload["status"] == "partial"
    assert payload["online_cameras"] == ["CAM-01"]
    assert len(payload["tracked_objects"]) == 1
