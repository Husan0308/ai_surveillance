import pytest

from scripts.run_verified_candidate_two_source_replay import physical_camera_row


@pytest.fixture
def camera_mapping():
    return {0: "CAM-02", 1: "CAM-05"}, {"CAM-01": "CAM-02", "CAM-02": "CAM-05"}


def test_physical_mapping_preserves_runtime_alias_and_maps_canonical_objects(camera_mapping):
    source_ids, aliases = camera_mapping
    row = {
        "camera_id": "CAM-01",
        "frame_num": 113,
        "objects": [{"camera_id": "CAM-01", "native_track_id": 3, "bbox": [1, 2, 3, 4]}],
    }
    mapped = physical_camera_row(row, source_ids, aliases)

    assert row["camera_id"] == "CAM-01"
    assert mapped["runtime_camera_id"] == "CAM-01"
    assert mapped["camera_id"] == "CAM-02"
    assert mapped["physical_camera_id"] == "CAM-02"
    assert mapped["objects"][0]["runtime_camera_id"] == "CAM-01"
    assert mapped["objects"][0]["camera_id"] == "CAM-02"


def test_source_id_is_authoritative_when_row_has_no_camera_alias(camera_mapping):
    source_ids, aliases = camera_mapping
    mapped = physical_camera_row({"source_id": 1, "frame": 10}, source_ids, aliases)

    assert mapped["physical_camera_id"] == "CAM-05"
    assert mapped["camera_id"] == "CAM-05"


def test_conflicting_source_id_and_runtime_alias_fails_closed(camera_mapping):
    source_ids, aliases = camera_mapping

    with pytest.raises(RuntimeError, match="source/camera alias conflict"):
        physical_camera_row({"source_id": 0, "camera_id": "CAM-02"}, source_ids, aliases)
