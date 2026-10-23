from __future__ import annotations

import numpy as np
import pytest

from services.mv3dt_staging.live_map_projection import (
    FrameJoinError,
    LiveNativeMapProjection,
    choose_marker_label_position,
)


def _projector(*, person_class_id=None, width=300, height=250):
    return LiveNativeMapProjection(
        {"cam02": np.eye(3), "cam05": np.eye(3)},
        layout_width=width,
        layout_height=height,
        person_class_id=person_class_id,
    )


def _record(source_id, frame, pts, objects):
    return {
        "source_id": source_id,
        "input_frame_index": frame,
        "source_pts_ns": pts,
        "objects": objects,
    }


def _object(tracker_id, box=(10, 20, 20, 30), class_id=0):
    return {
        "tracker_id": tracker_id,
        "class_id": class_id,
        "left": box[0], "top": box[1], "width": box[2], "height": box[3],
    }


def test_live_projection_requires_exact_source_frame_and_pts_join():
    projector = _projector()
    record = _record(0, 7, 700, [_object(11)])
    with pytest.raises(FrameJoinError, match="source ID"):
        projector.update("CAM-02", record, cached_source_id=1,
                         cached_frame_index=7, cached_source_pts_ns=700)
    with pytest.raises(FrameJoinError, match="frame index"):
        projector.update("CAM-02", record, cached_source_id=0,
                         cached_frame_index=8, cached_source_pts_ns=700)
    with pytest.raises(FrameJoinError, match="PTS"):
        projector.update("CAM-02", record, cached_source_id=0,
                         cached_frame_index=7, cached_source_pts_ns=701)


def test_live_projection_uses_bottom_center_and_camera_local_ids_are_scoped():
    projector = _projector()
    cam02 = projector.update(
        "CAM-02", _record(0, 1, 100, [_object(5)]), cached_source_id=0,
        cached_frame_index=1, cached_source_pts_ns=100, now_mono_ns=1_000_000_000,
    )
    cam05 = projector.update(
        "CAM-05", _record(1, 1, 100, [_object(5)]), cached_source_id=1,
        cached_frame_index=1, cached_source_pts_ns=100, now_mono_ns=1_000_000_000,
    )
    assert cam02["markers"][0]["local_tracker_id"] == "5"
    assert cam05["markers"][0]["local_tracker_id"] == "5"
    assert cam02["markers"][0]["layout_xy"] == [20.0, 50.0]
    assert cam02["markers"][0]["foot_anchor_inference_xy"] == [20.0, 50.0]
    assert cam02["markers"][0]["camera_id"] != cam05["markers"][0]["camera_id"]


def test_out_of_bounds_markers_are_flagged_not_clamped():
    projector = _projector(width=100, height=100)
    state = projector.update(
        "CAM-02", _record(0, 1, 100, [_object(9, box=(180, 180, 30, 30))]),
        cached_source_id=0, cached_frame_index=1, cached_source_pts_ns=100,
    )
    assert state["markers"] == []
    assert state["out_of_bounds_ids"] == ["9"]
    assert projector.counters["CAM-02"]["out_of_bounds"] == 1


def test_duplicate_local_id_in_one_frame_is_flagged_and_not_projected():
    projector = _projector()
    state = projector.update(
        "CAM-02", _record(0, 1, 100, [_object(3), _object(3, box=(30, 20, 20, 30))]),
        cached_source_id=0, cached_frame_index=1, cached_source_pts_ns=100,
    )
    assert state["markers"] == []
    assert state["duplicate_local_ids"] == ["3"]
    assert projector.counters["CAM-02"]["duplicate_local_ids"] == 1


def test_person_class_filter_and_monotonicity_are_enforced():
    projector = _projector(person_class_id=2)
    first = projector.update(
        "CAM-02", _record(0, 1, 100, [_object(1, class_id=2), _object(2, class_id=1)]),
        cached_source_id=0, cached_frame_index=1, cached_source_pts_ns=100,
    )
    assert [row["local_tracker_id"] for row in first["markers"]] == ["1"]
    with pytest.raises(FrameJoinError, match="non-monotonic"):
        projector.update(
            "CAM-02", _record(0, 1, 101, [_object(1, class_id=2)]),
            cached_source_id=0, cached_frame_index=1, cached_source_pts_ns=101,
        )


def test_large_jump_is_flagged_and_stale_snapshot_clears_markers():
    projector = _projector()
    projector.update(
        "CAM-02", _record(0, 1, 100_000_000, [_object(1)]),
        cached_source_id=0, cached_frame_index=1,
        cached_source_pts_ns=100_000_000, now_mono_ns=1_000_000_000,
    )
    jumped = projector.update(
        "CAM-02", _record(0, 2, 150_000_000, [_object(1, box=(180, 20, 20, 30))]),
        cached_source_id=0, cached_frame_index=2,
        cached_source_pts_ns=150_000_000, now_mono_ns=1_050_000_000,
    )
    assert jumped["jump_candidate_ids"] == ["1"]
    stale = projector.snapshot("CAM-02", now_mono_ns=1_900_000_000)
    assert stale["status"] == "STALE"
    assert stale["markers"] == []


def test_map_label_positions_stay_in_bounds_and_avoid_markers_and_each_other():
    bounds = (20, 30, 240, 160)
    markers = [(115, 94, 16, 16), (122, 97, 16, 16), (135, 103, 16, 16)]
    first = choose_marker_label_position((123, 102), (56, 18), bounds, markers)
    assert first is not None
    occupied = markers + [(first[0], first[1], 56, 18)]
    second = choose_marker_label_position((129, 108), (56, 18), bounds, occupied)
    assert second is not None

    for x, y in (first, second):
        assert bounds[0] <= x <= bounds[0] + bounds[2] - 56
        assert bounds[1] <= y <= bounds[1] + bounds[3] - 18
    for x, y in (first, second):
        for ox, oy, ow, oh in markers:
            assert not (x < ox + ow + 3 and x + 56 + 3 > ox
                        and y < oy + oh + 3 and y + 18 + 3 > oy)
    assert not (first[0] < second[0] + 56 + 3 and first[0] + 56 + 3 > second[0]
                and first[1] < second[1] + 18 + 3 and first[1] + 18 + 3 > second[1])


def test_map_label_layout_fails_closed_when_pane_is_too_small():
    position = choose_marker_label_position(
        (5, 5), (30, 20), (0, 0, 20, 10), [],
    )
    assert position is None
