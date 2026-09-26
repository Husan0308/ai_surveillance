from services.mv3dt_room.presence import (CurrentPresencePositionFilter, active_observations, assert_presence_contract, fused_world_positions)


def observation(camera, frame, identity, world, native, confidence=0.9):
    return {
        "camera_id": camera,
        "frame": frame,
        "application_id": identity,
        "world": world,
        "native_track_id": native,
        "confidence": confidence,
    }


def test_presence_uses_latest_frame_and_fuses_one_marker_per_application_id():
    rows = [
        observation("CAM-01", 10, "Person_01", [1.0, 2.0], 4),
        observation("CAM-04", 10, "Person_01", [1.2, 2.2], 9),
        observation("CAM-01", 10, "Person_02", [4.0, 5.0], 6),
        observation("CAM-01", 9, "Person_01", [99.0, 99.0], 1),
    ]
    frame, active = active_observations(rows)
    assert frame == 10
    assert len(active) == 3
    assert fused_world_positions(active)["Person_01"] == (1.1, 2.1)
    result = assert_presence_contract(active, {"Person_01", "Person_02"})
    assert result["no_stale_last_position_marker"]


def test_absent_application_id_cannot_keep_a_marker_alive():
    rows = [observation("CAM-01", 20, "Person_01", [1.0, 2.0], 4)]
    _, active = active_observations(rows, 21)
    assert active == []
    result = assert_presence_contract(active, set())
    assert result["rendered_ids"] == []


def test_current_position_filter_smooths_only_while_currently_visible():
    position_filter = CurrentPresencePositionFilter(alpha=0.30)
    first = position_filter.update([observation("CAM-01", 1, "Person_01", [0.0, 0.0], 1)])
    assert first == {"Person_01": (0.0, 0.0)}
    second = position_filter.update([observation("CAM-04", 2, "Person_01", [6.0, 0.0], 2)])
    assert second["Person_01"] == (1.7999999999999998, 0.0)
    assert position_filter.update([]) == {}
    # A later reacquisition starts from its current observation, never from a hidden ghost.
    assert position_filter.update([observation("CAM-04", 5, "Person_01", [9.0, 1.0], 3)]) == {"Person_01": (9.0, 1.0)}
