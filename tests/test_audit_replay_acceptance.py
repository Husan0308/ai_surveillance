from collections import defaultdict

from scripts.dev_room_mv3dt.audit_replay_acceptance import (
    canonical_false_merge_diagnostics,
    native_fragment_overlap_diagnostics,
    unnecessary_allocation_diagnostics,
    presence_frame,
    presence_disagreement_status,
    preview_health_status,
)
from services.mv3dt_room.presence import CurrentPresencePositionFilter


def observation(camera, frame, app_id, native_id, world, bbox, confidence=0.8):
    return {
        "camera_id": camera,
        "frame": frame,
        "application_id": app_id,
        "native_track_id": native_id,
        "world": world,
        "bbox": bbox,
        "confidence": confidence,
    }


def test_same_person_native_fragments_are_diagnostic_and_reduce_to_one_marker():
    rows = [
        observation("CAM-01", 10, "Person_01", 17, [1.0, 2.0], [10, 10, 100, 180], 0.91),
        observation("CAM-01", 10, "Person_01", 23, [1.4, 2.2], [14, 12, 96, 175], 0.82),
    ]
    by_frame = defaultdict(list, {10: rows})
    overlap = native_fragment_overlap_diagnostics(by_frame, 11)
    labels = {
        (10, "CAM-01", "17"): "tester-a",
        (10, "CAM-01", "23"): "tester-a",
    }
    merges = canonical_false_merge_diagnostics(by_frame, labels)

    active, positions, marker_ids = presence_frame(rows, 10, CurrentPresencePositionFilter())

    assert len(overlap) == 1
    assert overlap[0]["native_track_ids"] == ["17", "23"]
    assert overlap[0]["active_observation_count_for_camera_identity"] == 1
    assert overlap[0]["renderable_bev_marker_count_for_identity"] == 1
    assert merges["confirmed_count"] == 0
    assert merges["unresolved_spatial_candidate_count"] == 0
    assert len(active) == 1
    assert set(positions) == {"Person_01"}
    assert marker_ids == ["Person_01"]
    assert len(marker_ids) - len(set(marker_ids)) == 0


def test_distinct_labeled_people_sharing_one_canonical_id_is_a_false_merge():
    rows = [
        observation("CAM-01", 20, "Person_01", 31, [1.0, 2.0], [10, 10, 80, 170]),
        observation("CAM-01", 20, "Person_01", 32, [5.0, 2.0], [250, 10, 330, 170]),
    ]
    by_frame = defaultdict(list, {20: rows})
    labels = {
        (20, "CAM-01", "31"): "tester-a",
        (20, "CAM-01", "32"): "tester-b",
    }

    result = canonical_false_merge_diagnostics(by_frame, labels)

    assert result["confirmed_count"] == 1
    assert result["confirmed_conflicts"][0]["physical_person_labels"] == ["tester-a", "tester-b"]


def test_same_physical_person_in_both_cameras_is_one_fused_marker_not_a_merge():
    rows = [
        observation("CAM-01", 40, "Person_01", 8, [1.0, 2.0], [10, 10, 90, 180]),
        observation("CAM-04", 40, "Person_01", 91, [1.2, 2.1], [20, 10, 100, 180]),
    ]
    by_frame = defaultdict(list, {40: rows})
    labels = {
        (40, "CAM-01", "8"): "tester-a",
        (40, "CAM-04", "91"): "tester-a",
    }

    merges = canonical_false_merge_diagnostics(by_frame, labels)
    active, positions, marker_ids = presence_frame(rows, 40, CurrentPresencePositionFilter())

    assert merges["confirmed_count"] == 0
    assert len(active) == 2
    assert set(positions) == {"Person_01"}
    assert marker_ids == ["Person_01"]


def test_stale_observation_cannot_leave_a_ghost_marker():
    source_rows = [observation("CAM-01", 50, "Person_01", 5, [1.0, 2.0], [10, 10, 80, 170])]
    position_filter = CurrentPresencePositionFilter()
    _, first_positions, _ = presence_frame(source_rows, 50, position_filter)
    active, positions, marker_ids = presence_frame(source_rows, 51, position_filter)

    assert set(first_positions) == {"Person_01"}
    assert active == []
    assert positions == {}
    assert marker_ids == []


def test_replay_skip_render_makes_preview_only_health_not_applicable():
    status, gate_value = preview_health_status("replay", False, {})

    assert status["applicable"] is False
    assert status["status"] == "N/A"
    assert gate_value is None


def test_skip_render_presence_telemetry_is_na_but_computed_presence_still_gates():
    status, gate_value = presence_disagreement_status(
        -1, -1, 2400, 0, source_mode="replay", visual_telemetry_supplied=False
    )
    assert status["applicable"] is False
    assert status["status"] == "N/A"
    assert gate_value is True

    _, bad_gate_value = presence_disagreement_status(
        -1, -1, 2400, 1, source_mode="replay", visual_telemetry_supplied=False
    )
    assert bad_gate_value is False


def test_available_visual_presence_telemetry_remains_strict():
    status, gate_value = presence_disagreement_status(
        0, 2400, 2400, 0, source_mode="replay", visual_telemetry_supplied=True
    )
    assert status["applicable"] is True
    assert gate_value is True

    _, incomplete_gate_value = presence_disagreement_status(
        0, 2399, 2400, 0, source_mode="replay", visual_telemetry_supplied=True
    )
    assert incomplete_gate_value is False


def test_partial_visual_presence_telemetry_does_not_hide_a_disagreement():
    _, passed = presence_disagreement_status(
        1, -1, 2400, 0, source_mode="replay", visual_telemetry_supplied=True
    )
    assert passed is False


def test_absent_live_visual_presence_telemetry_remains_required():
    _, passed = presence_disagreement_status(
        -1, -1, 2400, 0, source_mode="live", visual_telemetry_supplied=False
    )
    assert passed is False


def test_unresolved_cross_camera_observations_are_not_a_canonical_false_merge():
    rows = [
        observation("CAM-01", 40, "Unknown", 8, [1.0, 2.0], [10, 10, 90, 180]),
        observation("CAM-04", 40, "Unknown", 91, [12.0, 2.1], [20, 10, 100, 180]),
    ]
    merges = canonical_false_merge_diagnostics(defaultdict(list, {40: rows}), {})
    assert merges["confirmed_count"] == 0
    assert merges["unresolved_spatial_candidate_count"] == 0


def test_explicit_bad_preview_telemetry_remains_gating():
    status, gate_value = preview_health_status(
        "replay",
        True,
        {"CAM-03": {"state": "STALLED", "reconnects": 0, "stalled_periods": 1}},
    )

    assert status["applicable"] is True
    assert status["status"] == "FAIL"
    assert gate_value is False


def test_initial_allocations_for_two_observed_replay_people_are_not_unnecessary():
    allocations = [
        {"application_id": "Person_01", "reason": "positive_novelty_evidence"},
        {"application_id": "Person_02", "reason": "positive_novelty_evidence"},
    ]

    result = unnecessary_allocation_diagnostics(
        ["Person_01", "Person_02"], allocations
    )

    assert result["count"] == 0
    assert result["observed_identity_allocation_counts"] == {
        "Person_01": 1,
        "Person_02": 1,
    }


def test_orphaned_or_repeated_canonical_create_is_unnecessary():
    allocations = [
        {"application_id": "Person_01"},
        {"application_id": "Person_02"},
        {"application_id": "Person_03"},
        {"application_id": "Person_01"},
    ]

    result = unnecessary_allocation_diagnostics(
        ["Person_01", "Person_02"], allocations
    )

    assert result["count"] == 2
    assert [row["application_id"] for row in result["allocations"]] == [
        "Person_03",
        "Person_01",
    ]
