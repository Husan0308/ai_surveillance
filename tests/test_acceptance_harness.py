from __future__ import annotations

import json

import pytest

from services.mv3dt_room.acceptance_harness import AcceptanceHarness
from services.mv3dt_room.room_pair_state import RoomPairState


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def row(camera: str, native: int, person: str, frame: int, world=None) -> dict:
    return {
        "camera_id": camera,
        "native_track_id": native,
        "application_id": person,
        "global_person_id": person,
        "source_frame_number": frame,
        "source_timestamp": f"2026-09-24T10:00:{frame % 60:02d}Z",
        "bbox": [100.0, 100.0, 200.0, 300.0],
        "world": world,
        "confidence": 0.9,
        "visibility": 1.0,
    }


def harness(tmp_path, clock: FakeClock) -> AcceptanceHarness:
    value = AcceptanceHarness(
        tmp_path,
        monotonic=clock,
        wall_time=lambda: f"wall-{clock.now:.1f}",
        exit_grace_seconds=2,
        absent_confirmation_seconds=10,
        pre_exit_stability_seconds=5,
    )
    value.enable(True)
    return value


def ready_subtest_a(value: AcceptanceHarness) -> None:
    value.confirm_subtest_a_ready(operator_confirmed_distinct_people=True)


def test_annotations_are_operator_only_and_do_not_mutate_identity_rows(tmp_path):
    clock = FakeClock()
    value = harness(tmp_path, clock)
    value.start_subtest("A", "run-a")
    observation = row("CAM-01", 7, "Person_01", 10, [1.0, 2.0])
    original = json.loads(json.dumps(observation))

    events = value.assign("Tester_A", observation, "review.png")

    assert observation == original
    assert events[0]["event"] == "TESTER_A_ASSIGNED"
    assert events[0]["ground_truth_only"] is True
    assert value.testers["Tester_A"]["canonical_person_id"] == "Person_01"
    assert "Tester_A" not in observation
    log = (tmp_path / "run-a_subtest_a.jsonl").read_text()
    assert '"event":"TESTER_A_ASSIGNED"' in log


def test_subtest_a_records_cross_camera_crossing_and_stable_followthrough(tmp_path):
    clock = FakeClock()
    value = harness(tmp_path, clock)
    value.start_subtest("A", "run-a")
    value.assign("Tester_A", row("CAM-01", 1, "Person_01", 100, [0.0, 0.0]))
    value.assign("Tester_B", row("CAM-01", 2, "Person_02", 100, [4.0, 0.0]))
    ready_subtest_a(value)

    events = value.observe({"source_mode": "live", "people": [
        row("CAM-01", 1, "Person_01", 101, [0.0, 0.0]),
        row("CAM-01", 2, "Person_02", 101, [4.0, 0.0]),
    ]})
    assert any(event["event"] == "BOTH_PRESENT" for event in events)

    events = value.assign("Tester_A", row("CAM-04", 9, "Person_01", 200, [0.2, 0.0]))
    assert any(event["event"] == "CROSS_CAMERA_A" for event in events)
    events = value.assign("Tester_B", row("CAM-04", 8, "Person_02", 201, [4.1, 0.0]))
    assert any(event["event"] == "CROSS_CAMERA_B" for event in events)

    clock.advance(1)
    value.observe({"source_mode": "live", "people": [
        row("CAM-01", 1, "Person_01", 102, [0.0, 0.0]),
        row("CAM-01", 2, "Person_02", 102, [1.0, 0.0]),
    ]})
    clock.advance(2)
    value.observe({"source_mode": "live", "people": [
        row("CAM-01", 1, "Person_01", 103, [0.0, 0.0]),
        row("CAM-01", 2, "Person_02", 103, [2.2, 0.0]),
    ]})
    assert any(item["label"] == "near-pass/crossing" and item["done"] for item in value.checklist()["items"])

    clock.advance(31)
    value.observe({"source_mode": "live", "people": [
        row("CAM-01", 1, "Person_01", 104, [0.0, 0.0]),
        row("CAM-01", 2, "Person_02", 104, [4.0, 0.0]),
    ]})
    assert value.post_cross_stability_recorded
    assert value.failure is None
    assert "at least 2 minutes" in value.next_action()
    clock.advance(90)
    events = value.observe({"source_mode": "live", "people": [
        row("CAM-01", 1, "Person_01", 105, [0.0, 0.0]),
        row("CAM-01", 2, "Person_02", 105, [4.0, 0.0]),
    ]})
    assert any(event["event"] == "EVENT_SEQUENCE_COMPLETE" for event in events)
    assert "Subtest A evidence complete" == value.next_action()


def test_subtest_b_requires_absence_and_operator_confirmed_same_id_reacquisition(tmp_path):
    clock = FakeClock()
    value = harness(tmp_path, clock)
    value.start_subtest("B", "run-b")
    assigned = row("CAM-01", 11, "Person_01", 400, [2.0, 3.0])
    value.assign("Tester_A", assigned)
    with pytest.raises(RuntimeError, match="visibly tracked"):
        value.begin_exit_test("Tester_A")

    clock.advance(5)
    value.observe({"source_mode": "live", "people": [row("CAM-01", 11, "Person_01", 401, [2.0, 3.0])]})
    clock.advance(0.1)
    value.observe({"source_mode": "live", "people": [row("CAM-01", 11, "Person_01", 402, [2.0, 3.0])]})
    value.begin_exit_test("Tester_A")
    clock.advance(3)
    events = value.observe({"source_mode": "live", "people": []})
    assert any(event["event"] == "FULL_EXIT" for event in events)
    assert not any(event["event"] == "ABSENT_10S" for event in events)
    clock.advance(10)
    events = value.observe({"source_mode": "live", "people": []})
    assert any(event["event"] == "ABSENT_10S" for event in events)

    events = value.assign("Tester_A", row("CAM-04", 27, "Person_01", 500, [2.1, 3.0]))
    assert any(event["event"] == "REENTRY" for event in events)
    assert any(event["event"] == "REACQUISITION" for event in events)
    assert value.testers["Tester_A"]["same_person_reacquired"]
    assert value.failure is None
    clock.advance(102)
    events = value.observe({"source_mode": "live", "people": [row("CAM-04", 27, "Person_01", 501, [2.1, 3.0])]})
    assert any(event["event"] == "EVENT_SEQUENCE_COMPLETE" for event in events)
    assert all(item["done"] for item in value.checklist()["items"])


def test_identity_change_and_person_merge_are_acceptance_failures_only(tmp_path):
    clock = FakeClock()
    value = harness(tmp_path, clock)
    value.start_subtest("A", "run-errors")
    value.assign("Tester_A", row("CAM-01", 1, "Person_01", 1))
    value.assign("Tester_B", row("CAM-01", 2, "Person_02", 1))
    ready_subtest_a(value)
    change_events = value.assign("Tester_A", row("CAM-04", 8, "Person_02", 2))
    assert any(event["event"] == "IDENTITY_CHANGE" for event in change_events)
    assert value.failure["kind"] == "identity_change"

    merge_events = value.assign("Tester_B", row("CAM-04", 9, "Person_01", 3))
    assert any(event["event"] == "FALSE_MERGE" for event in merge_events)
    assert value.failure["kind"] == "identity_change"  # preserve first bad event


def test_unknown_and_non_room_rows_cannot_be_ground_truth_assignments(tmp_path):
    value = harness(tmp_path, FakeClock())
    value.start_subtest("A", "run-invalid")
    with pytest.raises(ValueError, match="canonical Person_XX"):
        value.assign("Tester_A", row("CAM-01", 1, "Unknown", 1))
    with pytest.raises(ValueError, match="CAM-01 or CAM-04"):
        value.assign("Tester_A", row("CAM-02", 1, "Person_01", 1))


def test_disabling_acceptance_mode_mid_subtest_invalidates_only_evaluator(tmp_path):
    clock = FakeClock()
    value = harness(tmp_path, clock)
    value.start_subtest("B", "run-disabled")
    value.assign("Tester_A", row("CAM-01", 1, "Person_01", 1))
    value.enable(False)
    assert value.failure == {"kind": "acceptance_mode_disabled_mid_subtest"}
    assert value.testers["Tester_A"]["canonical_person_id"] == "Person_01"


def test_subtest_a_waits_for_two_distinct_operator_observations_and_confirmation(tmp_path):
    clock = FakeClock()
    value = harness(tmp_path, clock)
    value.start_subtest("A", "run-a-waiting")
    assert value.started_monotonic is None
    assert value.checklist()["state"] == "WAITING"
    with pytest.raises(RuntimeError, match="WAITING FOR"):
        ready_subtest_a(value)

    value.assign("Tester_A", row("CAM-01", 7, "Person_01", 10))
    value.assign("Tester_B", row("CAM-01", 8, "Person_02", 10))
    assert value.started_monotonic is None
    with pytest.raises(RuntimeError, match="operator must confirm"):
        value.confirm_subtest_a_ready(False)
    assert value.started_monotonic is None

    ready_subtest_a(value)
    assert value.started_monotonic == clock.now
    assert value.operator_confirmed_distinct_people is True
    assert value.checklist()["state"] == "RUNNING"


def test_subtest_a_same_canonical_id_is_false_merge_at_start_after_operator_confirms_people(tmp_path):
    value = harness(tmp_path, FakeClock())
    value.start_subtest("A", "run-a-false-merge")
    value.assign("Tester_A", row("CAM-01", 7, "Person_01", 10))
    value.assign("Tester_B", row("CAM-01", 8, "Person_01", 10))
    event = value.confirm_subtest_a_ready(operator_confirmed_distinct_people=True)

    assert event["event"] == "FAIL_FALSE_MERGE_AT_START"
    assert value.failure["kind"] == "FAIL_FALSE_MERGE_AT_START"
    assert value.started_monotonic is None


def test_subtest_a_rejects_reusing_the_exact_same_bbox_observation(tmp_path):
    value = harness(tmp_path, FakeClock())
    value.start_subtest("A", "run-a-same-observation")
    same = row("CAM-01", 7, "Person_01", 10)
    value.assign("Tester_A", same)
    with pytest.raises(ValueError, match="distinct physical bbox"):
        value.assign("Tester_B", same)


def test_candidate_endpoint_is_read_only_acceptance_scoped_and_separate(tmp_path):
    state = RoomPairState(tmp_path)
    state.identity_dir.mkdir(parents=True, exist_ok=True)
    (state.identity_dir / "global_identity.jsonl").write_text(json.dumps({
        "camera_id": "CAM-01",
        "crop_provenance": {"source_frame_width": 2560, "source_frame_height": 1440},
    }) + "\n" + json.dumps({
        "camera_id": "CAM-04",
        "crop_provenance": {"source_frame_width": 3200, "source_frame_height": 1800},
    }) + "\n")
    trace_event = {
        "event": "identity_candidate_evaluated",
        "camera_id": "CAM-01",
        "native_track_id": 3,
        "observation_frame": 20,
        "best_candidate_similarity": 0.81,
        "decision_trace": {
            "observation": {"camera_id": "CAM-01", "native_track_id": 3, "frame": 20},
            "candidate_trace": [{
                "application_id": "Person_01",
                "cosine_best_gallery": 0.81,
                "gallery_size": 4,
                "world_distance_m": 0.4,
                "exact_rejection_reason": None,
            }],
            "novelty": {"reason": "existing_candidate_accepted"},
        },
    }
    (state.identity_dir / "identity_path_trace.jsonl").write_text(json.dumps(trace_event) + "\n")
    state._current_state = lambda: {
        "status": "running",
        "source_mode": "live",
        "session_id": "live-1",
        "active_frame_by_camera": {"CAM-01": 20},
        "people": [
            row("CAM-01", 3, "Person_01", 20, [1.0, 2.0]),
            row("CAM-02", 4, "Person_02", 20),
        ],
    }
    payload = state.acceptance_candidates()
    assert payload["acceptance_only"] is True
    assert payload["source_mode"] == "live"
    assert len(payload["candidates"]) == 1
    candidate = payload["candidates"][0]
    assert candidate["camera_id"] == "CAM-01"
    assert candidate["native_track_id"] == 3
    assert candidate["source_frame_width"] == 2560
    assert candidate["source_frame_height"] == 1440
    assert payload["coordinate_dimensions_ready"] is True
    assert payload["source_dimensions_by_camera"]["CAM-04"] == {"width": 3200, "height": 1800}
    assert "tester" not in candidate
    assert "ground_truth" not in candidate
    evidence = candidate["identity_evidence"]
    assert evidence["best_candidate_similarity"] == pytest.approx(0.81)
    assert evidence["candidates"][0]["application_id"] == "Person_01"
    assert "vector" not in json.dumps(evidence)


def test_live_acceptance_uses_verified_camera_dimensions_before_crop_log_flush(tmp_path):
    state = RoomPairState(tmp_path)
    state.identity_dir.mkdir(parents=True, exist_ok=True)
    state._current_state = lambda: {
        "status": "running",
        "source_mode": "live",
        "session_id": "live-dimensions",
        "people": [],
    }
    payload = state.acceptance_candidates()
    assert payload["coordinate_dimensions_ready"] is True
    assert payload["source_dimensions_by_camera"] == {
        "CAM-01": {"width": 2560, "height": 1440},
        "CAM-04": {"width": 3200, "height": 1800},
    }
    assert set(payload["source_dimensions_source"].values()) == {"verified_dev_room_profile_or_env"}


def test_subtest_b_requires_fresh_frames_and_records_pending_reacquisition_evidence(tmp_path):
    clock = FakeClock()
    value = harness(tmp_path, clock)
    value.start_subtest("B", "run-b-pending")
    with pytest.raises(RuntimeError, match="acceptance mode is disabled"):
        disabled = AcceptanceHarness(tmp_path / "disabled")
        disabled.start_subtest("B", "run-disabled")

    value.assign("Tester_A", row("CAM-01", 11, "Person_01", 100))
    clock.advance(5)
    assert not value.can_begin_exit_test("Tester_A")
    value.observe({"source_mode": "live", "people": [row("CAM-01", 11, "Person_01", 101)]})
    assert not value.can_begin_exit_test("Tester_A")
    value.observe({"source_mode": "live", "people": [row("CAM-01", 11, "Person_01", 102)]})
    assert value.can_begin_exit_test("Tester_A")
    value.begin_exit_test("Tester_A")
    clock.advance(3)
    value.observe({"source_mode": "live", "people": []})
    clock.advance(10)
    value.observe({"source_mode": "live", "people": []})

    returning = row("CAM-04", 27, "Unknown", 500, [2.1, 3.0])
    returning.update({
        "identity_state": "PENDING",
        "pending_age_frames": 8,
        "gallery_candidate": "Person_01",
        "identity_evidence": {"best_candidate_similarity": 0.78, "candidates": [{"application_id": "Person_01"}]},
    })
    events = value.assign("Tester_A", returning)
    pending = next(event for event in events if event["event"] == "REENTRY_PENDING")
    assert pending["identity_state"] == "PENDING"
    assert pending["pending_age_frames"] == 8
    assert pending["gallery_candidate"] == "Person_01"
    assert pending["identity_evidence"]["best_candidate_similarity"] == pytest.approx(0.78)
    assert value.testers["Tester_A"]["full_exit"] is not None

    resolved = row("CAM-04", 27, "Person_01", 501, [2.1, 3.0])
    resolved.update({"identity_state": "KNOWN", "decision_reason": "gallery_match", "cosine_similarity": 0.83})
    events = value.assign("Tester_A", resolved)
    reacquired = next(event for event in events if event["event"] == "REACQUISITION")
    assert reacquired["decision_reason"] == "gallery_match"
    assert reacquired["cosine_similarity"] == pytest.approx(0.83)
    assert reacquired["pending_duration_seconds"] >= 0
    assert reacquired["absence_duration_seconds"] >= 13
    assert reacquired["time_to_reacquire_seconds"] >= 10
    assert value.failure is None


def test_tester_observation_events_are_independent_per_tester_and_camera(tmp_path):
    clock = FakeClock()
    value = harness(tmp_path, clock)
    value.start_subtest("A", "run-two-people")
    value.assign("Tester_A", row("CAM-01", 1, "Person_01", 10))
    value.assign("Tester_B", row("CAM-01", 2, "Person_02", 10))
    ready_subtest_a(value)
    events = value.observe({"source_mode": "live", "people": [
        row("CAM-01", 1, "Person_01", 11),
        row("CAM-01", 2, "Person_02", 11),
    ]})
    assert {event["tester"] for event in events if event["event"] == "TESTER_OBSERVED"} == {"Tester_A", "Tester_B"}
