from __future__ import annotations

import json
import numpy as np
import pytest

from services.mv3dt_staging.automatic_provisional_global_id import (
    append_provisional_revision,
    resolve_provisional_assignments,
    score_and_assign_provisional_ids,
)
from services.mv3dt_staging.native_map_preview import _camera_panel


def _records() -> tuple[np.ndarray, list[dict], dict]:
    vectors = np.zeros((8, 512), dtype=np.float32)
    vectors[0:2, 0] = 1.0  # CAM-02 local A gallery
    vectors[2:4, 1] = 1.0  # CAM-02 local B gallery
    vectors[4:6, 0] = 1.0  # CAM-05 local X gallery
    vectors[6:8, 1] = 1.0  # CAM-05 local Y gallery
    records = []
    for camera, local_id, indexes, source_id in (
        ("CAM-02", "1", (0, 1), 0),
        ("CAM-02", "2", (2, 3), 0),
        ("CAM-05", "10", (4, 5), 1),
        ("CAM-05", "20", (6, 7), 1),
    ):
        for sample_no, index in enumerate(indexes):
            records.append({"camera_id": camera, "local_track_id": local_id,
                            "source_id": source_id, "frame": 1 + sample_no,
                            "source_pts_ns": (1 + sample_no) * 50_000_000,
                            "sample_id": f"S{index}", "bbox_inference_xywh": [0, 0, 10, 20]})
    tracks = {"CAM-02": {}, "CAM-05": {}}
    for camera, local_ids in (("CAM-02", ("1", "2")), ("CAM-05", ("10", "20"))):
        for frame in (1, 2):
            tracks[camera][frame] = [
                {"object_id": local_ids[0], "box": [0, 0, 10, 20]},
                {"object_id": local_ids[1], "box": [100, 0, 10, 20]},
            ]
    return vectors, records, tracks


def test_mutual_gallery_matches_get_one_to_one_provisional_ids_without_ground_truth():
    vectors, records, tracks = _records()
    result = score_and_assign_provisional_ids(
        vectors, records, tracks, minimum_separated_control_frames=2)

    assert len(result["same_camera_negative_like_controls"]) == 2
    assert len(result["matches"]) == 2
    assert [row["provisional_global_id"] for row in result["matches"]] == ["G001", "G002"]
    assert all(row["camera_local_pair"] in (
        {"CAM-02": "1", "CAM-05": "10"},
        {"CAM-02": "2", "CAM-05": "20"},
    ) for row in result["matches"])
    assert result["rule_contract"]["production_identity_write"] is False
    assert result["rule_contract"]["identity_probability_calibrated"] is False
    assert result["rule_contract"]["saved_human_review_decisions_read"] is False


def test_opt_in_matcher_trace_records_provenance_gates_and_competitors_without_changing_decisions():
    vectors, records, tracks = _records()
    observed = 4_000_000_000
    for row in records:
        row["observed_monotonic_ns"] = observed + int(row["frame"]) * 1_000_000
    normal = score_and_assign_provisional_ids(
        vectors, records, tracks, minimum_separated_control_frames=2)
    diagnostic = score_and_assign_provisional_ids(
        vectors, records, tracks, minimum_separated_control_frames=2,
        diagnostic_logging=True, diagnostic_now_ns=observed + 10_000_000,
    )

    assert diagnostic["matches"] == normal["matches"]
    assert "decision_trace" not in normal
    rows = {tuple(sorted(row["camera_local_pair"].items())): row
            for row in diagnostic["decision_trace"]}
    accepted = rows[(('CAM-02', '1'), ('CAM-05', '10'))]
    assert accepted["decision"] == "ACCEPTED_BY_MATCHER"
    assert accepted["candidate_global_id"] == "G001"
    assert accepted["source_observations"]["CAM-02"]["source_id"] == 0
    assert accepted["source_observations"]["CAM-02"]["latest_frame"] == 2
    assert accepted["source_observations"]["CAM-02"]["latest_source_pts_ns"] == 100_000_000
    assert accepted["source_observations"]["CAM-02"]["gallery_sample_count"] == 2
    assert accepted["source_observations"]["CAM-02"]["freshest_sample_age_ms_monotonic"] == 8.0
    assert accepted["reciprocal_match"] == {"CAM-02": True, "CAM-05": True}
    assert accepted["competing_track_pair_by_camera"]["CAM-02"] == {
        "CAM-02": "1", "CAM-05": "20",
    }

    rejected = rows[(('CAM-02', '1'), ('CAM-05', '20'))]
    assert rejected["decision"] == "REJECTED_OR_AMBIGUOUS"
    assert rejected["failed_gates"]
    assert rejected["reason"].startswith("failed existing gate(s): ")
    assert rejected["competing_similarity_score_by_camera"]["CAM-02"] is not None


def test_ambiguous_competition_abstains_instead_of_forcing_a_match():
    vectors = np.zeros((6, 512), dtype=np.float32)
    vectors[:, 0] = 1.0
    records = []
    for camera, local_id, start, source_id in (
        ("CAM-02", "1", 0, 0), ("CAM-05", "10", 2, 1), ("CAM-05", "20", 4, 1),
    ):
        for offset in range(2):
            index = start + offset
            records.append({"camera_id": camera, "local_track_id": local_id,
                            "source_id": source_id, "frame": 1 + offset,
                            "source_pts_ns": (1 + offset) * 50_000_000,
                            "sample_id": f"S{index}", "bbox_inference_xywh": [0, 0, 10, 20]})
    tracks = {
        "CAM-02": {frame: [{"object_id": "1", "box": [0, 0, 10, 20]}]
                   for frame in (1, 2)},
        "CAM-05": {frame: [{"object_id": "10", "box": [0, 0, 10, 20]},
                           {"object_id": "20", "box": [100, 0, 10, 20]}]
                   for frame in (1, 2)},
    }
    result = score_and_assign_provisional_ids(
        vectors, records, tracks, minimum_separated_control_frames=2)

    assert result["matches"] == []
    assert all(not row["automatic_candidate"] for row in result["candidate_pairs"])


def test_no_simultaneous_track_controls_fails_closed():
    vectors = np.zeros((8, 512), dtype=np.float32)
    vectors[:, 0] = 1.0
    records = []
    for camera, local_id, source_id, start in (
        ("CAM-02", "1", 0, 1), ("CAM-02", "2", 0, 3),
        ("CAM-05", "10", 1, 1), ("CAM-05", "20", 1, 3),
    ):
        for offset in range(2):
            frame = start + offset
            records.append({"camera_id": camera, "local_track_id": local_id,
                            "source_id": source_id, "frame": frame,
                            "source_pts_ns": frame * 50_000_000,
                            "sample_id": f"{camera}-{local_id}-{offset}"})
    # No IDs coexist in the same source frames, so there is no empirical
    # same-camera negative-like control to support automatic assignment.
    tracks = {
        "CAM-02": {1: [{"object_id": "1", "box": [0, 0, 10, 20]}],
                   2: [{"object_id": "1", "box": [0, 0, 10, 20]}],
                   3: [{"object_id": "2", "box": [100, 0, 10, 20]}],
                   4: [{"object_id": "2", "box": [100, 0, 10, 20]}]},
        "CAM-05": {1: [{"object_id": "10", "box": [0, 0, 10, 20]}],
                   2: [{"object_id": "10", "box": [0, 0, 10, 20]}],
                   3: [{"object_id": "20", "box": [100, 0, 10, 20]}],
                   4: [{"object_id": "20", "box": [100, 0, 10, 20]}]},
    }

    result = score_and_assign_provisional_ids(
        vectors, records, tracks, minimum_separated_control_frames=2)

    assert result["status"] == "ABSTAIN_NO_AUTOMATIC_CONTROLS"
    assert result["matches"] == []
    assert result["one_to_many_conflicts_rejected"] == []


def test_new_local_track_is_scored_against_retained_cross_camera_gallery():
    """Synthetic feature geometry checks scoring only, not real identity accuracy."""
    vectors = np.zeros((10, 512), dtype=np.float32)
    # CAM-02 old track is unrelated in this constructed fixture; new track 7
    # has the same deterministic feature as the retained CAM-05 track 17.
    vectors[0:2, 1] = 1.0  # old CAM-02 local 5
    vectors[2:4, 0] = 1.0  # returned/new CAM-02 local 7
    vectors[4:6, 2] = 1.0  # CAM-02 control local 9
    vectors[6:8, 0] = 1.0  # retained CAM-05 local 17
    vectors[8:10, 3] = 1.0  # CAM-05 control local 22

    records = []
    track_samples = (
        ("CAM-02", "5", 0, (1, 2)),
        ("CAM-02", "7", 0, (100, 101)),
        ("CAM-02", "9", 0, (100, 101)),
        ("CAM-05", "17", 1, (10, 11)),
        ("CAM-05", "22", 1, (10, 11)),
    )
    tracks = {"CAM-02": {}, "CAM-05": {}}
    for camera, local_id, source_id, frames in track_samples:
        for frame in frames:
            records.append({
                "camera_id": camera, "local_track_id": local_id,
                "source_id": source_id, "frame": frame,
                "source_pts_ns": frame * 50_000_000,
                "sample_id": f"synthetic-{camera}-{local_id}-{frame}",
            })
            tracks[camera].setdefault(frame, []).append({
                "object_id": local_id,
                "box": ([0, 0, 20, 40] if local_id != "9" and local_id != "22"
                        else [100, 0, 20, 40]),
            })

    result = score_and_assign_provisional_ids(
        vectors, records, tracks,
        minimum_separated_control_frames=2,
        control_reference_max=0.64508455991745,
        control_reference_source="test-only saved unlabeled control envelope",
    )

    assert len(result["matches"]) == 1
    assert result["matches"][0]["camera_local_pair"] == {
        "CAM-02": "7", "CAM-05": "17",
    }
    assert result["matches"][0]["automatic_candidate"] is True
    assert result["rule_contract"]["production_identity_write"] is False
    # This proves that a new camera-local ID can be compared to retained
    # gallery evidence under the existing gates; the fixture embeds known
    # vectors and makes no assertion about a real person.


def test_overlay_keeps_local_and_provisional_ids_visibly_distinct(monkeypatch):
    texts = []
    original = __import__("cv2").putText

    def capture_text(image, text, *args, **kwargs):
        texts.append(text)
        return original(image, text, *args, **kwargs)

    monkeypatch.setattr("services.mv3dt_staging.native_map_preview.cv2.putText", capture_text)
    _camera_panel(
        np.zeros((1080, 1920, 3), dtype=np.uint8), "CAM-02",
        [{"object_id": "5", "box": [10, 20, 40, 80]}],
        provisional_global_ids={"5": {"provisional_global_id": "G001"}},
    )
    assert any("CAM-02 local 5" in text and "G001?" in text for text in texts)


def test_provisional_assignment_can_be_revoked_with_append_only_audit(tmp_path):
    (tmp_path / "automatic_global_id_assignments.json").write_text(json.dumps({
        "matches": [{"provisional_global_id": "G001",
                     "camera_local_pair": {"CAM-02": "5", "CAM-05": "17"}}]
    }))
    (tmp_path / "automatic_global_id_diagnostics.json").write_text(json.dumps({"candidate_pairs": []}))
    history = tmp_path / "assignment_revision_history.jsonl"
    history.write_text("")

    event = append_provisional_revision(tmp_path, "G001", action="revoke",
                                        rationale="later evidence contradicted the appearance association")

    assert event["current_pair"] is None
    assert resolve_provisional_assignments(tmp_path) == {"G001": None}
    saved = json.loads(history.read_text().strip())
    assert saved["previous_pair"] == {"CAM-02": "5", "CAM-05": "17"}
    assert saved["production_identity_write"] is False
    assert saved["evidence_status"].startswith("provisional_only")


def test_revision_replacement_must_pass_automatic_candidate_gates(tmp_path):
    (tmp_path / "automatic_global_id_assignments.json").write_text(json.dumps({
        "matches": [{"provisional_global_id": "G001",
                     "camera_local_pair": {"CAM-02": "5", "CAM-05": "17"}}]
    }))
    (tmp_path / "automatic_global_id_diagnostics.json").write_text(json.dumps({
        "candidate_pairs": [{"automatic_candidate": False,
                              "camera_local_pair": {"CAM-02": "4", "CAM-05": "16"}}]
    }))
    (tmp_path / "assignment_revision_history.jsonl").write_text("")

    with pytest.raises(ValueError, match="did not pass"):
        append_provisional_revision(
            tmp_path, "G001", action="replace", rationale="alternative candidate",
            replacement_pair={"CAM-02": "4", "CAM-05": "16"},
        )
