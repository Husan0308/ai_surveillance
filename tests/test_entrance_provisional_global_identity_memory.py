from __future__ import annotations

import numpy as np
import pytest

from services.mv3dt_staging.live_tracking.live_bridge import (
    camera_track_label,
    provisional_identity_summary,
    provisional_camera_overlay,
    visible_provisional_identity_groups,
)
from services.mv3dt_staging.provisional_global_identity_memory import (
    ProvisionalGlobalIdentityMemory,
)


def _vector(index: int) -> np.ndarray:
    value = np.zeros(512, dtype=np.float32)
    value[index] = 1.0
    return value


def _gallery(camera: str, local_id: str, feature: int, frames=(1, 2), *, offset=0):
    source_id = 0 if camera == "CAM-02" else 1
    return [{
        "vector": _vector(feature),
        "record": {
            "camera_id": camera, "source_id": source_id,
            "local_track_id": local_id, "frame": frame + offset,
            "source_pts_ns": (frame + offset) * 50_000_000,
            "bbox_inference_xywh": [100.0, 100.0, 60.0, 200.0],
        },
    } for frame in frames]


def _match(left: str, right: str, score=0.92):
    return {
        "camera_local_pair": {"CAM-02": left, "CAM-05": right},
        "cosine_similarity_median": score,
        "cosine_similarity_min": score - 0.02,
        "samples": {"CAM-02": 2, "CAM-05": 2},
    }


def _tracks(*camera_ids):
    result = {"CAM-02": {}, "CAM-05": {}}
    for camera, local_id, frames in camera_ids:
        for frame in frames:
            result[camera].setdefault(frame, []).append({
                "object_id": local_id, "box": [100.0, 100.0, 60.0, 200.0],
            })
    return result


def _seed_one(memory, now=0):
    galleries = {
        ("CAM-02", "5"): _gallery("CAM-02", "5", 0),
        ("CAM-05", "17"): _gallery("CAM-05", "17", 0),
    }
    row = _match("5", "17")
    memory.touch_observations("CAM-02", ["5"], now)
    memory.touch_observations("CAM-05", ["17"], now)
    state = memory.reconcile([row], [row], galleries, _tracks(),
                             control_reference_max=0.645,
                             now_ns=now, expected_revision=memory.revision)
    assert state["accepted"]
    assert memory.assignment_map()["CAM-02"]["5"]["provisional_global_id"] == "G001"
    return galleries


def test_one_camera_local_id_change_inherits_existing_gid_only_when_not_simultaneous():
    memory = ProvisionalGlobalIdentityMemory()
    old = _seed_one(memory)
    galleries = dict(old)
    galleries[("CAM-02", "7")] = _gallery("CAM-02", "7", 0, (10, 11))
    row = _match("7", "17")
    tracks = _tracks(("CAM-02", "5", (1, 2)), ("CAM-02", "7", (10, 11)),
                     ("CAM-05", "17", (1, 2)))
    memory.touch_observations("CAM-02", ["7"], 2_000_000_000)
    memory.touch_observations("CAM-05", ["17"], 2_000_000_000)
    result = memory.reconcile([row], [row], galleries, tracks,
                              control_reference_max=0.645,
                              now_ns=2_000_000_000,
                              expected_revision=memory.revision)
    assignments = memory.assignment_map()
    assert result["accepted"]
    assert assignments["CAM-02"]["7"]["provisional_global_id"] == "G001"
    assert assignments["CAM-05"]["17"]["provisional_global_id"] == "G001"
    assert "5" not in assignments["CAM-02"]
    snapshot = memory.snapshot(2_000_000_000)
    gid = snapshot["identities"][0]
    assert set(gid["camera_local_track_histories"]["CAM-02"]) == {"5", "7"}
    assert any(event["event"] == "LOCAL_TRACK_BINDING_WITHDRAWN" for event in snapshot["events"])
    source_record = gid["embedding_provenance_by_camera"]["CAM-02"][-1]
    assert source_record["source_id"] == 0
    assert source_record["frame"] == 11
    assert source_record["source_pts_ns"] == 550_000_000

    simultaneous = ProvisionalGlobalIdentityMemory()
    _seed_one(simultaneous)
    simultaneous.touch_observations("CAM-02", ["5", "7"], 2_000_000_000)
    simultaneous.touch_observations("CAM-05", ["17"], 2_000_000_000)
    conflict_tracks = _tracks(("CAM-02", "5", (10,)), ("CAM-02", "7", (10, 11)),
                              ("CAM-05", "17", (10,)))
    result = simultaneous.reconcile(
        [row], [row], galleries, conflict_tracks, control_reference_max=0.645,
        now_ns=2_000_000_000, expected_revision=simultaneous.revision,
    )
    assert result["accepted"] == []
    assert simultaneous.snapshot(2_000_000_000)["counters"]["simultaneous_track_conflicts"] == 1


def test_both_camera_local_ids_change_with_reciprocal_margin_and_competitors():
    memory = ProvisionalGlobalIdentityMemory()
    old_galleries = {
        ("CAM-02", "5"): _gallery("CAM-02", "5", 0),
        ("CAM-05", "17"): _gallery("CAM-05", "17", 0),
        ("CAM-02", "6"): _gallery("CAM-02", "6", 1),
        ("CAM-05", "18"): _gallery("CAM-05", "18", 1),
    }
    old_rows = [_match("5", "17"), _match("6", "18", 0.90)]
    old_tracks = _tracks(("CAM-02", "5", (1, 2)), ("CAM-05", "17", (1, 2)),
                         ("CAM-02", "6", (1, 2)), ("CAM-05", "18", (1, 2)))
    memory.touch_observations("CAM-02", ["5", "6"], 0)
    memory.touch_observations("CAM-05", ["17", "18"], 0)
    seeded = memory.reconcile(old_rows, old_rows, old_galleries, old_tracks,
                              control_reference_max=0.645, now_ns=0,
                              expected_revision=memory.revision)
    assert {row["provisional_global_id"] for row in seeded["accepted"]} == {"G001", "G002"}

    memory.touch_observations("CAM-02", ["7", "8"], 1_000_000_000)
    memory.touch_observations("CAM-05", ["19", "20"], 1_000_000_000)

    new_galleries = dict(old_galleries)
    new_galleries.update({
        ("CAM-02", "7"): _gallery("CAM-02", "7", 0, (10, 11)),
        ("CAM-05", "19"): _gallery("CAM-05", "19", 0, (10, 11)),
        ("CAM-02", "8"): _gallery("CAM-02", "8", 1, (10, 11)),
        ("CAM-05", "20"): _gallery("CAM-05", "20", 1, (10, 11)),
    })
    changed_rows = [_match("7", "19"), _match("8", "20", 0.90)]
    competitor_rows = changed_rows + [
        _match("7", "20", 0.68), _match("8", "19", 0.67),
    ]
    changed_tracks = _tracks(("CAM-02", "7", (10, 11)), ("CAM-05", "19", (10, 11)),
                             ("CAM-02", "8", (10, 11)), ("CAM-05", "20", (10, 11)))
    updated = memory.reconcile(changed_rows, competitor_rows, new_galleries, changed_tracks,
                               control_reference_max=0.645, now_ns=1_000_000_000,
                               expected_revision=memory.revision)
    assignments = memory.assignment_map()
    assert assignments["CAM-02"]["7"]["provisional_global_id"] == "G001"
    assert assignments["CAM-05"]["19"]["provisional_global_id"] == "G001"
    assert assignments["CAM-02"]["8"]["provisional_global_id"] == "G002"
    assert assignments["CAM-05"]["20"]["provisional_global_id"] == "G002"
    assert memory.snapshot(1_000_000_000)["counters"]["both_camera_id_continuations"] == 2
    assert all(row["provisional_global_id"] in {"G001", "G002"} for row in updated["accepted"])


@pytest.mark.parametrize("gap_seconds", [30, 120, 300, 540])
def test_changed_track_id_return_before_ttl_can_reuse_gid(gap_seconds):
    memory = ProvisionalGlobalIdentityMemory()
    old = _seed_one(memory)
    now = gap_seconds * 1_000_000_000
    new = dict(old)
    new[("CAM-02", "7")] = _gallery("CAM-02", "7", 0, (10, 11), offset=gap_seconds * 20)
    row = _match("7", "17")
    tracks = _tracks(("CAM-02", "5", (1, 2)),
                     ("CAM-02", "7", (gap_seconds * 20 + 10, gap_seconds * 20 + 11)),
                     ("CAM-05", "17", (1, 2)))
    memory.touch_observations("CAM-02", ["7"], now)
    memory.touch_observations("CAM-05", ["17"], now)
    memory.reconcile([row], [row], new, tracks, control_reference_max=0.645,
                     now_ns=now, expected_revision=memory.revision)
    assert memory.assignment_map()["CAM-02"]["7"]["provisional_global_id"] == "G001"
    assert memory.snapshot(now)["identities"][0]["state"] == "ACTIVE"
    lifecycle = [event for event in memory.snapshot(now)["events"]
                 if event.get("provisional_global_id") == "G001"
                 and event.get("event") == "IDENTITY_STATE_CHANGED"]
    assert [(event["from_state"], event["to_state"]) for event in lifecycle] == [
        ("ACTIVE", "LOST"), ("LOST", "ACTIVE")
    ]


def test_expired_gid_is_not_reused_and_candidate_withdrawal_clears_labels():
    memory = ProvisionalGlobalIdentityMemory()
    gallery = _seed_one(memory)
    assert memory.assignment_map()["CAM-05"]["17"]
    assert provisional_identity_summary(memory.assignment_map())
    withdrawn = memory.reconcile([], [], gallery, _tracks(), control_reference_max=0.645,
                                 now_ns=1_000_000_000,
                                 expected_revision=memory.revision)
    assert withdrawn["accepted"] == []
    assert memory.assignment_map() == {"CAM-02": {}, "CAM-05": {}}
    assert provisional_identity_summary(memory.assignment_map()) == []

    now = 601_000_000_000
    assert memory.expire(now) == ["G001"]
    assert memory.snapshot(now)["expired_id_tombstones"][0]["state"] == "EXPIRED"
    assert memory.snapshot(now)["expired_id_tombstones"][0]["embeddings_retained"] is False
    new_row = _match("30", "40")
    new_gallery = {
        ("CAM-02", "30"): _gallery("CAM-02", "30", 0),
        ("CAM-05", "40"): _gallery("CAM-05", "40", 0),
    }
    memory.touch_observations("CAM-02", ["30"], now)
    memory.touch_observations("CAM-05", ["40"], now)
    memory.reconcile([new_row], [new_row], new_gallery, _tracks(),
                     control_reference_max=0.645, now_ns=now,
                     expected_revision=memory.revision)
    assert memory.assignment_map()["CAM-02"]["30"]["provisional_global_id"] == "G002"
    assert memory.snapshot(now)["next_id_is_monotonic_and_not_reused"] is True


def test_identical_appearance_galleries_abstain_and_stale_revision_cannot_overwrite():
    memory = ProvisionalGlobalIdentityMemory()
    old = {
        ("CAM-02", "5"): _gallery("CAM-02", "5", 0),
        ("CAM-05", "17"): _gallery("CAM-05", "17", 0),
        ("CAM-02", "6"): _gallery("CAM-02", "6", 0),
        ("CAM-05", "18"): _gallery("CAM-05", "18", 0),
    }
    rows = [_match("5", "17"), _match("6", "18", 0.90)]
    memory.touch_observations("CAM-02", ["5", "6"], 0)
    memory.touch_observations("CAM-05", ["17", "18"], 0)
    memory.reconcile(rows, rows, old,
                     _tracks(("CAM-02", "5", (1,)), ("CAM-02", "6", (1,)),
                             ("CAM-05", "17", (1,)), ("CAM-05", "18", (1,))),
                     control_reference_max=0.645, now_ns=0,
                     expected_revision=memory.revision)
    new = dict(old)
    new.update({
        ("CAM-02", "7"): _gallery("CAM-02", "7", 0, (10, 11)),
        ("CAM-05", "19"): _gallery("CAM-05", "19", 0, (10, 11)),
        ("CAM-02", "8"): _gallery("CAM-02", "8", 0, (10, 11)),
        ("CAM-05", "20"): _gallery("CAM-05", "20", 0, (10, 11)),
    })
    candidates = [_match("7", "19"), _match("8", "20"),
                  _match("7", "20"), _match("8", "19")]
    matches = [_match("7", "19"), _match("8", "20")]
    memory.touch_observations("CAM-02", ["7", "8"], 1_000_000_000)
    memory.touch_observations("CAM-05", ["19", "20"], 1_000_000_000)
    result = memory.reconcile(matches, candidates, new,
                              _tracks(("CAM-02", "7", (10,)), ("CAM-02", "8", (10,)),
                                      ("CAM-05", "19", (10,)), ("CAM-05", "20", (10,))),
                              control_reference_max=0.645, now_ns=1_000_000_000,
                              expected_revision=memory.revision)
    assert result["accepted"] == []
    assert result["abstained"]

    stale_revision = memory.revision - 1
    stale = memory.reconcile([], [], new, _tracks(), control_reference_max=0.645,
                              now_ns=2_000_000_000, expected_revision=stale_revision)
    assert stale["stale"] is True
    assert memory.snapshot(2_000_000_000)["counters"]["stale_decisions_rejected"] == 1


def test_video_and_map_identity_summary_keeps_local_namespaces_separate():
    mapping = {
        "CAM-02": {"7": {"provisional_global_id": "G001"}},
        "CAM-05": {"19": {"provisional_global_id": "G001"}},
    }
    assert provisional_identity_summary(mapping) == [
        "G001? · CAM-02 local 7 / CAM-05 local 19 · map points remain separate"
    ]
    label = camera_track_label(7, 0.91, "G001")
    assert label == "local 7 · G001? · det 0.91"
    assert "local 7" in label and "G001?" in label
    assert provisional_camera_overlay(mapping, "CAM-02", {"7"}) == "PROVISIONAL G001? · local 7"
    assert provisional_camera_overlay(mapping, "CAM-05", {"20"}) is None


def test_map_global_label_requires_exact_per_camera_frame_and_updates_on_withdrawal():
    mapping = {
        "CAM-02": {"7": {"provisional_global_id": "G001"}},
        "CAM-05": {"19": {"provisional_global_id": "G001"}},
    }
    frame_keys = {"CAM-02": (0, 100, 5_000), "CAM-05": (1, 200, 7_000)}
    map_states = {
        "CAM-02": {"status": "LIVE", "source_id": 0, "frame_index": 100,
                   "source_pts_ns": 5_000, "markers": [{"local_tracker_id": "7"}]},
        "CAM-05": {"status": "LIVE", "source_id": 1, "frame_index": 200,
                   "source_pts_ns": 7_000, "markers": [{"local_tracker_id": "19"}]},
    }
    visible = visible_provisional_identity_groups(
        mapping, {"CAM-02": {"7"}, "CAM-05": {"19"}}, frame_keys, map_states,
        mapping, mapping,
    )
    assert visible == [{
        "provisional_global_id": "G001",
        "camera_local_ids": {"CAM-02": "7", "CAM-05": "19"},
        "per_camera_exact_frame_keys": {
            "CAM-02": {"source_id": 0, "frame_index": 100, "source_pts_ns": 5_000},
            "CAM-05": {"source_id": 1, "frame_index": 200, "source_pts_ns": 7_000},
        },
        "map_points_fused": False,
        "identity_accuracy": "UNVERIFIED",
    }]
    mismatched = dict(map_states)
    mismatched["CAM-05"] = {**map_states["CAM-05"], "source_pts_ns": 7_001}
    assert not visible_provisional_identity_groups(
        mapping, {"CAM-02": {"7"}, "CAM-05": {"19"}}, frame_keys, mismatched,
        mapping, mapping,
    )
    stale_video_mapping = {"CAM-02": mapping["CAM-02"], "CAM-05": {}}
    assert not visible_provisional_identity_groups(
        mapping, {"CAM-02": {"7"}, "CAM-05": {"19"}}, frame_keys, map_states,
        stale_video_mapping, mapping,
    )
    assert not visible_provisional_identity_groups(
        {"CAM-02": {}, "CAM-05": {}},
        {"CAM-02": {"7"}, "CAM-05": {"19"}}, frame_keys, map_states,
    )


def test_historical_match_rescoring_does_not_refresh_lost_identity():
    memory = ProvisionalGlobalIdentityMemory()
    gallery = _seed_one(memory)
    row = _match("5", "17")
    result = memory.reconcile(
        [row], [row], gallery, _tracks(), control_reference_max=0.645,
        now_ns=10_000_000_000, expected_revision=memory.revision,
    )
    assert result["accepted"]
    snapshot = memory.snapshot(10_000_000_000)
    identity = snapshot["identities"][0]
    assert identity["state"] == "LOST"
    assert identity["last_seen_ns"] == 0
