import numpy as np

from services.mv3dt_room.global_identity_manager import GlobalIdentityManager
from services.mv3dt_room.identity_metrics import IdentityMetrics
from services.mv3dt_room.identity_queue import LatestObservationQueue, ObservationEnvelope


def obs(camera="CAM-01", native=1, frame=1, world=(1.0, 1.0)):
    return {
        "camera_id": camera,
        "native_track_id": native,
        "frame": frame,
        "bbox": (10.0, 10.0, 100.0, 200.0),
        "world": world,
    }



def test_borderline_wide_person_crop_is_matching_only_not_gallery_quality(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "services/mv3dt_room"))
    from services.mv3dt_room.extract_osnet import crop_quality
    from services.mv3dt_room.live_identity_worker import matching_only_crop_quality

    # Captured CAM-01 native fragment: DeepStream encoded it successfully,
    # but aspect 0.956 just exceeded the gallery-quality 0.95 bound.
    box = tuple(value * 0.75 for value in (1492.848, 506.733, 1661.627, 683.296))
    gallery_usable, quality = crop_quality(box)
    assert not gallery_usable
    assert 0.95 < quality["aspect"] < 1.0
    assert matching_only_crop_quality(quality)
    assert not matching_only_crop_quality({**quality, "border_truncated": True})
    assert not matching_only_crop_quality({**quality, "inside_fraction": 0.8})
    assert not matching_only_crop_quality({**quality, "aspect": 1.2})
    bottom_box = (789.676, 716.148, 1037.684, 1078.775)
    bottom_gallery_usable, bottom_quality = crop_quality(bottom_box)
    assert not bottom_gallery_usable
    assert bottom_quality["border_truncated"]
    assert matching_only_crop_quality({**bottom_quality, "quality_bbox": bottom_box})
    top_box = (789.676, 0.0, 1037.684, 362.0)
    _, top_quality = crop_quality(top_box)
    assert not matching_only_crop_quality({**top_quality, "quality_bbox": top_box})

def test_latest_queue_coalesces_routine_updates_but_keeps_critical_marker():
    metrics = IdentityMetrics()
    queue = LatestObservationQueue(max_tracks=4, metrics=metrics)
    queue.submit(ObservationEnvelope(obs(frame=1), generation=1, critical=True))
    queue.submit(ObservationEnvelope(obs(frame=2), generation=1, critical=False))
    result = queue.get_batch(8)
    assert [item.observation["frame"] for item in result] == [2]
    assert metrics.snapshot()["counters"]["coalesced_stale_updates"] == 1


def test_sticky_track_cannot_switch_canonical_identity_on_later_embedding():
    manager = GlobalIdentityManager(np.empty((0, 2), np.float32))
    first, _, _ = manager.process(obs(frame=1), np.asarray([1.0, 0.0], np.float32), [])
    second, evidence, _ = manager.process(
        obs(frame=2),
        np.asarray([0.0, 1.0], np.float32),
        [],
        allow_reassignment=False,
    )
    assert second == first
    assert evidence["reason"] == "sticky_native_track"
    assert not [event for event in manager.events if event["event"] == "MERGE"]


def test_joint_assignment_rejects_same_camera_duplicate_identity():
    manager = GlobalIdentityManager(np.empty((0, 2), np.float32))
    first, _, _ = manager.process(obs(native=1, frame=1), np.asarray([1.0, 0.0], np.float32), [])
    active = [{**obs(native=1, frame=2), "global_person_id_num": first}]
    first_duplicate = obs(native=2, frame=2, world=(1.1, 1.0))
    first_duplicate["bbox"] = (220.0, 10.0, 320.0, 200.0)
    second_duplicate = obs(native=3, frame=2, world=(1.2, 1.0))
    second_duplicate["bbox"] = (420.0, 10.0, 520.0, 200.0)
    resolved = manager.process_batch(
        [
            {"obs": first_duplicate, "vector": np.asarray([1.0, 0.0], np.float32)},
            {"obs": second_duplicate, "vector": np.asarray([1.0, 0.0], np.float32)},
        ],
        active,
    )
    assert len(resolved) == 2
    assert resolved[0][0] is None
    assert resolved[1][0] is None
    assert manager.next_number == 2
    assert manager.conflict_guard_events >= 1


def test_recent_same_camera_target_blocks_old_historical_reid_match():
    manager = GlobalIdentityManager(np.empty((0, 2), np.float32))
    vector = np.asarray([1.0, 0.0], np.float32)
    old_position = {
        **obs(native=1, frame=90, world=(15.0, -25.0)),
        "timestamp": "2026-09-23T11:45:50.000Z",
    }
    gid = manager.confirm_new(old_position, vector, "test_initial")
    recent_distinct_target = {
        **obs(native=43, frame=100, world=(20.0, -15.0)),
        "timestamp": "2026-09-23T11:45:50.500Z",
    }
    manager.accept_existing(gid, recent_distinct_target, vector, {"reason": "test_recent_track"})

    # This candidate matches the old same-camera sample, but a different
    # native target was still visible two frames ago at a spatially distinct
    # position. Historical ReID must not override that concurrent evidence.
    candidate = {
        **obs(native=52, frame=102, world=(15.2, -25.1)),
        "timestamp": "2026-09-23T11:45:50.600Z",
        "bbox": (250.0, 10.0, 350.0, 200.0),
    }
    resolved, evidence = manager.resolve_existing(candidate, vector, [])
    assert resolved is None
    assert evidence["reason"] == "pending_no_acceptable_candidate"
    trace = manager.candidate_trace(candidate, vector, [])
    assert trace[0]["exact_rejection_reason"] == "same_camera_simultaneous"


def test_recent_camera_fragment_can_reassociate_with_contemporaneous_peer_evidence():
    manager = GlobalIdentityManager(np.empty((0, 2), np.float32))
    gallery = np.asarray([1.0, 0.0], np.float32)
    candidate_vector = np.asarray([0.634, (1.0 - 0.634**2) ** 0.5], np.float32)
    cam04_prior = {
        **obs(camera="CAM-04", native=6, frame=245, world=(24.36, -18.24)),
        "timestamp": "2026-09-23T11:45:50.000Z",
        "bbox": (790.0, 18.0, 973.0, 415.0),
    }
    gid = manager.confirm_new(cam04_prior, gallery, "test_initial")
    cam01_peer = {
        **obs(camera="CAM-01", native=7, frame=256, world=(25.44, -13.01)),
        "timestamp": "2026-09-23T11:45:50.443Z",
    }
    manager.accept_existing(gid, cam01_peer, gallery, {"reason": "test_cross_view"})
    cam04_fragment = {
        **obs(camera="CAM-04", native=9, frame=257, world=(21.11, -17.65)),
        "timestamp": "2026-09-23T11:45:50.443Z",
        "bbox": (685.0, 46.0, 870.0, 461.0),
    }
    resolved, evidence = manager.resolve_existing(cam04_fragment, candidate_vector, [])
    assert resolved == gid
    assert evidence["same_camera_fragment"] is True
    assert evidence["reason"] in {"same_camera_fragment_handoff", "cross_camera_reid_geometry"}


def test_recent_same_camera_fragment_survives_remote_projection_conflict():
    manager = GlobalIdentityManager(np.empty((0, 2), np.float32))
    vector = np.asarray([1.0, 0.0], np.float32)
    predecessor = {
        **obs(camera="CAM-04", native=14, frame=2718, world=(13.02, -17.48)),
        "timestamp": "2026-09-23T13:36:20.263Z",
        "bbox": (586.8, 423.9, 1121.1, 1032.4),
    }
    gid = manager.confirm_new(predecessor, vector, "test_initial")
    remote_projection = {
        **obs(camera="CAM-01", native=15, frame=2722, world=(18.86, -11.14)),
        "timestamp": "2026-09-23T13:36:20.513Z",
    }
    manager.accept_existing(gid, remote_projection, vector, {"reason": "test_live_pair"})
    fragment = {
        **obs(camera="CAM-04", native=16, frame=2724, world=(12.48, -17.04)),
        "timestamp": "2026-09-23T13:36:20.562Z",
        "bbox": (585.5, 394.9, 1000.8, 1037.3),
    }
    active_peer = [{**remote_projection, "global_person_id_num": gid}]

    resolved, evidence = manager.resolve_existing(fragment, vector, active_peer)

    assert resolved == gid
    assert evidence["reason"] == "same_camera_direct_continuity"
    assert manager.next_number == 2


def test_cross_camera_pending_observation_reuses_existing_identity_by_time_and_geometry():
    manager = GlobalIdentityManager(np.empty((0, 2), np.float32))
    first, _, _ = manager.process(obs(camera="CAM-01", native=1, frame=100, world=(21.0, -29.0)), np.asarray([1.0, 0.0], np.float32), [])
    candidate = obs(camera="CAM-04", native=8, frame=100, world=(21.4, -29.2))
    candidate["crop_quality_usable"] = False
    active = [{**obs(camera="CAM-01", native=1, frame=100, world=(21.0, -29.0)), "global_person_id_num": first}]
    resolved, evidence = manager.resolve_existing(candidate, None, active)
    assert resolved == first
    assert evidence["reason"] == "cross_camera_geometry_time"
    assert manager.next_number == 2


def test_cropless_wide_cross_camera_candidate_stays_pending_but_tight_match_resolves():
    manager = GlobalIdentityManager(np.empty((0, 2), np.float32))
    vector = np.asarray([1.0, 0.0], np.float32)
    peer = {
        **obs(camera="CAM-04", native=4, frame=78, world=(19.76, -29.75)),
        "timestamp": "2026-09-23T11:45:50.806Z",
    }
    person_a = manager.confirm_new(peer, vector, "test_initial")
    active_peer = [{**peer, "global_person_id_num": person_a}]

    # Reproduces the live false merge: 8.01 m apart with only 38 ms
    # difference, and no embedding to distinguish the person.
    ambiguous = {
        **obs(camera="CAM-01", native=5, frame=78, world=(11.83, -30.86)),
        "timestamp": "2026-09-23T11:45:50.844Z",
    }
    resolved, evidence = manager.resolve_existing(ambiguous, None, active_peer)
    assert resolved is None
    assert evidence["reason"] == "pending_no_acceptable_candidate"
    trace = manager.candidate_trace(ambiguous, None, active_peer)
    assert trace[0]["exact_rejection_reason"] == "geometry_only_requires_reid_or_native_evidence"

    # The actual live counterpart is only 1.38 m from the peer and remains
    # eligible for tight geometry-only association while the wider target is
    # pending rather than poisoning the same-camera conflict guard.
    actual = {
        **obs(camera="CAM-01", native=6, frame=100, world=(20.47, -28.51)),
        "timestamp": "2026-09-23T11:45:51.906Z",
    }
    active_peer = [{
        **obs(camera="CAM-04", native=4, frame=100, world=(19.79, -29.72)),
        "timestamp": "2026-09-23T11:45:51.906Z",
        "global_person_id_num": person_a,
    }]
    resolved, evidence = manager.resolve_existing(actual, None, active_peer)
    assert resolved == person_a
    assert evidence["reason"] == "cross_camera_geometry_time"
    manager.accept_existing(resolved, actual, None, evidence)

    # A separate same-camera target cannot then be assigned to that identity.
    active_same_camera = [{**actual, "global_person_id_num": person_a}]
    other_person = {
        **obs(camera="CAM-01", native=5, frame=101, world=(12.85, -29.84)),
        "timestamp": "2026-09-23T11:45:51.956Z",
    }
    resolved, evidence = manager.resolve_existing(other_person, None, active_same_camera)
    assert resolved is None
    assert evidence["reason"] == "pending_no_acceptable_candidate"
    assert manager.next_number == 2


def test_rejected_wide_geometry_alternative_does_not_overwrite_accepted_same_camera_reason():
    manager = GlobalIdentityManager(np.empty((0, 2), np.float32))
    vector = np.asarray([1.0, 0.0], np.float32)
    first = {
        **obs(camera="CAM-04", native=4, frame=1, world=(0.0, 0.0)),
        "timestamp": "2026-09-23T11:45:50.000Z",
    }
    gid = manager.confirm_new(first, vector, "test_initial")
    prior_cam01 = {
        **obs(camera="CAM-01", native=1, frame=1, world=(0.2, 0.0)),
        "timestamp": "2026-09-23T11:45:50.000Z",
    }
    manager.accept_existing(gid, prior_cam01, vector, {"reason": "test_cross_view"})

    continuing = {
        **obs(camera="CAM-01", native=1, frame=2, world=(0.4, 0.0)),
        "timestamp": "2026-09-23T11:45:50.050Z",
    }
    active_peer = [{
        **obs(camera="CAM-04", native=4, frame=2, world=(7.8, 0.0)),
        "timestamp": "2026-09-23T11:45:50.050Z",
        "global_person_id_num": gid,
    }]
    resolved, evidence = manager.resolve_existing(continuing, None, active_peer)
    assert resolved == gid
    assert evidence["reason"] == "same_camera_direct_continuity"


def test_stale_spatially_impossible_gallery_does_not_block_recent_anchor_reacquisition():
    manager = GlobalIdentityManager(np.empty((0, 2), np.float32))
    candidate_vector = np.asarray([1.0, 0.0], np.float32)

    person3_vector = np.asarray([0.607, (1.0 - 0.607**2) ** 0.5], np.float32)
    person3 = manager.confirm_new(
        {
            **obs(camera="CAM-01", native=3, frame=1, world=(10.6, -14.6)),
            "timestamp": "2026-09-23T12:22:57.689Z",
        },
        person3_vector,
        "test_person3",
    )
    manager.identities[person3].persisted_anchor = {
        "world": (10.6, -14.6),
        "timestamp": "2026-09-23T12:22:57.689Z",
        "camera_id": "CAM-01",
    }

    person4_vector = np.asarray([0.602, (1.0 - 0.602**2) ** 0.5], np.float32)
    person4 = manager.confirm_new(
        {
            **obs(camera="CAM-01", native=4, frame=1, world=(39.9, -26.9)),
            "timestamp": "2026-09-23T05:59:21.146Z",
        },
        person4_vector,
        "test_person4",
    )
    manager.identities[person4].persisted_anchor = {
        "world": (39.9, -26.9),
        "timestamp": "2026-09-23T05:59:21.146Z",
        "camera_id": "CAM-01",
    }

    reacquired = {
        **obs(camera="CAM-04", native=34, frame=3069, world=(12.13, -11.11)),
        "timestamp": "2026-09-23T12:27:12.480Z",
        "pending_good_crops": 3,
    }
    candidates = manager._persistent_reacquisition_scores(reacquired, candidate_vector, [])
    assert candidates
    assert candidates[0][1] == person3
    assert candidates[0][2]["reason"] == "persistent_recent_world_reacquisition"


def test_two_spatially_plausible_recent_gallery_candidates_remain_ambiguous():
    manager = GlobalIdentityManager(np.empty((0, 2), np.float32))
    candidate_vector = np.asarray([1.0, 0.0], np.float32)
    for native, world, vector, stamp in (
        (3, (10.6, -14.6), np.asarray([0.607, (1.0 - 0.607**2) ** 0.5], np.float32), "2026-09-23T12:22:57.689Z"),
        (4, (13.0, -11.0), np.asarray([0.602, (1.0 - 0.602**2) ** 0.5], np.float32), "2026-09-23T12:22:57.689Z"),
    ):
        gid = manager.confirm_new(
            {**obs(camera="CAM-01", native=native, frame=1, world=world), "timestamp": stamp},
            vector,
            "test_recent_candidate",
        )
        manager.identities[gid].persisted_anchor = {
            "world": world,
            "timestamp": stamp,
            "camera_id": "CAM-01",
        }
    reacquired = {
        **obs(camera="CAM-04", native=34, frame=3069, world=(12.13, -11.11)),
        "timestamp": "2026-09-23T12:27:12.480Z",
        "pending_good_crops": 3,
    }
    assert manager._persistent_reacquisition_scores(reacquired, candidate_vector, []) == []
    assert manager._persistent_match_ambiguous


def test_async_same_camera_peer_blocks_persistent_reid_for_simultaneous_different_target():
    manager = GlobalIdentityManager(np.empty((0, 2), np.float32))
    known = {
        **obs(camera="CAM-01", native=1, frame=60, world=(19.804, -28.495)),
        "timestamp": "2026-09-23T12:36:39.270Z",
    }
    gid = manager.confirm_new(known, np.asarray([1.0, 0.0], np.float32), "test_initial")
    simultaneous_other = {
        **obs(camera="CAM-01", native=3, frame=61, world=(8.682, -31.559)),
        "timestamp": "2026-09-23T12:36:39.309Z",
        "pending_good_crops": 3,
        "bbox": (250.0, 10.0, 350.0, 200.0),
    }
    # No frame_assignments reproduces the async path from the live run.
    resolved, evidence = manager.resolve_existing(
        simultaneous_other,
        np.asarray([0.792, (1.0 - 0.792**2) ** 0.5], np.float32),
        [],
    )
    assert resolved is None
    assert evidence["reason"] == "pending_no_acceptable_candidate"
    trace = manager.candidate_trace(
        simultaneous_other,
        np.asarray([0.792, (1.0 - 0.792**2) ** 0.5], np.float32),
        [],
    )
    assert trace[0]["active_conflict_reason"] == "same_camera_simultaneous"
    assert trace[0]["exact_rejection_reason"] == "same_camera_simultaneous"
    assert manager.root(gid) == gid


def test_async_same_camera_close_fragment_retains_identity():
    manager = GlobalIdentityManager(np.empty((0, 2), np.float32))
    vector = np.asarray([1.0, 0.0], np.float32)
    known = {
        **obs(camera="CAM-01", native=1, frame=60, world=(19.804, -28.495)),
        "timestamp": "2026-09-23T12:36:39.270Z",
    }
    gid = manager.confirm_new(known, vector, "test_initial")
    close_fragment = {
        **obs(camera="CAM-01", native=2, frame=61, world=(20.1, -28.6)),
        "timestamp": "2026-09-23T12:36:39.309Z",
    }
    resolved, evidence = manager.resolve_existing(close_fragment, vector, [])
    assert resolved == gid
    assert evidence["reason"] == "same_camera_direct_continuity"


def test_poor_crop_does_not_force_new_identity_when_geometry_is_compatible():
    manager = GlobalIdentityManager(np.empty((0, 2), np.float32))
    first, _, _ = manager.process(obs(camera="CAM-01", native=1, frame=20, world=(10.0, 10.0)), np.asarray([1.0, 0.0], np.float32), [])
    poor = obs(camera="CAM-04", native=9, frame=20, world=(10.8, 10.4))
    poor["crop_quality_usable"] = False
    resolved, _ = manager.resolve_existing(poor, None, [{**obs(camera="CAM-01", native=1, frame=20, world=(10.0, 10.0)), "global_person_id_num": first}])
    assert resolved == first
    assert manager.next_number == 2



def test_multiple_poor_cam04_crops_keep_existing_canonical_identity():
    manager = GlobalIdentityManager(np.empty((0, 2), np.float32))
    first, _, _ = manager.process(
        obs(camera="CAM-01", native=1, frame=100, world=(12.0, 8.0)),
        np.asarray([1.0, 0.0], np.float32),
        [],
    )
    for frame in (101, 102, 103, 104, 105):
        poor = obs(camera="CAM-04", native=7, frame=frame, world=(12.4, 8.2))
        poor["crop_quality_usable"] = False
        resolved, evidence = manager.resolve_existing(
            poor,
            None,
            [{**obs(camera="CAM-01", native=1, frame=frame, world=(12.0, 8.0)), "global_person_id_num": first}],
        )
        assert resolved == first
        assert evidence["reason"] in {"cross_camera_geometry_time", "same_camera_direct_continuity"}
        manager.accept_existing(resolved, poor, None, evidence)
    assert manager.next_number == 2


def test_recent_conflicting_track_blocks_bounded_history_reid():
    manager = GlobalIdentityManager(np.empty((0, 4), np.float32))
    first_obs = obs(native=31, frame=100, world=(39.8, -25.2))
    gid = manager.confirm_new(first_obs, np.asarray([1.0, 0.0, 0.0, 0.0], np.float32), "test_initial")
    # A later native fragment temporarily overwrites the latest camera state.
    manager.accept_existing(
        gid,
        obs(native=32, frame=150, world=(10.8, -11.8)),
        np.asarray([0.9, 0.1, 0.0, 0.0], np.float32),
        {"reason": "test_fragment"},
    )
    reacquired = obs(native=33, frame=160, world=(39.3, -27.0))
    resolved, evidence = manager.resolve_existing(reacquired, np.asarray([1.0, 0.0, 0.0, 0.0], np.float32), [])
    assert resolved is None
    assert evidence["reason"] == "pending_no_acceptable_candidate"
    trace = manager.candidate_trace(reacquired, np.asarray([1.0, 0.0, 0.0, 0.0], np.float32), [])
    assert trace[0]["exact_rejection_reason"] == "same_camera_simultaneous"
    assert manager.next_number == 2


def test_active_cross_camera_impossibility_blocks_historical_same_camera_reid():
    manager = GlobalIdentityManager(np.empty((0, 4), np.float32))
    vector = np.asarray([1.0, 0.0, 0.0, 0.0], np.float32)

    first = obs(native=31, frame=100, world=(21.0, -7.5))
    first["timestamp"] = "2026-09-23T10:00:00.000Z"
    gid = manager.confirm_new(first, vector, "test_initial")
    manager.accept_existing(
        gid,
        {**obs(native=31, frame=200, world=(21.1, -7.4)), "timestamp": "2026-09-23T10:00:10.000Z"},
        vector,
        {"reason": "test_same_camera_history"},
    )
    # This is still the same canonical identity, currently visible from CAM-04,
    # but 26 m from the historical CAM-01 point.
    manager.accept_existing(
        gid,
        {**obs(camera="CAM-04", native=30, frame=1000, world=(9.8, -31.1)), "timestamp": "2026-09-23T10:01:00.000Z"},
        vector,
        {"reason": "test_active_peer"},
    )

    reacquired = {
        **obs(camera="CAM-01", native=34, frame=1002, world=(21.2, -7.6)),
        "timestamp": "2026-09-23T10:01:00.040Z",
    }
    resolved, evidence = manager.resolve_existing(reacquired, vector, [])

    assert resolved is None
    assert evidence["reason"] == "pending_no_acceptable_candidate"
    assert manager.next_number == 2


def test_repeated_good_crops_confirm_new_person_when_stale_history_conflicts_with_active_peer():
    manager = GlobalIdentityManager(np.empty((0, 4), np.float32))
    vector = np.asarray([1.0, 0.0, 0.0, 0.0], np.float32)
    gid = manager.confirm_new(
        {**obs(native=31, frame=100, world=(21.0, -7.5)), "timestamp": "2026-09-23T10:00:00.000Z"},
        vector,
        "test_initial",
    )
    manager.accept_existing(
        gid,
        {**obs(native=32, frame=200, world=(40.0, -25.0)), "timestamp": "2026-09-23T10:00:10.000Z"},
        vector,
        {"reason": "test_fragment"},
    )
    manager.accept_existing(
        gid,
        {**obs(camera="CAM-04", native=30, frame=1000, world=(40.0, -25.0)), "timestamp": "2026-09-23T10:01:00.000Z"},
        vector,
        {"reason": "test_active_peer"},
    )
    candidate = {
        **obs(camera="CAM-01", native=34, frame=1002, world=(21.2, -7.6)),
        "timestamp": "2026-09-23T10:01:00.040Z",
    }
    active = [
        {**obs(native=32, frame=1002, world=(40.0, -25.0)), "global_person_id_num": gid,
         "timestamp": "2026-09-23T10:01:00.000Z"},
        {**obs(camera="CAM-04", native=30, frame=1000, world=(40.0, -25.0)),
         "global_person_id_num": gid, "timestamp": "2026-09-23T10:01:00.000Z"},
    ]

    weak = manager.novelty_evidence(candidate, vector, active, good_crops=2, attempts=2)
    confirmed = manager.novelty_evidence(candidate, vector, active, good_crops=3, attempts=3)

    assert weak["positive"] is False
    assert weak["reason"] == "historical_continuity_unknown_not_novelty"
    assert confirmed["positive"] is True
    assert confirmed["reason"] == "active_spatial_conflict_overrides_stale_history"
    assert confirmed["candidates"][0]["historical_same_camera_evidence"]["frame"] == 100


def test_stale_motion_rejection_is_not_positive_novelty():
    manager = GlobalIdentityManager(np.empty((0, 4), np.float32))
    gid = manager.confirm_new(obs(native=41, frame=100, world=(39.8, -25.2)), np.asarray([1.0, 0.0, 0.0, 0.0], np.float32), "test_initial")
    manager.accept_existing(
        gid, obs(native=42, frame=150, world=(10.8, -11.8)),
        np.asarray([0.0, 1.0, 0.0, 0.0], np.float32), {"reason": "test_fragment"},
    )
    pending = obs(native=43, frame=160, world=(39.3, -27.0))
    novelty = manager.novelty_evidence(pending, np.asarray([0.0, 0.0, 1.0, 0.0], np.float32), [], 2, 2)
    assert novelty["positive"] is False
    assert novelty["reason"] == "historical_continuity_unknown_not_novelty"
    assert manager.next_number == 2


def test_co_located_native_fragment_handoff_preserves_canonical_identity():
    manager = GlobalIdentityManager(np.empty((0, 4), np.float32))
    first = obs(native=101, frame=100, world=(24.0, -12.0))
    first["bbox"] = (100.0, 100.0, 300.0, 500.0)
    vector = np.asarray([1.0, 0.0, 0.0, 0.0], np.float32)
    gid, _, _ = manager.process(first, vector, [])
    predecessor = obs(native=102, frame=110, world=(24.1, -12.1))
    predecessor["bbox"] = first["bbox"]
    manager.accept_existing(gid, predecessor, vector, {"reason": "test_predecessor"})
    fragment = obs(native=103, frame=111, world=(24.2, -12.0))
    fragment["bbox"] = first["bbox"]
    active = [{**predecessor, "global_person_id_num": gid}]
    resolved, evidence = manager.resolve_existing(fragment, vector, active)
    assert resolved == gid
    assert evidence["reason"] == "same_camera_fragment_handoff"
    assert manager.next_number == 2

def test_supported_image_reid_handoff_after_confirmed_track_gap():
    manager = GlobalIdentityManager(np.empty((0, 2), np.float32))
    predecessor = obs(camera="CAM-04", native=62, frame=2112, world=(12.20, -10.97))
    predecessor["bbox"] = (255.96, 289.89, 431.74, 511.92)
    gid = manager.confirm_new(predecessor, np.asarray([1.0, 0.0], np.float32), "test_initial")
    fragment = obs(camera="CAM-04", native=65, frame=2159, world=(10.69, -14.67))
    fragment["bbox"] = (249.97, 270.35, 448.63, 614.38)
    fragment["pending_good_crops"] = 3
    weak = np.asarray([0.47, np.sqrt(1.0 - 0.47 ** 2)], np.float32)
    resolved, evidence = manager.resolve_existing(fragment, weak, [])
    assert resolved == gid
    assert evidence["reason"] == "same_camera_supported_image_reid_handoff"
    assert manager.next_number == 2
    fragment["pending_good_crops"] = 1
    resolved, _ = manager.resolve_existing(fragment, weak, [])
    assert resolved is None
    fragment["pending_good_crops"] = 3
    unsupported = np.asarray([0.30, np.sqrt(1.0 - 0.30 ** 2)], np.float32)
    resolved, _ = manager.resolve_existing(fragment, unsupported, [])
    assert resolved is None
    active = [{**predecessor, "frame": 2159, "global_person_id_num": gid}]
    resolved, _ = manager.resolve_existing(fragment, weak, active)
    assert resolved is None
