from datetime import datetime, timedelta, timezone

import numpy as np

from services.mv3dt_room.global_identity_manager import GlobalIdentityManager
from services.mv3dt_room.identity_gallery_store import IdentityGalleryStore
from services.mv3dt_room.presence import CurrentPresencePositionFilter, assert_presence_contract


def make_obs(camera, native, frame, when, world):
    return {
        "camera_id": camera,
        "native_track_id": native,
        "frame": frame,
        "timestamp": when.isoformat().replace("+00:00", "Z"),
        "bbox": (10.0, 10.0, 110.0, 230.0),
        "world": world,
        "crop_quality_usable": True,
        "crop_quality": {
            "area_fraction": 0.05, "inside_fraction": 1.0, "height": 500.0,
        },
    }


def seed_gallery(store_path):
    store = IdentityGalleryStore(store_path)
    manager = GlobalIdentityManager(np.empty((0, 4), np.float32), gallery_store=store)
    start = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)
    white = manager.confirm_new(
        make_obs("CAM-01", 1, 100, start, (21.0, -29.0)),
        np.asarray([1.0, 0.0, 0.0, 0.0], np.float32),
        "test_initial_white",
    )
    black = manager.confirm_new(
        make_obs("CAM-01", 2, 100, start, (30.0, -20.0)),
        np.asarray([0.0, 1.0, 0.0, 0.0], np.float32),
        "test_initial_black",
    )
    # Add a second representative from the other camera, then simulate the
    # person leaving: no active assignments are supplied after this point.
    manager.accept_existing(
        white, make_obs("CAM-04", 7, 101, start, (21.2, -29.1)),
        np.asarray([0.98, 0.0, 0.20, 0.0], np.float32), {"reason": "test_cam04"},
    )
    manager.accept_existing(
        black, make_obs("CAM-04", 8, 101, start, (30.2, -20.1)),
        np.asarray([0.0, 0.98, 0.20, 0.0], np.float32), {"reason": "test_cam04"},
    )
    return store, manager, start, white, black


def test_long_gap_reacquires_both_people_without_visible_ghosts(tmp_path):
    store, manager, start, white, black = seed_gallery(tmp_path / "gallery.sqlite3")
    filter_ = CurrentPresencePositionFilter()
    active = [
        {"camera_id": "CAM-01", "application_id": "Person_01", "world": (21.0, -29.0), "frame": 100},
        {"camera_id": "CAM-01", "application_id": "Person_02", "world": (30.0, -20.0), "frame": 100},
    ]
    assert_presence_contract(active, {"Person_01", "Person_02"})
    assert filter_.update(active) == {"Person_01": (21.0, -29.0), "Person_02": (30.0, -20.0)}
    assert filter_.update([]) == {}
    assert_presence_contract([], set())

    for gap_minutes in (5, 60, 360):
        when = start + timedelta(minutes=gap_minutes)
        recovered_white, evidence_white = manager.resolve_existing(
            make_obs("CAM-04", 70 + gap_minutes, 100 + gap_minutes * 20, when, (40.0, -5.0)),
            np.asarray([1.0, 0.0, 0.0, 0.0], np.float32), [],
        )
        recovered_black, evidence_black = manager.resolve_existing(
            make_obs("CAM-01", 80 + gap_minutes, 101 + gap_minutes * 20, when, (-10.0, 11.0)),
            np.asarray([0.0, 1.0, 0.0, 0.0], np.float32), [],
        )
        assert recovered_white == white
        assert recovered_black == black
        assert evidence_white["reason"] in {"persistent_gallery_reacquisition", "rolling_reid"}
        assert evidence_black["reason"] in {"persistent_gallery_reacquisition", "rolling_reid"}

    assert manager.next_number == 3
    assert manager.persistence_stats["gallery_entries_loaded"] == 0
    store.close()


def test_restart_reloads_gallery_but_not_visible_presence(tmp_path):
    path = tmp_path / "gallery.sqlite3"
    store, manager, start, white, black = seed_gallery(path)
    store.close()

    reloaded_store = IdentityGalleryStore(path)
    reloaded = GlobalIdentityManager(np.empty((0, 4), np.float32), gallery_store=reloaded_store)
    assert reloaded.application_id(white) == "Person_01"
    assert reloaded.application_id(black) == "Person_02"
    assert reloaded.persistence_stats["identities_loaded"] == 2
    assert reloaded.persistence_stats["gallery_entries_loaded"] == 4

    # A restart restores memory only; no active observations means no marker.
    assert_presence_contract([], set())
    recovered, evidence = reloaded.resolve_existing(
        make_obs("CAM-04", 901, 7200, start + timedelta(hours=1), (100.0, 100.0)),
        np.asarray([1.0, 0.0, 0.0, 0.0], np.float32), [],
    )
    assert recovered == white
    assert evidence["reason"] == "persistent_gallery_reacquisition"
    assert reloaded.next_number == 3
    reloaded_store.close()


def test_ambiguous_or_unknown_reentry_stays_pending_without_allocation(tmp_path):
    store, manager, start, white, black = seed_gallery(tmp_path / "gallery.sqlite3")
    unknown, evidence = manager.resolve_existing(
        make_obs("CAM-01", 99, 7200, start + timedelta(hours=1), (0.0, 0.0)),
        np.asarray([0.7071, 0.7071, 0.0, 0.0], np.float32), [],
    )
    assert unknown is None
    assert evidence["reason"] == "pending_no_acceptable_candidate"
    assert manager.next_number == 3
    store.close()


def test_recent_fragment_continuity_beats_weaker_gallery_match(tmp_path):
    store = IdentityGalleryStore(tmp_path / "recent.sqlite3")
    manager = GlobalIdentityManager(np.empty((0, 4), np.float32), gallery_store=store)
    start = datetime(2026, 9, 23, 4, 0, tzinfo=timezone.utc)
    first_box = (2048.0, 651.0, 2189.0, 873.0)
    new_box = (2010.0, 653.0, 2169.0, 910.0)
    first = make_obs("CAM-01", 1, 100, start, (8.4, -10.7))
    first["bbox"] = first_box
    gid_first = manager.confirm_new(first, np.asarray([1., 0., 0., 0.], np.float32))
    other = make_obs("CAM-01", 2, 100, start, (40.1, -25.5))
    other["bbox"] = (368., 1043., 623., 1388.)
    gid_other = manager.confirm_new(other, np.asarray([0., 1., 0., 0.], np.float32))
    latest_first = make_obs("CAM-01", 1, 212, start + timedelta(seconds=5.6), (8.4, -10.7))
    latest_first["bbox"] = first_box
    manager.accept_existing(gid_first, latest_first, None, {"reason": "sticky"})
    latest_other = make_obs("CAM-01", 2, 147, start + timedelta(seconds=2.35), (40.3, -25.8))
    latest_other["bbox"] = other["bbox"]
    manager.accept_existing(gid_other, latest_other, None, {"reason": "sticky"})
    fragment = make_obs("CAM-01", 5, 220, start + timedelta(seconds=6), (10.3, -10.6))
    fragment["bbox"] = new_box
    # The new crop resembles both galleries, but direct track continuity
    # and a 33 m impossible displacement identify the first person.
    vector = np.asarray([0.85, 0.527, 0., 0.], np.float32)
    resolved, evidence = manager.resolve_existing(fragment, vector, [])
    assert resolved == gid_first
    assert evidence["reason"] == "same_camera_direct_continuity"
    assert manager.next_number == 3
    store.close()



def test_strong_image_handoff_resolves_short_crop_unavailable_fragment():
    manager = GlobalIdentityManager(np.empty((0, 4), np.float32))
    start = datetime(2026, 9, 23, 5, 38, 29, tzinfo=timezone.utc)
    old = make_obs("CAM-01", 14, 1740, start, (39.94, -26.13))
    old["bbox"] = (359.38, 1099.22, 647.58, 1367.25)
    gid = manager.confirm_new(old, np.asarray([1., 0., 0., 0.], np.float32))
    fragment = make_obs("CAM-01", 15, 1776, start + timedelta(seconds=1.8), (39.84, -26.61))
    fragment["bbox"] = (358.19, 1098.86, 639.56, 1351.33)
    resolved, evidence = manager.resolve_existing(fragment, None, [])
    assert resolved == gid
    assert evidence["reason"] == "same_camera_strong_image_handoff"
    far_image = {**fragment, "bbox": (1492.0, 506.0, 1662.0, 683.0)}
    assert manager.resolve_existing(far_image, None, [])[0] is None
    too_late = {**fragment, "frame": 1800, "timestamp": (start + timedelta(seconds=3)).isoformat()}
    assert manager.resolve_existing(too_late, None, [])[0] is None
    assert manager.next_number == 2

def test_recent_cross_camera_impossible_motion_stays_pending(tmp_path):
    store = IdentityGalleryStore(tmp_path / "recent_cross.sqlite3")
    manager = GlobalIdentityManager(np.empty((0, 4), np.float32), gallery_store=store)
    start = datetime(2026, 9, 23, 4, 0, tzinfo=timezone.utc)
    previous = make_obs("CAM-01", 2, 953, start, (39.8, -27.3))
    gid = manager.confirm_new(previous, np.asarray([0., 1., 0., 0.], np.float32))
    reentry = make_obs("CAM-04", 14, 966, start + timedelta(milliseconds=650), (10.2, -31.5))
    resolved, evidence = manager.resolve_existing(
        reentry, np.asarray([0.0, 1.0, 0.0, 0.0], np.float32), []
    )
    assert resolved is None
    assert evidence["reason"] == "pending_no_acceptable_candidate"
    assert manager.persistence_stats["reacquisition_rejected_recent_world_motion"] >= 1
    assert manager.next_number == gid + 1
    store.close()




def test_recent_persisted_world_anchor_resolves_viewpoint_shift_after_restart(tmp_path):
    path = tmp_path / "gallery.sqlite3"
    store, manager, start, white, black = seed_gallery(path)
    store.close()
    reloaded_store = IdentityGalleryStore(path)
    reloaded = GlobalIdentityManager(np.empty((0, 4), np.float32), gallery_store=reloaded_store)
    assert reloaded.identities[black].last_by_camera == {}  # No visible state restored.
    obs = make_obs("CAM-01", 90, 400, start + timedelta(seconds=30), (30.4, -20.2))
    vector = np.asarray([0.0, 0.62, 0.0, 0.784], np.float32)
    obs["pending_good_crops"] = 1
    assert reloaded.resolve_existing(obs, vector, [])[0] is None
    obs["pending_good_crops"] = 2
    far = {**obs, "world": (100.0, 100.0)}
    assert reloaded.resolve_existing(far, vector, [])[0] is None
    within_recent_window = {**obs, "timestamp": (start + timedelta(minutes=20)).isoformat()}
    assert reloaded.resolve_existing(within_recent_window, vector, [])[0] == black
    aged = {**obs, "timestamp": (start + timedelta(hours=1)).isoformat()}
    assert reloaded.resolve_existing(aged, vector, [])[0] is None
    recovered, evidence = reloaded.resolve_existing(obs, vector, [])
    assert recovered == black
    assert evidence["reason"] == "persistent_recent_world_reacquisition"
    assert evidence["persisted_anchor_world_distance_m"] < 1.0
    assert reloaded.next_number == 3
    reloaded_store.close()


def test_unavailable_active_person_does_not_block_recent_anchor_match(tmp_path):
    path = tmp_path / "gallery.sqlite3"
    store, manager, start, white, black = seed_gallery(path)
    store.close()
    reloaded_store = IdentityGalleryStore(path)
    reloaded = GlobalIdentityManager(np.empty((0, 4), np.float32), gallery_store=reloaded_store)
    obs = make_obs("CAM-01", 90, 400, start + timedelta(seconds=30), (30.4, -20.2))
    obs["pending_good_crops"] = 3
    vector = np.asarray([0.62, 0.65, 0.0, 0.441], np.float32)
    active_white = {**make_obs("CAM-01", 1, 400, start + timedelta(seconds=30),
                               (21.0, -29.0)), "global_person_id_num": white}
    recovered, evidence = reloaded.resolve_existing(obs, vector, [active_white])
    assert recovered == black
    assert evidence["reason"] == "persistent_recent_world_reacquisition"
    assert reloaded.next_number == 3
    reloaded_store.close()

def test_weak_gallery_only_resemblance_is_not_a_reacquisition(tmp_path):
    store, manager, start, white, black = seed_gallery(tmp_path / "gallery.sqlite3")
    # Captured false live match: appearance 0.530 without continuity.
    observation = make_obs("CAM-01", 9, 5000, start + timedelta(minutes=5), (100.0, 100.0))
    embedding = np.asarray([0.0, 0.53, 0.0, 0.848], np.float32)
    resolved, evidence = manager.resolve_existing(observation, embedding, [])
    assert resolved is None
    assert evidence["reason"] == "pending_no_acceptable_candidate"
    assert manager.next_number == 3
    assert manager.persistence_stats["reacquisition_rejected_gallery_only_similarity"] >= 1
    store.close()


def test_stale_gallery_aggregate_only_score_cannot_reuse_identity(tmp_path, monkeypatch):
    """Regression for the empty-room CAM-04 false Person_06 reacquisition."""
    store = IdentityGalleryStore(tmp_path / "gallery.sqlite3")
    manager = GlobalIdentityManager(np.empty((0, 2), np.float32), gallery_store=store)
    anchor_time = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
    gid = manager.confirm_new(
        make_obs("CAM-01", 1, 100, anchor_time, (24.0, -20.0)),
        np.asarray([1.0, 0.0], np.float32),
        "test_stale_anchor",
    )
    manager.identities[gid].persisted_anchor = {
        "world": (24.0, -20.0),
        "timestamp": anchor_time.isoformat(),
        "camera_id": "CAM-01",
    }
    now = datetime(2026, 9, 24, 14, 30, 35, tzinfo=timezone.utc)
    observation = make_obs("CAM-04", 23, 2322, now, (37.5539, -27.1330))
    observation["pending_good_crops"] = 1

    # Captured false case: pooled gallery score crossed 0.70, but neither
    # camera-specific score did; the persisted anchor was stale and distant.
    monkeypatch.setattr(
        manager,
        "appearance",
        lambda vector, ident: (0.705637, {"CAM-01": 0.666029, "CAM-04": 0.672060}),
    )
    assert manager.resolve_existing(
        observation, np.asarray([1.0, 0.0], np.float32), []
    )[0] is None
    assert manager.persistence_stats["reacquisition_rejected_gallery_camera_support"] == 1
    assert manager.next_number == gid + 1
    false_candidate = manager.candidate_trace(
        observation, np.asarray([1.0, 0.0], np.float32), []
    )[0]
    assert false_candidate["exact_rejection_reason"] == "stale_gallery_missing_current_camera_support"
    assert false_candidate["persisted_anchor_age_ms"] > 30 * 60 * 1000
    assert false_candidate["persisted_anchor_distance_m"] > 8.5
    assert false_candidate["gallery_camera_similarity"] == 0.672060

    # The operator-labeled genuine CAM-04 reacquisition had strong
    # current-camera evidence and remains eligible under the unchanged gate.
    monkeypatch.setattr(
        manager,
        "appearance",
        lambda vector, ident: (0.895649, {"CAM-01": 0.748123, "CAM-04": 0.928822}),
    )
    nearby = {**observation, "world": (24.1, -20.1)}
    recovered, evidence = manager.resolve_existing(
        nearby, np.asarray([1.0, 0.0], np.float32), []
    )
    assert recovered == gid
    assert evidence["reason"] == "persistent_gallery_reacquisition"
    assert evidence["current_camera_gallery_similarity"] == 0.928822
    assert manager.next_number == gid + 1
    store.close()

def test_retained_native_track_is_not_simultaneously_visible_after_camera_advances(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "services/mv3dt_room"))
    from services.mv3dt_room.live_identity_worker import LiveIdentityWorker, TrackState

    worker = LiveIdentityWorker.__new__(LiveIdentityWorker)
    worker.manager = GlobalIdentityManager(np.empty((0, 4), np.float32))
    gid = worker.manager.create(100, "test")
    old = TrackState("CAM-01", 1)
    old.latest = make_obs("CAM-01", 1, 212, datetime.now(timezone.utc), (8.4, -10.7))
    old.canonical_internal_identity = gid
    worker.track_states = {("CAM-01", 1): old}
    worker.latest_camera_frame = {"CAM-01": 220, "CAM-04": 220}
    assert worker._active_assignments() == []
    worker.latest_camera_frame["CAM-01"] = 212
    assert len(worker._active_assignments()) == 1
