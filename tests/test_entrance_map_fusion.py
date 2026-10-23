"""Injected identity fixtures test display mechanics, not OSNet accuracy."""
import copy
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from services.mv3dt_staging.map_fusion import (
    ACCEPTED_STATUS, ARRIVAL_DOMAIN, IDENTITY_AUTHORITY, ExperimentalMapFusion,
)

NOW = 10_000_000_000


def state(camera, local="4", xy=(200, 200), *, age_ns=0, frame=10, pts=1_000):
    sid = {"CAM-02": 0, "CAM-05": 1}[camera]
    return {"camera_id": camera, "source_id": sid, "frame_index": frame,
            "source_pts_ns": pts, "updated_mono_ns": NOW - age_ns,
            "observation_time_domain": ARRIVAL_DOMAIN, "status": "LIVE",
            "markers": [{"camera_id": camera, "source_id": sid,
                "frame_index": frame, "source_pts_ns": pts, "local_tracker_id": local,
                "layout_xy": list(xy), "jump_candidate": False}]}


def identities(pairs=None, *, revision=1, sampled=NOW):
    pairs = pairs if pairs is not None else {"CAM-02": {"4": "G001"}, "CAM-05": {"0": "G001"}}
    return {"authority": IDENTITY_AUTHORITY, "revision": revision,
            "sampled_monotonic_ns": sampled, "assignments": {
                cam: {local: {"provisional_global_id": gid, "status": ACCEPTED_STATUS,
                      "identity_memory_state": "ACTIVE", "persistent": False,
                      "identity_expires_monotonic_ns": NOW + 600_000_000_000}
                      for local, gid in pairs.get(cam, {}).items()}
                for cam in ("CAM-02", "CAM-05")}}


def scene():
    return {"CAM-02": state("CAM-02"),
            "CAM-05": state("CAM-05", "0", (220, 210), frame=80, pts=999_000)}


def engine():
    return ExperimentalMapFusion(layout_width=1448, layout_height=1086)


def test_same_accepted_gid_has_one_marker_with_both_exact_source_keys():
    result = engine().update(scene(), identities(), now_mono_ns=NOW)
    assert len(result["markers"]) == 1
    marker = result["markers"][0]
    assert marker["label"] == "G001?"
    assert marker["layout_xy"] == [210.0, 205.0]
    assert marker["fusion_status"] == "EXPERIMENTAL_MAP_PIXEL_MEAN"
    assert [(o["source_id"], o["frame_index"], o["source_pts_ns"]) for o in marker["contributors"]] == [
        (0, 10, 1_000), (1, 80, 999_000)]
    assert result["policy"]["verified_world_frame"] is False


@pytest.mark.parametrize("mapping", [
    {"CAM-02": {"4": "G001"}, "CAM-05": {"0": "G002"}}, {},
])
def test_different_or_unaccepted_ids_remain_separate(mapping):
    out = engine().update(scene(), identities(mapping), now_mono_ns=NOW)
    assert len(out["markers"]) == 2


def test_one_camera_observation_has_one_provisional_marker():
    out = engine().update({"CAM-02": state("CAM-02")}, identities(), now_mono_ns=NOW)
    assert len(out["markers"]) == 1
    assert out["markers"][0]["fusion_status"] == "SINGLE_CAMERA_PROVISIONAL"


def test_withdrawal_splits_immediately_and_clears_gid_trail_and_label():
    fusion = engine()
    first = fusion.update(scene(), identities(), now_mono_ns=NOW)
    second = fusion.update(scene(), identities({}, revision=2), now_mono_ns=NOW)
    assert len(first["markers"]) == 1 and len(second["markers"]) == 2
    assert all("G001" not in m["label"] for m in second["markers"])
    assert "global:G001" not in fusion._trails
    assert set(fusion._trails) == {"local:CAM-02:4", "local:CAM-05:0"}


@pytest.mark.parametrize("changed", [("9", "0"), ("9", "11")])
def test_new_local_ids_use_old_gid_only_after_fresh_accepted_binding(changed):
    fusion = engine()
    fusion.update(scene(), identities(), now_mono_ns=NOW)
    a, b = changed
    next_scene = {"CAM-02": state("CAM-02", a), "CAM-05": state("CAM-05", b)}
    assert all(m["key"].startswith("local:") for m in
               fusion.update(next_scene, identities({}, revision=2), now_mono_ns=NOW)["markers"])
    out = fusion.update(next_scene, identities({"CAM-02": {a: "G001"}, "CAM-05": {b: "G001"}},
                                              revision=3), now_mono_ns=NOW)
    assert len(out["markers"]) == 1 and out["markers"][0]["label"] == "G001?"
    assert {o["local_tracker_id"] for o in out["markers"][0]["contributors"]} == {a, b}


@pytest.mark.parametrize("field,value", [("identity_memory_state", "EXPIRED"),
    ("identity_memory_state", "LOST"), ("status", "REJECTED"), ("persistent", True)])
def test_expired_lost_rejected_or_persistent_bindings_cannot_fuse(field, value):
    mapping = identities()
    for camera in mapping["assignments"].values():
        for binding in camera.values():
            binding[field] = value
    result = engine().update(scene(), mapping, now_mono_ns=NOW)
    assert all(m["provisional_global_id"] is None for m in result["markers"])


def test_competing_same_camera_tracks_cannot_share_a_primary_gid():
    states = scene()
    extra = copy.deepcopy(states["CAM-02"]["markers"][0])
    extra["local_tracker_id"] = "9"
    states["CAM-02"]["markers"].append(extra)
    mapping = identities({"CAM-02": {"4": "G001", "9": "G001"}, "CAM-05": {"0": "G001"}})
    out = engine().update(states, mapping, now_mono_ns=NOW)
    assert len(out["markers"]) == 3
    assert all(m["provisional_global_id"] is None for m in out["markers"])
    assert "SAME_CAMERA_MULTIPLE_TRACK_CONFLICT" in out["groups"][0]["reasons"]


@pytest.mark.parametrize("reason", ["SPATIAL_CONTRADICTION", "HOST_ARRIVAL_SKEW",
                                         "UNSUPPORTED_TIMESTAMP_DOMAIN", "LOCAL_JUMP_CANDIDATE"])
def test_contradictions_are_flagged_and_not_averaged(reason):
    states = scene()
    other = states["CAM-05"]
    if reason == "SPATIAL_CONTRADICTION": other["markers"][0]["layout_xy"] = [600, 600]
    if reason == "HOST_ARRIVAL_SKEW": other["updated_mono_ns"] = NOW - 151_000_000
    if reason == "UNSUPPORTED_TIMESTAMP_DOMAIN": other["observation_time_domain"] = "UNKNOWN_SOURCE_CLOCK"
    if reason == "LOCAL_JUMP_CANDIDATE": other["markers"][0]["jump_candidate"] = True
    result = engine().update(states, identities(), now_mono_ns=NOW)
    assert len(result["markers"]) == 2
    assert all(m["provisional_global_id"] is None for m in result["markers"])
    assert reason in result["groups"][0]["reasons"]


def test_equal_frame_numbers_do_not_override_timestamp_skew():
    states = scene()
    states["CAM-05"] = state("CAM-05", "0", age_ns=400_000_000)
    out = engine().update(states, identities(), now_mono_ns=NOW)
    assert out["groups"][0]["decision"] == "SPLIT"


def test_stale_sources_are_removed_instead_of_fused_with_current_frames():
    states = scene()
    states["CAM-05"]["updated_mono_ns"] = NOW - 751_000_000
    out = engine().update(states, identities(), now_mono_ns=NOW)
    assert len(out["markers"]) == 1 and len(out["markers"][0]["contributors"]) == 1
    assert out["rejected"]["invalid_or_stale_source_snapshot"] == 1


@pytest.mark.parametrize("field,value", [("source_id", 0), ("frame_index", 79), ("source_pts_ns", 999)])
def test_bad_frame_source_or_pts_is_quarantined(field, value):
    states = scene()
    states["CAM-05"]["markers"][0][field] = value
    out = engine().update(states, identities(), now_mono_ns=NOW)
    assert len(out["markers"]) == 1
    assert out["rejected"]["invalid_observation_attribution_or_position"] == 1


def test_repeated_paint_does_not_duplicate_markers_or_trails_and_memory_is_bounded():
    fusion, states = engine(), scene()
    for _ in range(30):
        out = fusion.update(states, identities(), now_mono_ns=NOW)
    assert len(out["markers"]) == 1 and len(out["markers"][0]["trail"]) == 1
    for index in range(30):
        for state_row in states.values():
            state_row["frame_index"] += 1
            state_row["source_pts_ns"] += 1
            for obs in state_row["markers"]:
                obs["frame_index"] = state_row["frame_index"]
                obs["source_pts_ns"] = state_row["source_pts_ns"]
        out = fusion.update(states, identities(), now_mono_ns=NOW)
    assert len(out["markers"][0]["trail"]) == 16
    assert fusion.update({}, identities(), now_mono_ns=NOW)["markers"] == []
    assert not fusion._trails


def test_stale_async_identity_revision_or_lease_cannot_restore_a_gid():
    fusion = engine()
    fusion.update(scene(), identities({}, revision=4), now_mono_ns=NOW)
    out = fusion.update(scene(), identities(revision=3), now_mono_ns=NOW)
    assert all(m["provisional_global_id"] is None for m in out["markers"])
    out = fusion.update(scene(), identities(revision=5, sampled=NOW - 751_000_000), now_mono_ns=NOW)
    assert all(m["provisional_global_id"] is None for m in out["markers"])


def test_actual_ttl_deadline_overrides_a_still_fresh_display_snapshot():
    mapping = identities(sampled=NOW - 50_000_000)
    for camera in mapping["assignments"].values():
        for binding in camera.values():
            binding["identity_expires_monotonic_ns"] = NOW - 1
    out = engine().update(scene(), mapping, now_mono_ns=NOW)
    assert all(m["provisional_global_id"] is None for m in out["markers"])


def test_local_mode_retains_both_original_positions_and_mode_switch_clears_trails():
    fusion = engine()
    fused = fusion.update(scene(), identities(), now_mono_ns=NOW)
    local = fusion.update(scene(), identities(), now_mono_ns=NOW, mode="local")
    assert len(fused["markers"]) == 1 and len(local["markers"]) == 2
    assert [m["layout_xy"] for m in local["markers"]] == [[200.0, 200.0], [220.0, 210.0]]


def test_actual_qt_marker_painter_synthetic_fusion_and_withdrawal(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "scripts/render_entrance_map_fusion_fixture.py",
                             "--output", str(tmp_path / "fusion.png")],
                            cwd=root, env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FUSION_AND_WITHDRAWAL_UI=PASS" in result.stdout
    assert (tmp_path / "fusion.png").stat().st_size > 1000


def test_real_memory_snapshot_api_expires_bindings_at_600_seconds_without_model_calls():
    from services.mv3dt_staging.provisional_reid_shadow import ProvisionalReIdShadow

    class NoInference:
        def embed_batch(self, _crops):
            raise AssertionError("lifecycle fixture must not perform model inference")

    clock = [NOW]
    shadow = ProvisionalReIdShadow(NoInference(), control_reference_max=0.645,
                                  control_reference_source="synthetic lifecycle fixture",
                                  person_class_id=0, clock_ns=lambda: clock[0])
    try:
        memory = shadow._identity_memory
        vector = np.eye(1, 512, dtype=np.float32)[0]
        galleries = {(camera, local): [
            {"vector": vector.copy(), "record": {"frame": frame,
             "source_id": sid, "source_pts_ns": frame * 50_000_000}}
            for frame in (1, 2)] for camera, local, sid in
            (("CAM-02", "4", 0), ("CAM-05", "0", 1))}
        match = {"camera_local_pair": {"CAM-02": "4", "CAM-05": "0"},
                 "cosine_similarity_median": 0.92, "cosine_similarity_min": 0.90,
                 "samples": {"CAM-02": 2, "CAM-05": 2}}
        for camera, local in (("CAM-02", "4"), ("CAM-05", "0")):
            memory.touch_observations(camera, [local], NOW)
        memory.reconcile([match], [match], galleries, {"CAM-02": {}, "CAM-05": {}},
                         control_reference_max=0.645, now_ns=NOW,
                         expected_revision=memory.revision)
        snapshot = shadow.display_identity_snapshot()
        assert snapshot["assignments"]["CAM-02"]["4"]["provisional_global_id"] == "G001"
        assert snapshot["sampled_monotonic_ns"] == NOW
        assert snapshot["revision"] == memory.revision
        clock[0] += 600_000_000_001
        expired = shadow.display_identity_snapshot()
        assert all(not rows for rows in expired["assignments"].values())
        assert expired["revision"] > snapshot["revision"]
    finally:
        shadow.close()
