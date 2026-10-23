from __future__ import annotations

import time
import threading

import numpy as np
import pytest

from services.mv3dt_staging import provisional_reid_shadow as shadow_module
from services.mv3dt_staging.provisional_reid_shadow import (
    ProvisionalReIdShadow,
    ShadowJoinError,
    crop_quality_rejection,
    crop_quality_xywh,
    load_preserved_control_reference,
    measure_crop_quality,
)


class FakeEmbedder:
    def __init__(self, block: threading.Event | None = None):
        self.block = block
        self.calls = 0

    def embed_batch(self, crops):
        self.calls += 1
        if self.block is not None:
            self.block.wait(timeout=2)
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = 1.0
        return np.repeat(vector[None, :], len(crops), axis=0)


class InspectingEmbedder(FakeEmbedder):
    def __init__(self, block=None):
        super().__init__(block)
        self.crops = []

    def embed_batch(self, crops):
        self.crops.extend(crops)
        return super().embed_batch(crops)


def _frame():
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    yy, xx = np.indices((400, 200))
    pattern = (((xx // 8 + yy // 8) % 2) * 180 + 40).astype(np.uint8)
    for left in (300, 700, 1100):
        image[100:500, left:left + 200] = np.repeat(pattern[:, :, None], 3, axis=2)
    return image


def _submit(shadow, camera, frame, pts, track_id="7", confidence=0.8):
    left = {"7": 300.0, "8": 700.0, "9": 1100.0}.get(str(track_id), 300.0)
    return _submit_objects(shadow, camera, frame, pts, [{
        "class_id": 0,
        "tracker_id": track_id,
        "confidence": confidence,
        "left": left,
        "top": 100.0,
        "width": 200.0,
        "height": 400.0,
    }])


def _submit_objects(shadow, camera, frame, pts, objects):
    source_id = 0 if camera == "CAM-02" else 1
    return shadow.submit_frame(
        source_id=source_id,
        camera_id=camera,
        frame_index=frame,
        source_pts_ns=pts,
        cached_source_id=source_id,
        cached_frame_index=frame,
        cached_source_pts_ns=pts,
        inference_frame=_frame(),
        objects=objects,
    )


def _make_shadow(embedder=None, **kwargs):
    return ProvisionalReIdShadow(
        embedder or FakeEmbedder(),
        control_reference_max=0.64508455991745,
        control_reference_source="test-only saved control envelope; not labels",
        person_class_id=0,
        **kwargs,
    )


def test_live_crop_quality_matches_entrance_full_body_policy():
    assert crop_quality_xywh([300, 100, 200, 400])[0]
    assert not crop_quality_xywh([0, 100, 200, 400])[0]
    assert not crop_quality_xywh([300, 100, 20, 80])[0]
    assert not crop_quality_xywh([float("nan"), 100, 200, 400])[0]


def test_control_reference_is_hash_pinned_and_explicitly_unlabeled():
    maximum, source = load_preserved_control_reference()
    assert maximum == pytest.approx(0.64508455991745)
    assert "not ground-truth negatives" in source
    assert "sha256:" in source


def test_crop_quality_rejects_only_extreme_blur_or_overlap_proxy():
    crop = np.zeros((400, 200, 3), dtype=np.uint8)
    low_information = measure_crop_quality(crop, [300, 100, 200, 400])
    assert crop_quality_rejection(low_information) == "severely_blurred_or_low_information"
    textured = np.tile(np.arange(200, dtype=np.uint8)[None, :, None], (400, 1, 3))
    overlap = measure_crop_quality(
        textured, [300, 100, 200, 400], [[330, 100, 170, 400]],
        detector_confidence=0.9,
    )
    assert overlap["overlap_is_occlusion_ground_truth"] is False
    assert crop_quality_rejection(overlap) == "extreme_person_box_overlap_proxy"


def test_sampler_keeps_highest_quality_sample_in_each_temporal_window():
    shadow = _make_shadow(sample_window_frames=40, max_crops_per_track=4)
    try:
        _submit(shadow, "CAM-02", 0, 0, confidence=0.2)
        deadline = time.monotonic() + 2.0
        while shadow.snapshot()["counters"]["features_generated"] < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        _submit(shadow, "CAM-02", 20, 500_000_000, confidence=0.95)
        deadline = time.monotonic() + 2.0
        while shadow.snapshot()["counters"]["features_generated"] < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        # Once initial gallery coverage exists, later viewpoints still use the
        # pre-existing best-quality-per-window policy.
        _submit(shadow, "CAM-02", 40, 1_000_000_000, confidence=0.5)
        _submit(shadow, "CAM-02", 60, 1_500_000_000, confidence=0.95)
        _submit(shadow, "CAM-02", 80, 2_000_000_000, confidence=0.5)
        deadline = time.monotonic() + 2.0
        while shadow.snapshot()["counters"]["features_generated"] < 3 and time.monotonic() < deadline:
            time.sleep(0.005)
        samples = shadow.snapshot()["gallery_samples"]["CAM-02"]["7"]
        assert [row["frame"] for row in samples] == [0, 20, 60]
        assert shadow.snapshot()["counters"]["quality_ranked_window_replacements"] == 1
    finally:
        result = shadow.close()
    assert result["worker_alive"] is False


def test_gallery_ttl_is_ram_only_and_reappearance_after_expiry_starts_fresh():
    now = [0]
    shadow = _make_shadow(
        sample_window_frames=20,
        gallery_ttl_seconds=600.0,
        clock_ns=lambda: now[0],
    )
    try:
        # Initial crops are scheduled without waiting for the viewpoint window.
        _submit(shadow, "CAM-02", 0, 0)
        deadline = time.monotonic() + 2.0
        while shadow.snapshot()["counters"]["features_generated"] < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        _submit(shadow, "CAM-02", 20, 500_000_000)
        deadline = time.monotonic() + 2.0
        while shadow.snapshot()["counters"]["features_generated"] < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert shadow.snapshot()["gallery_memory"]["storage"] == "RAM_ONLY"
        assert shadow.snapshot()["gallery_memory"]["ttl_seconds"] == 600.0
        assert "7" in shadow.snapshot()["gallery_samples"]["CAM-02"]

        # Simulate an actual source-time disappearance/reappearance before
        # the 600-second wall-clock TTL expires. PTS gaps aid scheduling only;
        # TTL remains independent of source playback time.
        now[0] = 599_000_000_000
        _submit(shadow, "CAM-02", 40, 1_700_000_000)
        deadline = time.monotonic() + 2.0
        while shadow.snapshot()["counters"]["features_generated"] < 3 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert shadow.snapshot()["counters"]["reappeared_track_jobs_scheduled"] == 1
        assert shadow.expire_galleries(1_198_999_999_999) == []
        expired = shadow.expire_galleries(1_199_000_000_000)
        assert expired == [("CAM-02", "7")]
        assert "7" not in shadow.snapshot()["gallery_samples"]["CAM-02"]
        assert shadow.snapshot()["counters"]["gallery_ttl_expired_tracks"] == 1
        assert shadow.snapshot()["active_provisional_ids"] == []

        # Later source observations may establish a new RAM gallery, but TTL is
        # not evidence that an eventual reappearance was correctly recognized.
        features_before_reappearance = shadow.snapshot()["counters"]["features_generated"]
        now[0] = 1_200_000_000_000
        _submit(shadow, "CAM-02", 60, 2_000_000_000)
        deadline = time.monotonic() + 2.0
        while (shadow.snapshot()["counters"]["features_generated"] < features_before_reappearance + 1
               and time.monotonic() < deadline):
            time.sleep(0.005)
        samples = shadow.snapshot()["gallery_samples"]["CAM-02"]["7"]
        assert len(samples) == 1 and samples[0]["frame"] == 60
        assert shadow.snapshot()["gallery_memory"]["reappearance_recognition_accuracy"] == "UNVALIDATED"
    finally:
        result = shadow.close()
    assert result["worker_alive"] is False


def test_osnet_shadow_rejects_inexact_frame_or_pts_join():
    shadow = _make_shadow()
    try:
        with pytest.raises(ShadowJoinError, match="did not match"):
            shadow.submit_frame(
                source_id=0,
                camera_id="CAM-02",
                frame_index=10,
                source_pts_ns=500_000_000,
                cached_source_id=0,
                cached_frame_index=9,
                cached_source_pts_ns=450_000_000,
                inference_frame=_frame(),
                objects=[],
            )
        assert shadow.snapshot()["counters"]["features_generated"] == 0
    finally:
        result = shadow.close()
    assert result["worker_alive"] is False


def test_selected_crop_is_owned_and_keeps_exact_source_provenance():
    embedder = InspectingEmbedder()
    shadow = _make_shadow(embedder)
    source = _frame()
    expected = source[100:500, 300:500].copy()
    try:
        shadow.submit_frame(
            source_id=0, camera_id="CAM-02", frame_index=12, source_pts_ns=600_000_000,
            cached_source_id=0, cached_frame_index=12, cached_source_pts_ns=600_000_000,
            inference_frame=source,
            objects=[{"class_id": 0, "tracker_id": "7", "confidence": 0.8,
                      "left": 300.0, "top": 100.0, "width": 200.0, "height": 400.0}],
        )
        source[:] = 255  # prove the queued crop does not alias the frame raster
        deadline = time.monotonic() + 2.0
        while not embedder.crops and time.monotonic() < deadline:
            time.sleep(0.005)
        assert len(embedder.crops) == 1
        crop = embedder.crops[0]
        assert crop.flags.owndata and crop.base is None
        assert crop.shape == (400, 200, 3)
        np.testing.assert_array_equal(crop, expected)
        samples = shadow.snapshot()["gallery_samples"]["CAM-02"]["7"]
        assert samples[0]["frame"] == 12
        assert samples[0]["source_pts_ns"] == 600_000_000
    finally:
        result = shadow.close()
    assert result["worker_alive"] is False


def test_queued_job_over_one_second_expires_without_blocking_tracking():
    started = threading.Event()
    release = threading.Event()

    class FirstJobBlockedEmbedder(InspectingEmbedder):
        def embed_batch(self, crops):
            self.calls += 1
            self.crops.extend(crops)
            if self.calls == 1:
                started.set()
                release.wait(timeout=3.0)
            vector = np.zeros(512, dtype=np.float32)
            vector[0] = 1.0
            return np.repeat(vector[None, :], len(crops), axis=0)

    shadow = _make_shadow(FirstJobBlockedEmbedder(), queue_size=4)
    try:
        _submit(shadow, "CAM-02", 0, 0, track_id="7")
        assert started.wait(timeout=1.0)
        _submit(shadow, "CAM-02", 1, 50_000_000, track_id="8")
        time.sleep(1.2)
        release.set()
        deadline = time.monotonic() + 2.0
        while (shadow.snapshot()["counters"]["crop_jobs_expired_before_inference"] < 1
               and time.monotonic() < deadline):
            time.sleep(0.005)
        state = shadow.snapshot()
        assert state["counters"]["crop_jobs_expired_before_inference"] == 1
        assert state["counters"]["features_generated"] == 1
        assert "8" not in state["gallery_samples"]["CAM-02"]
        assert state["timing_ms"]["queue_wait"]["p95_ms"] >= 1000
        assert state["production_identity_write"] is False
    finally:
        release.set()
        result = shadow.close(timeout_sec=3)
    assert result["worker_alive"] is False


def test_inflight_embedding_cannot_resurrect_gallery_after_600s_ttl_expiry():
    now = [0]
    started = threading.Event()
    release = threading.Event()

    class BlockingEmbedder(InspectingEmbedder):
        def embed_batch(self, crops):
            started.set()
            self.crops.extend(crops)
            release.wait(timeout=3.0)
            vector = np.zeros(512, dtype=np.float32)
            vector[0] = 1.0
            return np.repeat(vector[None, :], len(crops), axis=0)

    shadow = _make_shadow(BlockingEmbedder(), gallery_ttl_seconds=600.0, clock_ns=lambda: now[0])
    try:
        _submit(shadow, "CAM-02", 0, 0)
        assert started.wait(timeout=1.0)
        now[0] = 600_000_000_000
        assert shadow.expire_galleries() == [("CAM-02", "7")]
        # Reappearance may queue a fresh generation while the stale forward
        # pass is still in flight; the stale completion must not clear it.
        _submit(shadow, "CAM-02", 20, 1_000_000_000)
        release.set()
        deadline = time.monotonic() + 2.0
        while (shadow.snapshot()["counters"]["embedding_jobs_completed"] < 2
               and time.monotonic() < deadline):
            time.sleep(0.005)
        state = shadow.snapshot()
        assert state["counters"]["stale_embedding_results_rejected"] == 1
        assert state["counters"]["features_generated"] == 1
        samples = state["gallery_samples"]["CAM-02"]["7"]
        assert len(samples) == 1 and samples[0]["frame"] == 20
    finally:
        release.set()
        result = shadow.close(timeout_sec=3)
    assert result["worker_alive"] is False


def test_close_releases_gallery_vectors_and_is_idempotent():
    shadow = _make_shadow()
    _submit(shadow, "CAM-02", 0, 0)
    result = shadow.close()
    assert result["shutdown_memory_release"]["gallery_vectors"] >= 1
    assert result["shutdown_memory_release"]["gallery_vectors"] <= 1
    assert shadow.snapshot()["shutdown_memory_release"] == result["shutdown_memory_release"]
    assert shadow.close() == result


def test_osnet_shadow_queue_is_bounded_and_drops_only_reid_samples():
    release = threading.Event()
    shadow = _make_shadow(FakeEmbedder(block=release), queue_size=1, sample_window_frames=40)
    try:
        queued = _submit_objects(shadow, "CAM-02", 0, 0, [
            {"class_id": 0, "tracker_id": str(track_id), "confidence": 0.8,
             "left": left, "top": 100.0, "width": 200.0, "height": 400.0}
            for track_id, left in ((7, 300.0), (8, 700.0), (9, 1100.0))
        ])
        assert 1 <= queued["queued_crops"] <= 2
        deadline = time.monotonic() + 1.0
        while not shadow.embedder.calls and time.monotonic() < deadline:
            time.sleep(0.005)
        state = shadow.snapshot()
        assert state["queue_depth"] <= 1
        assert state["max_queue_depth_observed"] == 1
        assert state["counters"]["queue_full_drops"] >= 1
        assert state["counters"]["tracker_observations_seen"] == 3
        assert state["production_identity_write"] is False
    finally:
        release.set()
        result = shadow.close(timeout_sec=2)
    assert result["worker_alive"] is False
    assert result["shutdown_memory_release"]["gallery_vectors"] >= 1
    assert shadow._vectors == [] and shadow._gallery == {}


def test_stale_embedding_age_is_measured_separately_from_queue_wait(monkeypatch):
    monkeypatch.setattr(shadow_module, "STALE_EMBEDDING_AGE_SECONDS", 0.01)
    release = threading.Event()
    shadow = _make_shadow(FakeEmbedder(block=release), sample_window_frames=20)
    try:
        _submit(shadow, "CAM-02", 0, 0)
        _submit(shadow, "CAM-02", 20, 1_000_000_000)
        deadline = time.monotonic() + 1.0
        while not shadow.embedder.calls and time.monotonic() < deadline:
            time.sleep(0.005)
        time.sleep(0.03)
        release.set()
        deadline = time.monotonic() + 2.0
        while shadow.snapshot()["counters"]["features_generated"] < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        state = shadow.snapshot()
        assert state["counters"]["embedding_jobs_completed"] >= 1
        assert state["counters"]["embedding_jobs_over_stale_age_threshold"] >= 1
        assert state["timing_ms"]["queue_wait"]["count"] >= 1
        assert state["timing_ms"]["source_observation_to_embedding_completion"]["count"] >= 1
    finally:
        release.set()
        result = shadow.close(timeout_sec=2)
    assert result["worker_alive"] is False


def test_shadow_candidate_keeps_camera_local_namespaces_and_is_nonpersistent(monkeypatch):
    def fake_score(vectors, records, tracks, **kwargs):
        matches = [] if len(records) < 4 else [{
            "camera_local_pair": {"CAM-02": "7", "CAM-05": "17"},
            "cosine_similarity_median": 0.82,
            "cosine_similarity_min": 0.78,
            "samples": {"CAM-02": 2, "CAM-05": 2},
        }]
        return {
            "status": "PROVISIONAL_MATCHES_ASSIGNED" if matches else "WAITING",
            "matches": matches,
            "candidate_pairs": [],
        }

    monkeypatch.setattr(shadow_module, "score_and_assign_provisional_ids", fake_score)
    shadow = _make_shadow(sample_window_frames=20)
    try:
        _submit(shadow, "CAM-02", 0, 10)
        _submit(shadow, "CAM-05", 0, 7_000_000_000, track_id="17")
        _submit(shadow, "CAM-02", 20, 1_000_000_010)
        _submit(shadow, "CAM-05", 20, 7_500_000_000, track_id="17")
        _submit(shadow, "CAM-02", 40, 2_000_000_010)
        _submit(shadow, "CAM-05", 40, 8_000_000_000, track_id="17")
        deadline = time.monotonic() + 2.0
        while shadow.snapshot()["counters"]["features_generated"] < 4 and time.monotonic() < deadline:
            time.sleep(0.005)
        mapping = shadow.assignment_map()
        assert mapping["CAM-02"]["7"]["provisional_global_id"] == "G001"
        assert mapping["CAM-05"]["17"]["provisional_global_id"] == "G001"
        assert mapping["CAM-02"]["7"]["display_label"] == "G001?"
        assert mapping["CAM-05"].get("7") is None
        state = shadow.snapshot()
        assert state["mode"] == "IN_MEMORY_NONPERSISTENT_SHADOW_ONLY"
        assert state["cross_camera_pts_used_as_sync_gate"] is False
        assert state["production_identity_write"] is False
        assert state["raw_tracker_records_modified"] is False
        assert len(state["revision_events"]) == 1
    finally:
        result = shadow.close()
    assert result["worker_alive"] is False


def test_opt_in_decision_trace_explains_exact_gate_that_withdraws_existing_gid():
    shadow = _make_shadow(decision_diagnostics=True)
    previous = {
        "CAM-02": {"4": {"provisional_global_id": "G001"}},
        "CAM-05": {"0": {"provisional_global_id": "G001"}},
    }
    trace = {
        "camera_local_pair": {"CAM-02": "4", "CAM-05": "0"},
        "source_observations": {
            "CAM-02": {"camera_id": "CAM-02", "source_id": 0,
                       "local_track_id": "4", "latest_frame": 121,
                       "latest_source_pts_ns": 6_050_000_000,
                       "gallery_sample_count": 3,
                       "freshest_sample_age_ms_monotonic": 12.0},
            "CAM-05": {"camera_id": "CAM-05", "source_id": 1,
                       "local_track_id": "0", "latest_frame": 122,
                       "latest_source_pts_ns": 6_100_000_000,
                       "gallery_sample_count": 2,
                       "freshest_sample_age_ms_monotonic": 18.0},
        },
        "candidate_global_id": "G001",
        "best_similarity_score": 0.61,
        "competing_similarity_score_by_camera": {"CAM-02": 0.59, "CAM-05": 0.58},
        "competing_track_pair_by_camera": {
            "CAM-02": {"CAM-02": "4", "CAM-05": "15"},
            "CAM-05": {"CAM-02": "6", "CAM-05": "0"},
        },
        "reciprocal_match": {"CAM-02": False, "CAM-05": True},
        "margin_result": {"CAM-02": {"margin": 0.02, "passed": False},
                          "CAM-05": {"margin": 0.03, "passed": True}},
        "gates": {"minimum_gallery_size": True, "reciprocal_best_match": False,
                  "positive_margin_exceeds_twice_observed_gallery_dispersion": False},
        "decision": "REJECTED_OR_AMBIGUOUS",
        "reason": "failed existing gate(s): reciprocal_best_match, positive_margin_exceeds_twice_observed_gallery_dispersion",
        "failed_gates": ["reciprocal_best_match",
                         "positive_margin_exceeds_twice_observed_gallery_dispersion"],
    }
    try:
        shadow._record_decision_diagnostics(
            {"status": "NO_MATCH_PASSED_CONSERVATIVE_GATES",
             "decision_trace": [trace]},
            previous, {"CAM-02": {}, "CAM-05": {}},
            {("4", "0"): {"provisional_global_id": "G001"}}, {},
            {"abstained": []}, 9_000_000_000,
        )
        state = shadow.snapshot()
        withdrawal = next(row for row in state["decision_diagnostic_events"]
                          if row["event"] == "GLOBAL_ID_WITHDRAWAL")
        assert withdrawal["candidate_global_id"] == "G001"
        assert withdrawal["camera_local_pair"] == {"CAM-02": "4", "CAM-05": "0"}
        assert withdrawal["competing_track_pair_by_camera"]["CAM-02"]["CAM-05"] == "15"
        assert "reciprocal_best_match" in withdrawal["reason"]
        assert "positive_margin_exceeds_twice_observed_gallery_dispersion" in withdrawal["failed_gates"]
        assert withdrawal["source_observations"]["CAM-05"]["latest_source_pts_ns"] == 6_100_000_000
        assert state["decision_diagnostic_event_count"] == 2
    finally:
        result = shadow.close()
    assert result["worker_alive"] is False


def _install_gate_approved_pair_fixture(monkeypatch):
    """Stub only the scorer result so these tests cover memory/ID lifecycle."""
    def score_fixture(vectors, records, tracks, **kwargs):
        counts = {}
        for row in records:
            key = (str(row["camera_id"]), str(row["local_track_id"]))
            counts[key] = counts.get(key, 0) + 1
        pair = {"CAM-02": "7", "CAM-05": "17"}
        ready = counts.get(("CAM-02", "7"), 0) >= 2 and counts.get(("CAM-05", "17"), 0) >= 2
        matches = ([{
            "camera_local_pair": pair,
            "cosine_similarity_median": 0.91,
            "cosine_similarity_min": 0.88,
            "samples": {"CAM-02": counts[("CAM-02", "7")],
                        "CAM-05": counts[("CAM-05", "17")]},
            "automatic_candidate_gates": {"fixture_gate_result": "PASS"},
        }] if ready else [])
        return {"status": "PROVISIONAL_MATCHES_ASSIGNED" if ready else "WAITING",
                "matches": matches, "candidate_pairs": []}

    monkeypatch.setattr(shadow_module, "score_and_assign_provisional_ids", score_fixture)


def _wait_for_features(shadow, count, timeout=2.0):
    deadline = time.monotonic() + timeout
    while shadow.snapshot()["counters"]["features_generated"] < count and time.monotonic() < deadline:
        time.sleep(0.005)
    return shadow.snapshot()


def _seed_synthetic_provisional_pair(shadow, now):
    now[0] = 0
    _submit(shadow, "CAM-02", 0, 0, track_id="7")
    _submit(shadow, "CAM-05", 0, 7_000_000_000, track_id="17")
    _wait_for_features(shadow, 2)
    # Provide continuous observations between crop samples. Otherwise a
    # 20-frame sampling interval looks like a one-second disappearance.
    for frame in range(1, 21):
        now[0] = frame * 50_000_000
        _submit(shadow, "CAM-02", frame, frame * 50_000_000, track_id="7")
        _submit(shadow, "CAM-05", frame, 7_000_000_000 + frame * 50_000_000,
                track_id="17")
    state = _wait_for_features(shadow, 4)
    mapping = shadow.assignment_map()
    assert mapping["CAM-02"]["7"]["provisional_global_id"] == "G001"
    assert mapping["CAM-05"]["17"]["provisional_global_id"] == "G001"
    return state


@pytest.mark.parametrize("gap_seconds", [30, 120, 300, 540])
def test_return_before_ttl_keeps_gallery_and_same_pair_provisional_id(monkeypatch, gap_seconds):
    """Synthetic lifecycle only: same local IDs and a gate-approved scorer fixture."""
    _install_gate_approved_pair_fixture(monkeypatch)
    now = [0]
    shadow = _make_shadow(
        sample_window_frames=20, gallery_ttl_seconds=600.0,
        clock_ns=lambda: now[0],
    )
    try:
        _seed_synthetic_provisional_pair(shadow, now)
        baseline_reappearances = shadow.snapshot()["counters"]["reappeared_track_jobs_scheduled"]
        now[0] = (gap_seconds + 1) * 1_000_000_000
        frame = 20 + gap_seconds * 20
        cam02_pts = 1_000_000_000 + gap_seconds * 1_000_000_000
        cam05_pts = 8_000_000_000 + gap_seconds * 1_000_000_000
        _submit(shadow, "CAM-02", frame, cam02_pts, track_id="7")
        _submit(shadow, "CAM-05", frame, cam05_pts, track_id="17")
        state = _wait_for_features(shadow, 6)
        assignments = shadow.assignment_map()

        assert state["counters"]["gallery_ttl_expired_tracks"] == 0
        assert (state["counters"]["reappeared_track_jobs_scheduled"]
                - baseline_reappearances) == 2
        assert assignments["CAM-02"]["7"]["provisional_global_id"] == "G001"
        assert assignments["CAM-05"]["17"]["provisional_global_id"] == "G001"
        event_types = [event["event"] for event in state["revision_events"]]
        assert event_types[0] == "CANDIDATE_CREATED"
        assert event_types.count("IDENTITY_STATE_CHANGED") == 2
        assert any(event.get("from_state") == "LOST" and event.get("to_state") == "ACTIVE"
                   for event in state["revision_events"])
        assert event_types.count("CANDIDATE_CREATED") == 1
        assert state["gallery_memory"]["clock_domain"].startswith("wall_monotonic_seconds")
    finally:
        result = shadow.close()
    assert result["worker_alive"] is False


def test_return_after_ttl_expires_old_gallery_and_does_not_reuse_old_id(monkeypatch):
    """A post-TTL return creates a fresh gallery; G001 is withdrawn, not revived."""
    _install_gate_approved_pair_fixture(monkeypatch)
    now = [0]
    shadow = _make_shadow(
        sample_window_frames=20, gallery_ttl_seconds=600.0,
        clock_ns=lambda: now[0],
    )
    try:
        _seed_synthetic_provisional_pair(shadow, now)
        now[0] = 602_000_000_000  # 601 seconds after the final continuous observation
        frame = 20 + 601 * 20
        cam02_pts = 1_000_000_000 + 601_000_000_000
        cam05_pts = 8_000_000_000 + 601_000_000_000
        _submit(shadow, "CAM-02", frame, cam02_pts, track_id="7")
        _submit(shadow, "CAM-05", frame, cam05_pts, track_id="17")
        state = _wait_for_features(shadow, 6)
        assert state["counters"]["gallery_ttl_expired_tracks"] == 2
        assert state["active_provisional_ids"] == []
        assert [event["event"] for event in state["revision_events"]][-1] == "CANDIDATE_WITHDRAWN"

        now[0] += 50_000_000
        _submit(shadow, "CAM-02", frame + 20, cam02_pts + 1_000_000_000, track_id="7")
        _submit(shadow, "CAM-05", frame + 20, cam05_pts + 1_000_000_000, track_id="17")
        state = _wait_for_features(shadow, 8)
        assignments = shadow.assignment_map()
        assert assignments["CAM-02"]["7"]["provisional_global_id"] == "G002"
        assert assignments["CAM-05"]["17"]["provisional_global_id"] == "G002"
        assert [event["provisional_global_id"] for event in state["revision_events"]
                if event["event"] == "CANDIDATE_WITHDRAWN"] == ["G001"]
        assert state["gallery_samples"]["CAM-02"]["7"][0]["frame"] == frame
    finally:
        result = shadow.close()
    assert result["worker_alive"] is False


def test_replay_pts_gap_does_not_expire_wall_clock_gallery(monkeypatch):
    """Accelerated source-time gaps do not masquerade as elapsed TTL wall time."""
    _install_gate_approved_pair_fixture(monkeypatch)
    now = [0]
    shadow = _make_shadow(sample_window_frames=20, clock_ns=lambda: now[0])
    try:
        _seed_synthetic_provisional_pair(shadow, now)
        baseline_reappearances = shadow.snapshot()["counters"]["reappeared_track_jobs_scheduled"]
        now[0] = 3_000_000_000  # only two wall-clock seconds elapsed after the last observation
        frame = 20 + 540 * 20
        _submit(shadow, "CAM-02", frame, 541_000_000_000, track_id="7")
        state = _wait_for_features(shadow, 5)
        assert state["counters"]["gallery_ttl_expired_tracks"] == 0
        assert (state["counters"]["reappeared_track_jobs_scheduled"]
                - baseline_reappearances) == 1
        assert shadow.assignment_map()["CAM-02"]["7"]["provisional_global_id"] == "G001"
    finally:
        result = shadow.close()
    assert result["worker_alive"] is False


def test_one_to_many_candidate_conflict_is_rejected(monkeypatch):
    def fake_score(vectors, records, tracks, **kwargs):
        pairs = [
            {"camera_local_pair": {"CAM-02": "7", "CAM-05": "17"},
             "cosine_similarity_median": 0.82, "cosine_similarity_min": 0.80,
             "samples": {"CAM-02": 2, "CAM-05": 2}},
            {"camera_local_pair": {"CAM-02": "7", "CAM-05": "18"},
             "cosine_similarity_median": 0.81, "cosine_similarity_min": 0.80,
             "samples": {"CAM-02": 2, "CAM-05": 2}},
        ]
        matches = pairs if len(records) >= 4 else []
        return {"status": "PROVISIONAL_MATCHES_ASSIGNED" if matches else "WAITING",
                "matches": matches, "candidate_pairs": []}

    monkeypatch.setattr(shadow_module, "score_and_assign_provisional_ids", fake_score)
    shadow = _make_shadow(sample_window_frames=20)
    try:
        for frame, pts in ((0, 10), (20, 1_000_000_010)):
            _submit(shadow, "CAM-02", frame, pts)
            _submit(shadow, "CAM-05", frame, pts + 7_000_000_000, track_id="17")
            _submit(shadow, "CAM-05", frame + 1, pts + 7_000_000_100, track_id="18")
        _submit(shadow, "CAM-02", 40, 2_000_000_010)
        _submit(shadow, "CAM-05", 40, 9_000_000_010, track_id="17")
        deadline = time.monotonic() + 2.0
        while shadow.snapshot()["counters"]["features_generated"] < 4 and time.monotonic() < deadline:
            time.sleep(0.005)
        state = shadow.snapshot()
        assert len(state["active_provisional_ids"]) == 1
        assert state["counters"]["one_to_many_conflicts_rejected"] >= 1
        ids = [row["provisional_global_id"] for row in state["active_provisional_ids"]]
        assert ids == ["G001"]
    finally:
        result = shadow.close()
    assert result["worker_alive"] is False
