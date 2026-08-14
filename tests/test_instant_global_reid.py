from __future__ import annotations

from pathlib import Path
import time
import unittest

import numpy as np

from services.ml_service.core_v1.global_reid import TrackletState
from services.ml_service.core_v1.instant_reid_safe import SafeInstantGlobalReIdCoordinator


class _FakeEmbedder:
    def metrics(self):
        return {"ready": True, "device": "cpu", "model_name": "fake", "last_error": ""}


class InstantGlobalReIdTests(unittest.TestCase):
    def _coordinator(self, **overrides):
        config = {
            "enabled": True,
            "min_samples": 3,
            "confirm_min_samples": 3,
            "max_samples": 6,
            "fast_single_similarity": 0.79,
            "fast_pair_similarity": 0.84,
            "fast_pair_margin": 0.045,
            "fast_confirm_similarity": 0.73,
            "confirmed_pair_merge_similarity": 0.88,
            "match_similarity": 0.78,
            "same_group_similarity": 0.74,
            "same_camera_similarity": 0.76,
            "strong_similarity": 0.87,
            "second_best_margin": 0.06,
            "strong_second_best_margin": 0.025,
            "active_timeout_sec": 2.0,
            "min_cross_group_transition_sec": 0.0,
            "overlap_groups": [
                ["CAM-01", "CAM-04"],
                ["CAM-02", "CAM-05"],
                ["CAM-03", "CAM-06"],
            ],
        }
        config.update(overrides)
        return SafeInstantGlobalReIdCoordinator(
            {}, {}, config, Path("."), embedder=_FakeEmbedder()
        )

    def test_track_gets_visible_global_id_immediately(self):
        reid = self._coordinator()
        identity = reid.identity_for_track("CAM-03", 30001)
        self.assertEqual(identity["global_id"], "G001")
        self.assertTrue(identity["provisional"])
        self.assertEqual(identity["reid_reason"], "instant_provisional")
        self.assertEqual(
            reid.identity_for_track("CAM-03", 30001)["global_id"], "G001"
        )

    def test_single_person_overlap_pair_unifies_after_first_embedding(self):
        reid = self._coordinator()
        now = time.monotonic()
        first = reid.identity_for_track("CAM-03", 30001)["global_id"]
        second = reid.identity_for_track("CAM-06", 60001)["global_id"]
        self.assertNotEqual(first, second)

        state_a = reid._tracks[("CAM-03", 30001)]
        state_b = reid._tracks[("CAM-06", 60001)]
        reid._accept_embedding(
            state_a, np.asarray([1.0, 0.0, 0.0], dtype=np.float32), 0.9, now
        )
        reid._accept_embedding(
            state_b, np.asarray([0.999, 0.02, 0.0], dtype=np.float32), 0.9, now + 0.05
        )

        gid_a = reid.identity_for_track("CAM-03", 30001)["global_id"]
        gid_b = reid.identity_for_track("CAM-06", 60001)["global_id"]
        self.assertEqual(gid_a, gid_b)
        self.assertGreaterEqual(reid.metrics()["fast_pair_matches"], 1)
        self.assertGreaterEqual(reid.metrics()["single_person_fast_matches"], 1)

    def test_different_people_in_overlap_pair_keep_different_ids(self):
        reid = self._coordinator()
        now = time.monotonic()
        reid.identity_for_track("CAM-03", 30001)
        reid.identity_for_track("CAM-06", 60001)
        a = reid._tracks[("CAM-03", 30001)]
        b = reid._tracks[("CAM-06", 60001)]
        reid._accept_embedding(a, np.asarray([1.0, 0.0], dtype=np.float32), 0.9, now)
        reid._accept_embedding(b, np.asarray([0.0, 1.0], dtype=np.float32), 0.9, now + 0.05)
        self.assertNotEqual(
            reid.identity_for_track("CAM-03", 30001)["global_id"],
            reid.identity_for_track("CAM-06", 60001)["global_id"],
        )

    def test_same_camera_tracks_never_share_fast_global_id(self):
        reid = self._coordinator()
        now = time.monotonic()
        reid.identity_for_track("CAM-06", 60001)
        reid.identity_for_track("CAM-06", 60002)
        a = reid._tracks[("CAM-06", 60001)]
        b = reid._tracks[("CAM-06", 60002)]
        vector = np.asarray([1.0, 0.0], dtype=np.float32)
        reid._accept_embedding(a, vector, 0.9, now)
        reid._accept_embedding(b, vector, 0.9, now + 0.05)
        self.assertNotEqual(
            reid.identity_for_track("CAM-06", 60001)["global_id"],
            reid.identity_for_track("CAM-06", 60002)["global_id"],
        )

    def test_conflicting_camera_ownership_refuses_canonical_merge(self):
        reid = self._coordinator()
        now = time.monotonic()
        reid.identity_for_track("CAM-03", 30001)
        reid.identity_for_track("CAM-06", 60001)
        a = reid._tracks[("CAM-03", 30001)]
        b = reid._tracks[("CAM-06", 60001)]
        reid._accept_embedding(a, np.asarray([1.0, 0.0], dtype=np.float32), 0.9, now)
        reid._accept_embedding(b, np.asarray([0.0, 1.0], dtype=np.float32), 0.9, now + 0.05)
        gid_a = reid._canonical_gid(a.global_id)
        gid_b = reid._canonical_gid(b.global_id)
        self.assertNotEqual(gid_a, gid_b)

        # Simulate a second active CAM-03 track already owning gid_b. A merge of
        # gid_a and gid_b must be rejected because CAM-03 would then have two
        # people with the same Global ID.
        conflict = TrackletState("CAM-03", 30002, last_seen=now + 0.05, global_id=gid_b)
        reid._tracks[("CAM-03", 30002)] = conflict
        merged = reid._merge_global_ids(gid_a, gid_b, now + 0.06, 0.99)
        self.assertIsNone(merged)
        self.assertEqual(reid._canonical_gid(a.global_id), gid_a)
        self.assertEqual(reid._canonical_gid(b.global_id), gid_b)

    def test_metrics_expose_instant_and_pair_reconcile_counters(self):
        metrics = self._coordinator().metrics()
        for key in (
            "provisional_created",
            "fast_pair_matches",
            "single_person_fast_matches",
            "pair_reconciles",
            "canonical_merges",
            "fast_rollbacks",
        ):
            self.assertIn(key, metrics)
        self.assertFalse(metrics["gpu_used"])


if __name__ == "__main__":
    unittest.main()
