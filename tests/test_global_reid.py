from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from services.ml_service.core_v1.conservative_reid import ConservativeGlobalReIdCoordinator


class _FakeEmbedder:
    def metrics(self):
        return {
            "ready": True,
            "device": "cpu",
            "model_name": "fake-osnet",
            "last_error": "",
        }


class GlobalReIdTests(unittest.TestCase):
    def _coordinator(self, **overrides):
        config = {
            "enabled": True,
            "match_similarity": 0.74,
            "same_group_similarity": 0.69,
            "same_camera_similarity": 0.72,
            "strong_similarity": 0.86,
            "second_best_margin": 0.06,
            "strong_second_best_margin": 0.025,
            "prototype_update_similarity": 0.86,
            "active_timeout_sec": 1.6,
            "gallery_ttl_sec": 1000,
            "min_cross_group_transition_sec": 0.0,
            "overlap_groups": [
                ["CAM-01", "CAM-04"],
                ["CAM-02", "CAM-05"],
                ["CAM-03", "CAM-06"],
            ],
        }
        config.update(overrides)
        return ConservativeGlobalReIdCoordinator(
            {}, {}, config, Path("."), embedder=_FakeEmbedder()
        )

    def test_same_person_can_share_global_id_across_overlap_pair(self):
        reid = self._coordinator()
        vector = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        gid1 = reid.resolve_tracklet("CAM-01", 10001, vector, now=1.0)
        gid2 = reid.resolve_tracklet("CAM-04", 40001, vector, now=1.1)
        self.assertEqual(gid1, "G001")
        self.assertEqual(gid2, gid1)

    def test_same_global_id_cannot_bind_two_tracks_in_same_camera(self):
        reid = self._coordinator()
        vector = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        gid1 = reid.resolve_tracklet("CAM-01", 10001, vector, now=1.0)
        gid2 = reid.resolve_tracklet("CAM-01", 10002, vector, now=1.1)
        self.assertNotEqual(gid1, gid2)
        self.assertGreaterEqual(reid.metrics()["active_conflicts"], 1)

    def test_same_global_id_cannot_be_active_in_unrelated_rooms(self):
        reid = self._coordinator()
        vector = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        gid1 = reid.resolve_tracklet("CAM-01", 10001, vector, now=1.0)
        gid2 = reid.resolve_tracklet("CAM-02", 20001, vector, now=1.1)
        self.assertNotEqual(gid1, gid2)
        self.assertGreaterEqual(reid.metrics()["active_conflicts"], 1)

    def test_inactive_identity_can_be_reacquired_in_another_room(self):
        reid = self._coordinator(active_timeout_sec=1.0)
        vector = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        gid1 = reid.resolve_tracklet("CAM-01", 10001, vector, now=1.0)
        gid2 = reid.resolve_tracklet("CAM-02", 20001, vector, now=3.0)
        self.assertEqual(gid2, gid1)

    def test_strong_but_ambiguous_lookalike_match_creates_new_global_id(self):
        reid = self._coordinator(active_timeout_sec=0.3)

        # A and B are deliberately similar enough that a later query can score
        # ~0.95 against both. B is created while A is active in another room,
        # which forces a distinct gallery identity despite appearance similarity.
        a = np.asarray([1.0, 0.0], dtype=np.float32)
        b = np.asarray([0.8, 0.6], dtype=np.float32)
        gid_a = reid.resolve_tracklet("CAM-01", 10001, a, now=1.0)
        gid_b = reid.resolve_tracklet("CAM-02", 20001, b, now=1.1)
        self.assertNotEqual(gid_a, gid_b)

        # This vector is essentially equidistant from both identities. Absolute
        # similarity is strong, but the runner-up margin is near zero, so reuse
        # would be unsafe. Conservative policy must allocate G003.
        ambiguous = np.asarray([0.9486833, 0.3162278], dtype=np.float32)
        gid_c = reid.resolve_tracklet("CAM-03", 30001, ambiguous, now=4.0)
        self.assertNotIn(gid_c, {gid_a, gid_b})
        self.assertGreaterEqual(reid.metrics()["ambiguous_rejects"], 1)

    def test_identity_provider_exposes_global_id_after_resolution(self):
        reid = self._coordinator()
        vector = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        gid = reid.resolve_tracklet("CAM-01", 10001, vector, now=1.0)
        identity = reid.identity_for_track("CAM-01", 10001)
        self.assertEqual(identity["global_id"], gid)
        self.assertFalse(identity["known"])


if __name__ == "__main__":
    unittest.main()
