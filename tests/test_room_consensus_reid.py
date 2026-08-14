from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from services.ml_service.core_v1.global_reid import GlobalIdentity, TrackletState, _normalize
from services.ml_service.core_v1.room_consensus_reid import RoomConsensusGlobalReIdCoordinator


class _FakeEmbedder:
    def metrics(self):
        return {"ready": True, "device": "cpu", "model_name": "fake", "last_error": ""}


class RoomConsensusReIdTests(unittest.TestCase):
    def _coordinator(self, **overrides):
        config = {
            "enabled": True,
            "min_samples": 3,
            "max_samples": 7,
            "room_embedding_weight": 0.90,
            "room_colour_weight": 0.10,
            "room_min_embedding_similarity": 0.66,
            "room_single_similarity": 0.71,
            "room_pair_similarity": 0.75,
            "room_pair_margin": 0.035,
            "room_confirmed_merge_similarity": 0.90,
            "same_camera_handoff_sec": 2.6,
            "same_camera_handoff_similarity": 0.84,
            "same_camera_handoff_distance": 0.85,
            "active_timeout_sec": 1.8,
            "match_similarity": 0.78,
            "same_group_similarity": 0.74,
            "same_camera_similarity": 0.76,
            "strong_similarity": 0.88,
            "second_best_margin": 0.055,
            "strong_second_best_margin": 0.025,
            "overlap_groups": [
                ["CAM-01", "CAM-04"],
                ["CAM-02", "CAM-05"],
                ["CAM-03", "CAM-06"],
            ],
        }
        config.update(overrides)
        return RoomConsensusGlobalReIdCoordinator(
            {}, {}, config, Path("."), embedder=_FakeEmbedder()
        )

    @staticmethod
    def _install_state(reid, camera, track_id, gid, vector, now, created, bbox):
        proto = _normalize(np.asarray(vector, dtype=np.float32))
        state = TrackletState(
            camera,
            track_id,
            embeddings=[proto],
            qualities=[0.9],
            last_seen=now,
            bbox=tuple(float(v) for v in bbox),
            global_id=gid,
            assignment_similarity=1.0,
            assignment_reason="provisional_embedding",
        )
        reid._tracks[(camera, track_id)] = state
        reid._globals[gid] = GlobalIdentity(
            gid,
            proto.copy(),
            created_at=created,
            last_seen=now,
            last_camera=camera,
        )
        reid._provisional_globals.add(gid)
        return state

    def test_cam03_one_vs_cam06_two_matches_only_correct_person(self):
        reid = self._coordinator()
        now = 100.0
        left = self._install_state(
            reid, "CAM-03", 30001, "G001", [1.0, 0.0, 0.0], now, 1.0, [50, 50, 100, 180]
        )
        right_same = self._install_state(
            reid, "CAM-06", 60001, "G002", [0.995, 0.06, 0.0], now, 2.0, [60, 40, 110, 175]
        )
        right_other = self._install_state(
            reid, "CAM-06", 60002, "G003", [0.0, 1.0, 0.0], now, 3.0, [200, 55, 255, 185]
        )
        reid._active_track_keys = {
            ("CAM-03", 30001),
            ("CAM-06", 60001),
            ("CAM-06", 60002),
        }

        reid._reconcile_overlap_pairs(now)

        self.assertEqual(reid._canonical_gid(left.global_id), reid._canonical_gid(right_same.global_id))
        self.assertNotEqual(reid._canonical_gid(left.global_id), reid._canonical_gid(right_other.global_id))
        self.assertEqual(reid.metrics()["room_pair_matches"], 1)

    def test_ambiguous_two_candidates_are_not_force_merged(self):
        reid = self._coordinator(room_pair_margin=0.06)
        now = 200.0
        left = self._install_state(
            reid, "CAM-03", 30001, "G001", [1.0, 0.0, 0.0], now, 1.0, [50, 50, 100, 180]
        )
        a = self._install_state(
            reid, "CAM-06", 60001, "G002", [0.98, 0.20, 0.0], now, 2.0, [60, 40, 110, 175]
        )
        b = self._install_state(
            reid, "CAM-06", 60002, "G003", [0.979, 0.205, 0.0], now, 3.0, [200, 55, 255, 185]
        )
        reid._active_track_keys = {
            ("CAM-03", 30001),
            ("CAM-06", 60001),
            ("CAM-06", 60002),
        }

        reid._reconcile_overlap_pairs(now)

        self.assertEqual(left.global_id, "G001")
        self.assertEqual(a.global_id, "G002")
        self.assertEqual(b.global_id, "G003")
        self.assertGreaterEqual(reid.metrics()["room_pair_ambiguous"], 1)

    def test_short_same_camera_fragment_hands_back_to_previous_gid(self):
        reid = self._coordinator()
        now = 300.0
        previous = self._install_state(
            reid, "CAM-04", 40010, "G010", [1.0, 0.0, 0.0], now - 0.8, 1.0, [100, 50, 150, 190]
        )
        current = self._install_state(
            reid, "CAM-04", 40011, "G011", [0.998, 0.05, 0.0], now, 2.0, [104, 52, 154, 192]
        )
        reid._active_track_keys = {("CAM-04", 40011)}

        reid._repair_same_camera_fragments(now)

        self.assertEqual(reid._canonical_gid(current.global_id), "G010")
        self.assertEqual(reid._canonical_gid(previous.global_id), "G010")
        self.assertEqual(reid.metrics()["same_camera_handoffs"], 1)

    def test_same_camera_active_people_never_share_gid(self):
        reid = self._coordinator()
        now = 400.0
        old = self._install_state(
            reid, "CAM-06", 60001, "G001", [1.0, 0.0], now - 0.5, 1.0, [100, 50, 150, 190]
        )
        current = self._install_state(
            reid, "CAM-06", 60002, "G002", [0.999, 0.03], now, 2.0, [103, 52, 153, 192]
        )
        reid._active_track_keys = {("CAM-06", 60001), ("CAM-06", 60002)}

        reid._repair_same_camera_fragments(now)

        self.assertNotEqual(reid._canonical_gid(old.global_id), reid._canonical_gid(current.global_id))


if __name__ == "__main__":
    unittest.main()
