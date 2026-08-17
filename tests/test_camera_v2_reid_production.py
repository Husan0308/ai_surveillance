import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.camera_v2.global_identity import STATE_CONFIRMED
from services.camera_v2.reid_production import ProductionGlobalIdentityCore


def vec(*values):
    row = np.asarray(values, dtype=np.float32)
    return row / np.linalg.norm(row)


A = vec(1.0, 0.0, 0.0, 0.0)
A2 = vec(0.98, 0.10, 0.0, 0.0)
A3 = vec(0.96, -0.12, 0.0, 0.0)
B = vec(0.0, 1.0, 0.0, 0.0)


class ProductionCoreTests(unittest.TestCase):
    def core(self):
        return ProductionGlobalIdentityCore({
            "camera_rooms": {"CAM-01": "Devs", "CAM-04": "Devs"},
            "min_samples": 3,
            "new_identity_confirm_samples": 4,
            "provisional_similarity": 0.62,
            "confirm_similarity": 0.72,
            "strong_similarity": 0.82,
            "reject_similarity": 0.56,
            "qwen_rescue_similarity": 0.58,
            "confirm_votes": 2,
            "prototype_update_similarity": 0.82,
            "reid_rollback_votes": 3,
        })

    def feed(self, core, camera, track_id, embeddings, start):
        result = None
        for index, embedding in enumerate(embeddings):
            result = core.observe_embedding(
                camera_id=camera,
                local_id=track_id,
                embedding=embedding,
                quality=0.9,
                captured_at=start + index * 0.25,
                room_id="Devs",
                bbox=(0, 0, 100, 250),
            )
        return result

    def test_low_conf_qwen_confirmation_does_not_poison_gallery(self):
        core = self.core()
        self.feed(core, "CAM-01", 1, [A, A2, A3, A], 1.0)
        gray = vec(0.65, 0.76, 0.0, 0.0)
        self.feed(core, "CAM-04", 2, [gray, gray, gray], 2.5)
        core.apply_qwen_result("CAM-04", 2, 1, "SAME", 0.97, now=4.0)
        origins = {row.origin_key for row in core._globals[1].gallery}
        self.assertNotIn(("CAM-04", 2), origins)

    def test_bad_merge_samples_cannot_self_validate(self):
        core = self.core()
        self.feed(core, "CAM-01", 1, [A, A2, A3, A], 10.0)
        self.feed(core, "CAM-04", 2, [A, A2, A3], 11.5)
        track = core._tracks[("CAM-04", 2)]
        track.assigned_score = 0.90
        for sample in track.samples:
            sample.embedding = B.copy()
        core._commit_track_to_gallery(track, 12.5)
        score = core._candidate_score(track, core._globals[1], 13.0)
        self.assertIsNotNone(score)
        self.assertLess(score.score, 0.56)

    def test_repeated_independent_reid_contradictions_rollback(self):
        core = self.core()
        self.feed(core, "CAM-01", 1, [A, A2, A3, A], 20.0)
        self.feed(core, "CAM-04", 2, [A, A2, A3, A], 21.5)
        track = core._tracks[("CAM-04", 2)]
        self.assertEqual(track.state, STATE_CONFIRMED)
        track.samples = []
        for index in range(6):
            core.observe_embedding(
                camera_id="CAM-04",
                local_id=2,
                embedding=B,
                quality=0.9,
                captured_at=24.0 + index * 0.25,
                room_id="Devs",
                bbox=(0, 0, 100, 250),
            )
        binding = core.binding_for_track("CAM-04", 2)
        self.assertTrue(binding is None or binding["global_id"] != 1)


if __name__ == "__main__":
    unittest.main()
