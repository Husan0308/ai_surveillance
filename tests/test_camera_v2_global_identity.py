import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.camera_v2.global_identity import GlobalIdentityCore, STATE_CONFIRMED


def vec(*values):
    a = np.asarray(values, dtype=np.float32)
    return a / np.linalg.norm(a)


A = vec(1.0, 0.0, 0.0, 0.0)
A2 = vec(0.98, 0.10, 0.0, 0.0)
A3 = vec(0.96, -0.12, 0.0, 0.0)
B = vec(0.0, 1.0, 0.0, 0.0)


class GlobalIdentityTests(unittest.TestCase):
    def core(self):
        return GlobalIdentityCore({
            "camera_rooms": {
                "CAM-01": "Devs", "CAM-04": "Devs",
                "CAM-02": "Entrance", "CAM-05": "Entrance",
                "CAM-03": "Main Rooms", "CAM-06": "Main Rooms",
            },
            "min_samples": 3,
            "new_identity_confirm_samples": 4,
            "confirm_votes": 2,
            "provisional_similarity": 0.62,
            "confirm_similarity": 0.72,
            "strong_similarity": 0.82,
            "min_margin": 0.04,
            "strong_margin": 0.02,
        })

    def feed(self, core, camera, tid, vectors, start, room=None):
        out = None
        for i, v in enumerate(vectors):
            out = core.observe_embedding(
                camera_id=camera, local_id=tid, embedding=v,
                quality=0.9, captured_at=start + i * 0.25,
                room_id=room, bbox=(100, 100, 200, 350),
            )
        return out

    def test_new_identity_confirms_after_multishot(self):
        core = self.core()
        self.feed(core, "CAM-01", 10, [A, A2, A3, A], 10.0, "Devs")
        b = core.binding_for_track("CAM-01", 10)
        self.assertIsNotNone(b)
        self.assertEqual(b["global_id"], 1)
        self.assertEqual(b["state"], STATE_CONFIRMED)

    def test_second_camera_same_room_reuses_global(self):
        core = self.core()
        self.feed(core, "CAM-01", 10, [A, A2, A3, A], 10.0, "Devs")
        self.feed(core, "CAM-04", 99, [A3, A2, A], 11.2, "Devs")
        b = core.binding_for_track("CAM-04", 99)
        self.assertEqual(b["global_id"], 1)
        core.observe_embedding(
            camera_id="CAM-04", local_id=99, embedding=A2, quality=.9,
            captured_at=12.0, room_id="Devs", bbox=(80, 90, 190, 355),
        )
        b = core.binding_for_track("CAM-04", 99)
        self.assertEqual(b["global_id"], 1)

    def test_same_camera_fragment_reconnects(self):
        core = self.core()
        self.feed(core, "CAM-02", 7, [A, A2, A3, A], 20.0, "Entrance")
        core.maintenance(23.0)
        self.feed(core, "CAM-02", 8, [A3, A2, A], 23.1, "Entrance")
        b = core.binding_for_track("CAM-02", 8)
        self.assertEqual(b["global_id"], 1)

    def test_same_camera_two_active_tracks_cannot_share(self):
        core = self.core()
        self.feed(core, "CAM-01", 1, [A, A2, A3, A], 30.0, "Devs")
        self.feed(core, "CAM-01", 2, [A, A2, A3], 30.7, "Devs")
        b2 = core.binding_for_track("CAM-01", 2)
        self.assertTrue(b2 is None or b2["global_id"] != 1)

    def test_different_room_simultaneous_is_hard_conflict(self):
        core = self.core()
        self.feed(core, "CAM-01", 1, [A, A2, A3, A], 40.0, "Devs")
        self.feed(core, "CAM-03", 5, [A, A2, A3], 40.8, "Main Rooms")
        b = core.binding_for_track("CAM-03", 5)
        self.assertTrue(b is None or b["global_id"] != 1)

    def test_qwen_different_rejects_tentative(self):
        core = self.core()
        self.feed(core, "CAM-01", 1, [A, A2, A3, A], 50.0, "Devs")
        self.feed(core, "CAM-04", 2, [A, A2, A3], 51.2, "Devs")
        b = core.binding_for_track("CAM-04", 2)
        self.assertEqual(b["global_id"], 1)
        result = core.apply_qwen_result("CAM-04", 2, 1, "DIFFERENT", .95, now=52.0)
        self.assertEqual(result["action"], "qwen_reject")
        self.assertIsNone(core.binding_for_track("CAM-04", 2))

    def test_qwen_same_confirms_tentative(self):
        core = self.core()
        self.feed(core, "CAM-01", 1, [A, A2, A3, A], 60.0, "Devs")
        self.feed(core, "CAM-04", 2, [A, A2, A3], 61.2, "Devs")
        b = core.binding_for_track("CAM-04", 2)
        self.assertEqual(b["global_id"], 1)
        result = core.apply_qwen_result("CAM-04", 2, 1, "SAME", .92, now=62.0)
        self.assertEqual(result["action"], "qwen_confirm")
        self.assertEqual(core.binding_for_track("CAM-04", 2)["state"], STATE_CONFIRMED)

    def test_lookalike_margin_prevents_blind_merge(self):
        core = self.core()
        self.feed(core, "CAM-01", 1, [A, A2, A3, A], 70.0, "Devs")
        core.maintenance(75.0)
        self.feed(core, "CAM-02", 2, [B, B, B, B], 75.1, "Entrance")
        core.maintenance(80.0)
        q = vec(0.707, 0.707, 0, 0)
        self.feed(core, "CAM-05", 3, [q, q, q], 80.1, "Entrance")
        b = core.binding_for_track("CAM-05", 3)
        if b is not None:
            self.assertNotEqual(b["global_id"], 1)

    def test_snapshot_missing_allows_fast_same_camera_reconnect(self):
        core = self.core()
        self.feed(core, "CAM-02", 7, [A, A2, A3, A], 90.0, "Entrance")
        core.observe_camera_snapshot("CAM-02", [], seen_at=91.6)
        self.feed(core, "CAM-02", 8, [A3, A2, A], 91.7, "Entrance")
        b = core.binding_for_track("CAM-02", 8)
        self.assertIsNotNone(b)
        self.assertEqual(b["global_id"], 1)

    def test_qwen_can_rescue_gray_zone_candidate(self):
        core = self.core()
        self.feed(core, "CAM-01", 1, [A, A2, A3, A], 100.0, "Devs")
        q = vec(0.82, 0.30, 0.0, 0.0)
        self.feed(core, "CAM-04", 2, [q, q, q], 101.2, "Devs")
        b = core.binding_for_track("CAM-04", 2)
        if b is None:
            result = core.apply_qwen_result("CAM-04", 2, 1, "SAME", .96, now=102.0)
            self.assertIn(result["action"], {"keep", "qwen_confirm"})
            b = core.binding_for_track("CAM-04", 2)
        self.assertIsNotNone(b)
        self.assertEqual(b["global_id"], 1)


if __name__ == "__main__":
    unittest.main()
