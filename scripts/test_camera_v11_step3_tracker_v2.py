#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v11.step3_tracker_v2 import V11PerCameraTrackerV2


class Step3TrackerV2Test(unittest.TestCase):
    def tracker(self):
        return V11PerCameraTrackerV2(["CAM-01", "CAM-02"])

    def test_stable_id_and_confirmation(self):
        tracker = self.tracker()
        first = tracker.update("CAM-01", [[100, 80, 220, 330, 0.85]], 1_000_000_000)
        second = tracker.update("CAM-01", [[108, 82, 228, 332, 0.82]], 1_500_000_000)
        self.assertEqual(first.created, 1)
        self.assertTrue(first.snapshots)
        self.assertTrue(second.snapshots)
        self.assertEqual(first.snapshots[0].track_id, second.snapshots[0].track_id)
        self.assertTrue(second.snapshots[0].confirmed)

    def test_low_confidence_does_not_create_track(self):
        tracker = self.tracker()
        update = tracker.update("CAM-01", [[100, 80, 220, 330, 0.22]], 1_000_000_000)
        self.assertEqual(update.created, 0)
        self.assertEqual(update.active, 0)

    def test_low_confidence_keeps_confirmed_track(self):
        tracker = self.tracker()
        tracker.update("CAM-01", [[100, 80, 220, 330, 0.90]], 1_000_000_000)
        confirmed = tracker.update("CAM-01", [[105, 80, 225, 330, 0.88]], 1_500_000_000)
        track_id = confirmed.snapshots[0].track_id
        low = tracker.update("CAM-01", [[110, 82, 230, 332, 0.24]], 2_000_000_000)
        self.assertTrue(low.snapshots)
        self.assertEqual(track_id, low.snapshots[0].track_id)
        self.assertGreaterEqual(low.matched_low, 1)
        self.assertEqual(low.created, 0)

    def test_recently_lost_track_recovers_on_low_confidence(self):
        tracker = self.tracker()
        tracker.update("CAM-01", [[100, 80, 220, 330, 0.90]], 1_000_000_000)
        confirmed = tracker.update("CAM-01", [[105, 80, 225, 330, 0.88]], 1_500_000_000)
        track_id = confirmed.snapshots[0].track_id
        tracker.update("CAM-01", [], 2_000_000_000)
        recovered = tracker.update("CAM-01", [[112, 82, 232, 332, 0.24]], 2_500_000_000)
        self.assertTrue(recovered.snapshots)
        self.assertEqual(track_id, recovered.snapshots[0].track_id)
        self.assertGreaterEqual(recovered.recovered, 1)
        self.assertGreaterEqual(recovered.matched_low, 1)
        self.assertEqual(recovered.created, 0)

    def test_last_observation_recovery_preserves_id_after_gap(self):
        tracker = self.tracker()
        tracker.update("CAM-01", [[100, 80, 220, 330, 0.90]], 1_000_000_000)
        confirmed = tracker.update("CAM-01", [[120, 80, 240, 330, 0.88]], 1_500_000_000)
        track_id = confirmed.snapshots[0].track_id
        tracker.update("CAM-01", [], 2_000_000_000)
        tracker.update("CAM-01", [], 2_500_000_000)
        recovered = tracker.update("CAM-01", [[126, 82, 246, 332, 0.80]], 3_000_000_000)
        self.assertTrue(recovered.snapshots)
        self.assertEqual(track_id, recovered.snapshots[0].track_id)
        self.assertGreaterEqual(recovered.recovered, 1)

    def test_distant_low_detection_cannot_teleport_lost_track(self):
        tracker = self.tracker()
        tracker.update("CAM-01", [[80, 70, 190, 330, 0.90]], 1_000_000_000)
        tracker.update("CAM-01", [[85, 70, 195, 330, 0.88]], 1_500_000_000)
        tracker.update("CAM-01", [], 2_000_000_000)
        far_low = tracker.update("CAM-01", [[500, 80, 620, 330, 0.24]], 2_500_000_000)
        self.assertEqual(far_low.recovered, 0)
        self.assertEqual(far_low.created, 0)

    def test_camera_namespaces_are_isolated(self):
        tracker = self.tracker()
        a = tracker.update("CAM-01", [[100, 80, 220, 330, 0.90]], 1_000_000_000)
        b = tracker.update("CAM-02", [[100, 80, 220, 330, 0.90]], 1_000_000_000)
        self.assertTrue(a.snapshots[0].track_id.startswith("CAM-01-T"))
        self.assertTrue(b.snapshots[0].track_id.startswith("CAM-02-T"))
        self.assertNotEqual(a.snapshots[0].track_id, b.snapshots[0].track_id)

    def test_invalid_rows_are_ignored(self):
        tracker = self.tracker()
        update = tracker.update(
            "CAM-01",
            [
                [1, 2, 3],
                [100, 100, 90, 200, 0.9],
                [100, 80, 220, 330, float("nan")],
            ],
            1_000_000_000,
        )
        self.assertEqual(update.detections, 0)
        self.assertEqual(update.created, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
