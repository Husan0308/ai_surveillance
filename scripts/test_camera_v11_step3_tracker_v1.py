#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v11.step3_tracker_v1 import V11PerCameraTrackerV1


class Step3TrackerV1Test(unittest.TestCase):
    def tracker(self):
        return V11PerCameraTrackerV1(["CAM01", "CAM02"])

    def test_stable_id_and_confirmation(self):
        tracker = self.tracker()
        first = tracker.update("CAM01", [[100, 80, 220, 330, 0.85]], 1_000_000_000)
        second = tracker.update("CAM01", [[108, 82, 228, 332, 0.82]], 1_500_000_000)
        self.assertEqual(first.created, 1)
        self.assertTrue(first.snapshots)
        self.assertTrue(second.snapshots)
        self.assertEqual(first.snapshots[0].track_id, second.snapshots[0].track_id)
        self.assertTrue(second.snapshots[0].confirmed)

    def test_low_confidence_does_not_create_track(self):
        tracker = self.tracker()
        update = tracker.update("CAM01", [[100, 80, 220, 330, 0.22]], 1_000_000_000)
        self.assertEqual(update.created, 0)
        self.assertEqual(update.active, 0)

    def test_low_confidence_can_recover_existing_track(self):
        tracker = self.tracker()
        tracker.update("CAM01", [[100, 80, 220, 330, 0.90]], 1_000_000_000)
        confirmed = tracker.update("CAM01", [[105, 80, 225, 330, 0.88]], 1_500_000_000)
        track_id = confirmed.snapshots[0].track_id
        tracker.update("CAM01", [], 2_000_000_000)
        recovered = tracker.update("CAM01", [[112, 82, 232, 332, 0.24]], 2_500_000_000)
        self.assertTrue(recovered.snapshots)
        self.assertEqual(track_id, recovered.snapshots[0].track_id)
        self.assertGreaterEqual(recovered.matched_low, 1)

    def test_camera_id_spaces_are_isolated(self):
        tracker = self.tracker()
        a = tracker.update("CAM01", [[100, 80, 220, 330, 0.90]], 1_000_000_000)
        b = tracker.update("CAM02", [[100, 80, 220, 330, 0.90]], 1_000_000_000)
        self.assertTrue(a.snapshots[0].track_id.startswith("CAM01-T"))
        self.assertTrue(b.snapshots[0].track_id.startswith("CAM02-T"))
        self.assertNotEqual(a.snapshots[0].track_id, b.snapshots[0].track_id)

    def test_invalid_rows_are_ignored(self):
        tracker = self.tracker()
        update = tracker.update(
            "CAM01",
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
