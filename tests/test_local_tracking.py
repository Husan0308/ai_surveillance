from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from services.ml_service.core_v1.local_tracker import LocalByteTracker, linear_sum_assignment
from services.ml_service.core_v1.visual_tracker import VisualBox


class LocalTrackingTests(unittest.TestCase):
    def test_hungarian_avoids_greedy_global_assignment_trap(self):
        # Greedy would take (row0,col0)=1.0 first and force row1->col1=100.
        # The global optimum is row0->col1=2.0 and row1->col0=1.1.
        costs = np.asarray([[1.0, 2.0], [1.1, 100.0]], dtype=np.float64)
        rows, cols = linear_sum_assignment(costs)
        pairs = set(zip(rows.tolist(), cols.tolist()))
        self.assertEqual(pairs, {(0, 1), (1, 0)})
        self.assertAlmostEqual(sum(costs[r, c] for r, c in pairs), 3.1, places=6)

    def test_low_confidence_detection_continues_existing_id(self):
        tracker = LocalByteTracker(
            strong_confirm_hits=1,
            weak_confirm_hits=2,
            start_conf=0.25,
            new_track_min_conf=0.25,
            byte_high_conf=0.25,
            byte_low_conf=0.08,
            low_conf_confirm=0.08,
            hold_ms=800,
            memory_ms=2500,
            prediction_ms=320,
        )
        first = SimpleNamespace(
            frame_id=1,
            frame_captured_monotonic=1.0,
            boxes=(VisualBox(100, 100, 180, 300, 0.80),),
        )
        tracker.update(first, now=1.0, source_width=640, source_height=360)
        visible1 = tracker.visible(1.0, target_time=1.0)
        self.assertEqual(len(visible1), 1)
        track_id = visible1[0].track_id
        self.assertGreater(track_id, 0)

        second = SimpleNamespace(
            frame_id=2,
            frame_captured_monotonic=1.10,
            boxes=(VisualBox(106, 100, 186, 300, 0.10),),
        )
        tracker.update(second, now=1.10, source_width=640, source_height=360)
        visible2 = tracker.visible(1.10, target_time=1.10)
        self.assertEqual(len(visible2), 1)
        self.assertEqual(visible2[0].track_id, track_id)
        self.assertGreaterEqual(tracker.metrics()["low_matches"], 1)

    def test_low_score_box_cannot_birth_new_track(self):
        tracker = LocalByteTracker(
            strong_confirm_hits=1,
            weak_confirm_hits=2,
            start_conf=0.25,
            new_track_min_conf=0.25,
            byte_high_conf=0.25,
            byte_low_conf=0.08,
            low_conf_confirm=0.08,
        )
        result = SimpleNamespace(
            frame_id=1,
            frame_captured_monotonic=2.0,
            boxes=(VisualBox(50, 50, 110, 220, 0.12),),
        )
        tracker.update(result, now=2.0, source_width=640, source_height=360)
        self.assertEqual(tracker.visible(2.0, target_time=2.0), [])
        self.assertEqual(tracker.metrics()["births"], 0)

    def test_snapshot_exposes_stable_local_id(self):
        tracker = LocalByteTracker(
            strong_confirm_hits=1,
            start_conf=0.25,
            new_track_min_conf=0.25,
            byte_high_conf=0.25,
            byte_low_conf=0.08,
            low_conf_confirm=0.08,
        )
        result = SimpleNamespace(
            frame_id=1,
            frame_captured_monotonic=3.0,
            boxes=(VisualBox(10, 20, 80, 200, 0.9),),
        )
        tracker.update(result, now=3.0, source_width=640, source_height=360)
        tracker.visible(3.0, target_time=3.0)
        snapshot = tracker.snapshot()
        self.assertEqual(len(snapshot), 1)
        self.assertGreater(snapshot[0]["track_id"], 0)
        self.assertEqual(len(snapshot[0]["bbox"]), 4)
        self.assertEqual(tracker.metrics()["assignment_solver"], "hungarian")


if __name__ == "__main__":
    unittest.main()
