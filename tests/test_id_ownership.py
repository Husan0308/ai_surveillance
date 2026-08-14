from __future__ import annotations

from types import SimpleNamespace
import unittest

from services.ml_service.core_v1.ownership_tracker import OwnershipLockedTracker
from services.ml_service.core_v1.visual_tracker import VisualBox


def make_tracker(camera_id: str, **overrides):
    kwargs = dict(
        camera_id=camera_id,
        ownership_lock=True,
        ownership_min_hits=1,
        ownership_margin=0.12,
        ownership_low_margin_multiplier=1.5,
        ownership_proximity=0.80,
        ownership_overlap_iou=0.05,
        ownership_iou_advantage=0.14,
        ownership_distance_advantage=0.14,
        ownership_max_competitor_age_ms=800,
        id_namespace_stride=10000,
        strong_confirm_hits=1,
        weak_confirm_hits=2,
        start_conf=0.25,
        new_track_min_conf=0.25,
        byte_high_conf=0.25,
        byte_low_conf=0.08,
        low_conf_confirm=0.08,
        hold_ms=850,
        memory_ms=2800,
        prediction_ms=340,
    )
    kwargs.update(overrides)
    return OwnershipLockedTracker(**kwargs)


class IdOwnershipTests(unittest.TestCase):
    def test_camera_namespaces_never_reuse_same_numeric_id(self):
        cam1 = make_tracker("CAM-01")
        cam2 = make_tracker("CAM-02")
        result = SimpleNamespace(
            frame_id=1,
            frame_captured_monotonic=1.0,
            boxes=(VisualBox(100, 60, 180, 300, 0.9),),
        )
        cam1.update(result, 1.0, 736, 416)
        cam2.update(result, 1.0, 736, 416)
        one = cam1.visible(1.0, target_time=1.0)[0]
        two = cam2.visible(1.0, target_time=1.0)[0]
        self.assertNotEqual(one.track_id, two.track_id)
        self.assertEqual(cam1.display_label(one.track_id), "C01-001")
        self.assertEqual(cam2.display_label(two.track_id), "C02-001")

    def test_ambiguous_close_person_detection_is_quarantined(self):
        tracker = make_tracker("CAM-01", ownership_margin=0.18)
        a = VisualBox(80, 60, 140, 300, 0.9)
        b = VisualBox(130, 60, 190, 300, 0.9)
        ta = tracker._init_track(a, 1.0, 1.0, 1, hits=5)
        tb = tracker._init_track(b, 1.0, 1.0, 1, hits=5)
        tracker._tracks[ta.track_id] = ta
        tracker._tracks[tb.track_id] = tb

        ambiguous_a = VisualBox(104, 60, 164, 300, 0.85)
        ambiguous_b = VisualBox(106, 60, 166, 300, 0.84)
        matches, used_tracks, used_dets = tracker._associate(
            [ta.track_id, tb.track_id],
            [ambiguous_a, ambiguous_b],
            observation=1.10,
            decisive_high=True,
        )
        self.assertEqual(matches, [])
        self.assertEqual(used_tracks, set())
        self.assertEqual(used_dets, set())
        self.assertGreaterEqual(tracker.metrics()["ownership_rejects"], 1)

    def test_clear_owner_is_not_blocked(self):
        tracker = make_tracker("CAM-01", ownership_margin=0.08)
        a = VisualBox(50, 60, 110, 300, 0.9)
        b = VisualBox(180, 60, 240, 300, 0.9)
        ta = tracker._init_track(a, 1.0, 1.0, 1, hits=5)
        tb = tracker._init_track(b, 1.0, 1.0, 1, hits=5)
        tracker._tracks[ta.track_id] = ta
        tracker._tracks[tb.track_id] = tb

        detections = [
            VisualBox(52, 60, 112, 300, 0.90),
            VisualBox(178, 60, 238, 300, 0.88),
        ]
        matches, used_tracks, used_dets = tracker._associate(
            [ta.track_id, tb.track_id],
            detections,
            observation=1.10,
            decisive_high=True,
        )
        self.assertEqual(len(matches), 2)
        self.assertEqual(used_tracks, {ta.track_id, tb.track_id})
        self.assertEqual(used_dets, {0, 1})

    def test_quarantined_detection_cannot_birth_duplicate_id(self):
        tracker = make_tracker("CAM-03")
        det = VisualBox(100, 60, 180, 300, 0.9)
        tracker._ownership_quarantine.add(id(det))
        result = tracker._confirm_birth(
            det,
            observation=2.0,
            now=2.0,
            frame_id=1,
            required_hits=1,
            used_candidates=set(),
        )
        self.assertIsNone(result)
        self.assertEqual(tracker.metrics()["ownership_birth_blocks"], 1)

    def test_snapshot_exposes_camera_and_display_id(self):
        tracker = make_tracker("CAM-04")
        result = SimpleNamespace(
            frame_id=1,
            frame_captured_monotonic=3.0,
            boxes=(VisualBox(20, 30, 90, 250, 0.9),),
        )
        tracker.update(result, 3.0, 736, 416)
        tracker.visible(3.0, target_time=3.0)
        row = tracker.snapshot()[0]
        self.assertEqual(row["camera_id"], "CAM-04")
        self.assertEqual(row["display_id"], "C04-001")
        self.assertEqual(row["local_sequence"], 1)


if __name__ == "__main__":
    unittest.main()
