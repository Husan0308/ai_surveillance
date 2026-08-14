from __future__ import annotations

from pathlib import Path
import unittest

import yaml


class FullFeatureProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        with (root / "config" / "core_v1.yaml").open("r", encoding="utf-8") as handle:
            cls.cfg = (yaml.safe_load(handle) or {}).get("core_v1", {})

    def test_primary_detector_stays_enabled_on_cuda(self):
        detector = dict(self.cfg.get("detector") or {})
        self.assertTrue(detector.get("enabled"))
        self.assertEqual(detector.get("model"), "yolo26m.pt")
        self.assertEqual(detector.get("device"), "cuda:0")
        self.assertFalse((detector.get("roi_second_pass") or {}).get("enabled"))

    def test_pose_is_spawned_cpu_side_path_with_hidden_overlay_default(self):
        pose = dict(self.cfg.get("pose") or {})
        self.assertTrue(pose.get("enabled"))
        self.assertEqual(pose.get("model"), "yolo26m-pose.pt")
        self.assertEqual(pose.get("device"), "cpu")
        self.assertEqual(int(pose.get("max_people", 0)), 1)
        self.assertFalse(bool(pose.get("overlay_default")))
        self.assertGreaterEqual(float(pose.get("restart_backoff_sec", 0)), 1.0)

    def test_heatmap_is_lightweight_and_has_bbox_fallback(self):
        heatmap = dict(self.cfg.get("heatmap") or {})
        self.assertTrue(heatmap.get("enabled"))
        self.assertTrue(heatmap.get("display_default"))
        self.assertEqual(heatmap.get("coordinate_system"), "camera_pixels")
        self.assertLess(float(heatmap.get("bbox_fallback_weight", 1)), float(heatmap.get("pose_weight", 0)))
        self.assertLessEqual(int(heatmap.get("camera_grid_width", 999)), 160)
        self.assertLess(float(heatmap.get("overlay_alpha", 1)), 0.40)

    def test_reid_is_cpu_hardened_tracklet_profile(self):
        reid = dict(self.cfg.get("reid") or {})
        tracklet = dict(reid.get("tracklet") or {})
        identity = dict(reid.get("identity") or {})
        self.assertTrue(reid.get("enabled"))
        self.assertEqual(reid.get("device"), "cpu")
        self.assertGreaterEqual(int(tracklet.get("pair_min_samples", 0)), 4)
        self.assertGreater(float(tracklet.get("mature_outlier_min_cos", 0)), 0.0)
        self.assertGreater(float(identity.get("historical_match_threshold", 0)), 0.0)
        self.assertGreater(float(identity.get("prototype_update_min_similarity", 0)), 0.0)

    def test_profile_declares_hardened_mode(self):
        self.assertEqual(self.cfg.get("profile"), "camera-heatmap-reid-hardened")


if __name__ == "__main__":
    unittest.main()
