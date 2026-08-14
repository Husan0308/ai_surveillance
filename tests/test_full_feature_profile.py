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

    def test_pose_is_enabled_but_has_no_cuda_context(self):
        pose = dict(self.cfg.get("pose") or {})
        self.assertTrue(pose.get("enabled"))
        self.assertEqual(pose.get("model"), "yolo26m-pose.pt")
        self.assertEqual(pose.get("device"), "cpu")
        self.assertFalse(pose.get("half"))
        self.assertEqual(int(pose.get("max_cameras_per_cycle", 0)), 1)

    def test_reid_and_heatmap_are_enabled_as_side_paths(self):
        reid = dict(self.cfg.get("reid") or {})
        heatmap = dict(self.cfg.get("heatmap") or {})
        self.assertTrue(reid.get("enabled"))
        self.assertEqual(reid.get("device"), "cpu")
        self.assertTrue(reid.get("model_url"))
        self.assertTrue(heatmap.get("enabled"))
        self.assertFalse(heatmap.get("fallback_bbox_bottom"))

    def test_profile_declares_full_safe_mode(self):
        self.assertEqual(self.cfg.get("profile"), "full-features-safe")


if __name__ == "__main__":
    unittest.main()
