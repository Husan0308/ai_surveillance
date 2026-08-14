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

    def test_pose_is_spawned_cpu_side_path(self):
        pose = dict(self.cfg.get("pose") or {})
        self.assertTrue(pose.get("enabled"))
        self.assertEqual(pose.get("model"), "yolo26m-pose.pt")
        self.assertEqual(pose.get("device"), "cpu")
        self.assertEqual(int(pose.get("max_people", 0)), 1)
        self.assertGreaterEqual(float(pose.get("restart_backoff_sec", 0)), 1.0)

    def test_reid_and_camera_heatmap_are_enabled(self):
        reid = dict(self.cfg.get("reid") or {})
        heatmap = dict(self.cfg.get("heatmap") or {})
        self.assertTrue(reid.get("enabled"))
        self.assertEqual(reid.get("device"), "cpu")
        self.assertTrue(reid.get("model_url"))
        self.assertTrue(heatmap.get("enabled"))
        self.assertEqual(heatmap.get("coordinate_system"), "camera_pixels")
        self.assertGreater(float(heatmap.get("overlay_alpha", 0)), 0.0)

    def test_profile_declares_camera_heatmap_isolation(self):
        self.assertEqual(
            self.cfg.get("profile"),
            "camera-ankle-heatmap-crash-isolated",
        )


if __name__ == "__main__":
    unittest.main()
