from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import time
import unittest

import numpy as np

from services.ml_service.core_v1.heatmap_publisher_v2 import _bbox_distance
from services.ml_service.heatmap.camera_overlay_v2 import CameraAnkleHeatmapCoordinator


class _Store:
    def get(self):
        return SimpleNamespace(image=np.zeros((180, 320, 3), dtype=np.uint8)), 1


class _Pose:
    def snapshot(self):
        return {}


class OperatorUiPoseHeatmapV2Tests(unittest.TestCase):
    def test_operator_ui_contains_reference_sections_and_three_by_two_grid(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "services/frontend/core_v1/operator_dashboard.py").read_text(
            encoding="utf-8"
        )
        patch = (root / "services/frontend/core_v1/operator_dashboard_v2.py").read_text(
            encoding="utf-8"
        )
        for label in (
            "AI Surveillance System",
            "Operator Console • v3.0 MUKAMMAL",
            "Dashboard",
            "Person Management",
            "Enrollment",
            "Analytics",
            "Events",
            "Settings",
            "LIVE STATUS",
            "Heatmap",
            "Pose",
        ):
            self.assertIn(label, source)
        self.assertIn("index // 3, index % 3", patch)
        self.assertIn('setCurrentText("3 × 2")', patch)

    def test_pose_bbox_distance_allows_tracker_reprojection(self):
        old_bbox = (100.0, 80.0, 180.0, 240.0)
        nearby = SimpleNamespace(x1=120.0, y1=90.0, x2=200.0, y2=250.0)
        far = SimpleNamespace(x1=800.0, y1=500.0, x2=900.0, y2=700.0)
        self.assertLess(_bbox_distance(old_bbox, nearby), 1.8)
        self.assertGreater(_bbox_distance(old_bbox, far), 1.8)

    def test_bbox_fallback_is_suppressed_only_near_recent_pose_contact(self):
        coordinator = CameraAnkleHeatmapCoordinator(
            _Pose(),
            {"CAM-01": _Store()},
            {"enabled": True, "pose_cover_sec": 2.0, "pose_cover_distance_norm": 0.08},
            detections=None,
        )
        now = time.monotonic()
        coordinator._remember_pose_contact("CAM-01", 100.0, 150.0, 320.0, 180.0, now)
        self.assertTrue(
            coordinator._is_pose_covered(
                "CAM-01", 105.0, 152.0, 320.0, 180.0, now + 0.1
            )
        )
        self.assertFalse(
            coordinator._is_pose_covered(
                "CAM-01", 260.0, 150.0, 320.0, 180.0, now + 0.1
            )
        )

    def test_v2_tunes_fallback_for_visible_lightweight_heat(self):
        coordinator = CameraAnkleHeatmapCoordinator(
            _Pose(),
            {"CAM-01": _Store()},
            {
                "enabled": True,
                "bbox_fallback_every_n": 4,
                "overlay_alpha": 0.28,
                "overlay_threshold": 0.025,
                "dedupe_window_ms": 450,
            },
            detections=None,
        )
        self.assertLessEqual(coordinator.fallback_every_n, 2)
        self.assertGreaterEqual(coordinator.overlay_alpha, 0.34)
        self.assertLessEqual(coordinator.overlay_threshold, 0.018)
        self.assertLessEqual(coordinator.dedupe_sec, 0.30)


if __name__ == "__main__":
    unittest.main()
