from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from services.ml_service.heatmap.camera_overlay import CameraAnkleHeatmapCoordinator
from services.ml_service.pose.process_coordinator import PoseProcessCoordinator


class _Pose:
    def snapshot(self):
        return {}


class _Store:
    def get(self):
        frame = SimpleNamespace(image=np.zeros((180, 320, 3), dtype=np.uint8))
        return frame, 1


class CameraAnkleHeatmapTests(unittest.TestCase):
    def test_ankle_midpoint_uses_coco_indices_15_and_16(self):
        coordinator = CameraAnkleHeatmapCoordinator(
            _Pose(),
            {"CAM-01": _Store()},
            {"enabled": True, "ankle_conf": 0.30},
        )
        points = [SimpleNamespace(x=0.0, y=0.0, confidence=0.0) for _ in range(17)]
        points[15] = SimpleNamespace(x=100.0, y=150.0, confidence=0.9)
        points[16] = SimpleNamespace(x=120.0, y=154.0, confidence=0.8)
        person = SimpleNamespace(keypoints=tuple(points))
        self.assertEqual(coordinator._ankle(person), (110.0, 152.0))

    def test_single_visible_ankle_is_accepted(self):
        coordinator = CameraAnkleHeatmapCoordinator(
            _Pose(),
            {"CAM-01": _Store()},
            {"enabled": True, "ankle_conf": 0.30},
        )
        points = [SimpleNamespace(x=0.0, y=0.0, confidence=0.0) for _ in range(17)]
        points[15] = SimpleNamespace(x=80.0, y=140.0, confidence=0.95)
        points[16] = SimpleNamespace(x=90.0, y=142.0, confidence=0.10)
        person = SimpleNamespace(keypoints=tuple(points))
        self.assertEqual(coordinator._ankle(person), (80.0, 140.0))

    def test_overlay_changes_pixels_when_heat_exists(self):
        coordinator = CameraAnkleHeatmapCoordinator(
            _Pose(),
            {"CAM-01": _Store()},
            {
                "enabled": True,
                "camera_grid_width": 80,
                "camera_grid_height": 45,
                "overlay_alpha": 0.5,
                "overlay_threshold": 0.01,
            },
        )
        with coordinator._lock:
            coordinator._grids["CAM-01"][20:25, 38:43] = 10.0
        image = np.zeros((180, 320, 3), dtype=np.uint8)
        result = coordinator.overlay("CAM-01", image)
        self.assertEqual(result.shape, image.shape)
        self.assertGreater(int(result.sum()), 0)

    def test_pose_runtime_declares_spawned_process_isolation(self):
        coordinator = PoseProcessCoordinator(
            {"CAM-01": _Store()},
            None,
            {"enabled": True, "model": "yolo26m-pose.pt"},
        )
        metrics = coordinator.metrics()
        self.assertEqual(metrics["start_method"], "spawn")
        self.assertEqual(metrics["device"], "cpu")
        self.assertEqual(metrics["isolation"], "separate_pose_process")
        self.assertFalse(metrics["detector_gating"])


if __name__ == "__main__":
    unittest.main()
