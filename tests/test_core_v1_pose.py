from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from services.ml_service.pose.coordinator import PoseCoordinator


class _Store:
    def get_frame(self, frame_id):
        return None


class _Detections:
    def snapshot(self):
        return {}


class PoseCoordinatorTests(unittest.TestCase):
    def test_disabled_pose_does_not_start_worker(self):
        coordinator = PoseCoordinator(
            {"CAM-01": _Store()},
            _Detections(),
            {"enabled": False},
        )
        coordinator.start()
        self.assertIsNone(coordinator._thread)
        self.assertFalse(coordinator.metrics()["enabled"])

    def test_crop_clamps_detector_box_to_source_frame(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        frame = SimpleNamespace(image=image)
        box = SimpleNamespace(x1=-20.0, y1=-5.0, x2=250.0, y2=120.0)

        crop, bounds = PoseCoordinator._crop(frame, box)

        self.assertEqual(bounds, (0, 0, 200, 100))
        self.assertEqual(crop.shape, (100, 200, 3))

    def test_metrics_are_safe_before_model_load(self):
        coordinator = PoseCoordinator(
            {"CAM-01": _Store()},
            _Detections(),
            {
                "enabled": True,
                "model": "yolo11n-pose.pt",
                "device": "cuda:0",
                "every_n": 3,
            },
        )
        metrics = coordinator.metrics()
        self.assertTrue(metrics["enabled"])
        self.assertFalse(metrics["ready"])
        self.assertEqual(metrics["processed"], 0)
        self.assertEqual(metrics["model"], "yolo11n-pose.pt")


if __name__ == "__main__":
    unittest.main()
