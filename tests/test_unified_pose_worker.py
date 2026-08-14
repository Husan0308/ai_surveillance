from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from services.ml_service.core_v1.unified_detector import (
    LatestPoseStore,
    _pose_crop,
    _raw_pose_person,
)
from services.ml_service.pose.coordinator import PoseResult


class _Tensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class UnifiedPoseWorkerTests(unittest.TestCase):
    def test_pose_crop_maps_source_box_into_detector_resize(self):
        entry = {
            "source_w": 1000,
            "source_h": 500,
            "full_shape": (250, 500),
            "full_image": np.zeros((250, 500, 3), dtype=np.uint8),
        }
        crop, offset, back_scale = _pose_crop(
            entry,
            (200.0, 100.0, 600.0, 300.0, 0.9),
        )
        self.assertEqual(crop.shape[:2], (100, 200))
        self.assertEqual(offset, (100, 50))
        self.assertEqual(back_scale, (2.0, 2.0))

    def test_pose_keypoints_map_back_to_original_camera_pixels(self):
        xy = [[10.0 + i, 20.0 + i] for i in range(17)]
        conf = [0.9] * 17
        prediction = SimpleNamespace(
            keypoints=SimpleNamespace(
                xy=[_Tensor(xy)],
                conf=[_Tensor(conf)],
            ),
            boxes=None,
        )
        person = _raw_pose_person(
            prediction,
            (200.0, 100.0, 600.0, 300.0, 0.8),
            offset=(100, 50),
            back_scale=(2.0, 2.0),
        )
        self.assertIsNotNone(person)
        self.assertEqual(person["bbox"], (200.0, 100.0, 600.0, 300.0))
        self.assertEqual(person["keypoints"][0], (220.0, 140.0, 0.9))
        self.assertEqual(person["keypoints"][15], (250.0, 170.0, 0.9))
        self.assertEqual(person["keypoints"][16], (252.0, 172.0, 0.9))

    def test_latest_pose_store_never_replaces_newer_frame_with_older_one(self):
        store = LatestPoseStore()
        newer = PoseResult("CAM-01", 12, 10.0, 11.0, ())
        older = PoseResult("CAM-01", 11, 9.0, 10.0, ())
        store.put(newer)
        store.put(older)
        self.assertEqual(store.snapshot()["CAM-01"].frame_id, 12)


if __name__ == "__main__":
    unittest.main()
