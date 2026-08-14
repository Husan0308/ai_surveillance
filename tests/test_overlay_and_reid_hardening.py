from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from services.ml_service.core_v1.heatmap_publisher import HeatmapJpegPublisher
from services.ml_service.core_v1.reid_hardening import HardenedGlobalIdentityManager
from services.ml_service.heatmap.camera_overlay import CameraAnkleHeatmapCoordinator


class _Store:
    def get(self):
        return SimpleNamespace(image=np.zeros((180, 320, 3), dtype=np.uint8)), 1


class _Pose:
    enabled = True

    def snapshot(self):
        return {}


class OverlayAndReIDHardeningTests(unittest.TestCase):
    def test_bbox_bottom_center_fallback(self):
        box = SimpleNamespace(x1=20.0, y1=10.0, x2=100.0, y2=160.0)
        self.assertEqual(CameraAnkleHeatmapCoordinator._bbox_contact(box), (60.0, 160.0))

    def test_bbox_fallback_has_lower_weight_than_pose(self):
        coordinator = CameraAnkleHeatmapCoordinator(
            _Pose(),
            {"CAM-01": _Store()},
            {
                "enabled": True,
                "pose_weight": 1.0,
                "bbox_fallback_weight": 0.38,
            },
        )
        self.assertLess(coordinator.bbox_weight, coordinator.pose_weight)

    def test_overlay_toggle_is_presentation_only(self):
        heatmap = CameraAnkleHeatmapCoordinator(
            _Pose(),
            {"CAM-01": _Store()},
            {"enabled": True},
        )
        publisher = HeatmapJpegPublisher(
            "CAM-01",
            _Store(),
            heatmap_provider=heatmap,
            pose_provider=_Pose(),
            heatmap_visible=True,
            pose_visible=False,
        )
        state = publisher.set_overlay_state(heatmap=False, pose=True)
        self.assertFalse(state["heatmap_visible"])
        self.assertTrue(state["pose_visible"])
        self.assertTrue(heatmap.enabled)
        self.assertTrue(publisher.pose_provider.enabled)

    def test_historical_identity_reuses_previous_global_id(self):
        manager = HardenedGlobalIdentityManager(
            {
                "historical_match_threshold": 0.60,
                "historical_strong_threshold": 0.70,
                "historical_margin": 0.035,
                "camera_rooms": {"CAM-01": "ROOM-1", "CAM-04": "ROOM-1"},
            }
        )
        vector = np.asarray([1.0, 0.2, 0.1], dtype=np.float32)
        gid1, reason1 = manager.ensure_track("CAM-01", 1, vector, 1.0)
        self.assertEqual(reason1, "new")
        manager.release_track("CAM-01", 1)
        gid2, reason2 = manager.ensure_track("CAM-04", 2, vector, 2.0)
        self.assertEqual(gid2, gid1)
        self.assertEqual(reason2, "historical")

    def test_bad_existing_descriptor_does_not_poison_prototype(self):
        manager = HardenedGlobalIdentityManager(
            {
                "prototype_update_min_similarity": 0.66,
                "gallery_update_min_similarity": 0.70,
            }
        )
        good = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        bad = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        gid, _ = manager.ensure_track("CAM-01", 1, good, 1.0)
        before = manager._identities[gid].prototype.copy()
        gid2, _ = manager.ensure_track("CAM-01", 1, bad, 2.0)
        after = manager._identities[gid].prototype.copy()
        self.assertEqual(gid2, gid)
        self.assertTrue(np.allclose(before, after))
        metrics = manager.metrics()
        self.assertGreaterEqual(metrics["prototype_freezes"], 1)
        self.assertGreaterEqual(metrics["gallery_freezes"], 1)


if __name__ == "__main__":
    unittest.main()
