from __future__ import annotations

from collections import deque
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import yaml

from services.ml_service.core_v1.detector import PersonBox
from services.ml_service.core_v1.reid_service import ReIDCoordinator, _LocalTrack
from services.ml_service.core_v1.spatial_calibration import RoomSpatialMapper


class SpatialCalibrationTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / "config" / "room_mapping.yaml"
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "room_mapping.yaml"
        self.path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        self.mapper = RoomSpatialMapper(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def _calibrate(self, camera_id):
        image = [
            [0, 0], [1000, 0], [1000, 500], [0, 500],
            [500, 0], [500, 500], [200, 300], [800, 250],
        ]
        room = [[x / 1000.0, y / 500.0] for x, y in image]
        return self.mapper.calibrate(camera_id, image, room, [1000, 500])

    def test_verified_pairs_are_persistent_but_geometry_starts_uncalibrated(self):
        self.assertEqual(
            self.mapper.camera_pairs(),
            [("CAM-01", "CAM-04"), ("CAM-02", "CAM-05"), ("CAM-03", "CAM-06")],
        )
        self.assertIsNone(
            self.mapper.project_box_footpoint(
                "CAM-01", PersonBox(100, 100, 200, 400, 0.9)
            )
        )
        self.assertEqual(self.mapper.snapshot()["summary"]["calibrated_cameras"], 0)

    def test_assisted_homography_uses_bottom_center_and_persists(self):
        calibration = self._calibrate("CAM-01")
        self.assertEqual(calibration["status"], "good")
        projected = self.mapper.project_box_footpoint(
            "CAM-01", PersonBox(300, 100, 500, 400, 0.9)
        )
        self.assertAlmostEqual(projected["x"], 0.4, places=4)
        self.assertAlmostEqual(projected["y"], 0.8, places=4)
        scaled = self.mapper.project_box_footpoint(
            "CAM-01", PersonBox(600, 100, 1000, 800, 0.9), source_size=(2000, 1000)
        )
        self.assertAlmostEqual(scaled["x"], 0.4, places=4)
        self.assertAlmostEqual(scaled["y"], 0.8, places=4)
        self.assertFalse(self.mapper.snapshot()["summary"]["spatial_fusion_active"])
        self._calibrate("CAM-04")
        self.assertEqual(self.mapper.snapshot()["summary"]["active_rooms"], ["ROOM-1"])
        reloaded = RoomSpatialMapper(self.path)
        self.assertAlmostEqual(reloaded.project_point("CAM-01", (400, 400))["x"], 0.4, places=4)

    def test_invalid_or_insufficient_landmarks_fail_closed(self):
        with self.assertRaises(ValueError):
            self.mapper.calibrate("CAM-01", [[0, 0]] * 5, [[0.2, 0.2]] * 5)
        with self.assertRaises(ValueError):
            self.mapper.calibrate("CAM-01", [[0, 0]] * 6, [[2.0, 0.2]] * 6)
        self.assertIsNone(self.mapper.snapshot()["calibrations"]["CAM-01"]["homography"])

    def test_spatial_fusion_boosts_nearby_and_rejects_impossible_pair(self):
        coordinator = ReIDCoordinator({}, None, {"enabled": False}, self.mapper)
        now = time.monotonic()
        near_a = _LocalTrack(1, object(), now, room_id="ROOM-1", room_position=(0.40, 0.50), spatial_observed_at=now)
        near_b = _LocalTrack(2, object(), now, room_id="ROOM-1", room_position=(0.44, 0.51), spatial_observed_at=now + 0.03)
        near = coordinator._fusion_detail(near_a, near_b, 0.55)
        self.assertTrue(near["spatial_available"])
        self.assertGreater(near["fusion_score"], 0.55)

        far = _LocalTrack(3, object(), now, room_id="ROOM-1", room_position=(0.98, 0.95), spatial_observed_at=now + 0.03)
        impossible = coordinator._fusion_detail(near_a, far, 0.99)
        self.assertTrue(impossible["impossible"])

    def test_already_merged_identity_is_split_on_impossible_simultaneous_positions(self):
        coordinator = ReIDCoordinator(
            {"CAM-01": object(), "CAM-04": object()}, None,
            {"enabled": False, "tracklet": {"min_samples": 2},
             "pair_matching": {"pairs": [["CAM-01", "CAM-04"]]}},
            self.mapper,
        )
        now = time.monotonic()
        descriptor = np.asarray([1.0, 0.0], dtype=np.float32)
        a = _LocalTrack(1, object(), now, descriptor=descriptor,
                        room_id="ROOM-1", room_position=(0.05, 0.05),
                        spatial_observed_at=now, samples=deque([1, 2]))
        b = _LocalTrack(2, object(), now, descriptor=descriptor,
                        room_id="ROOM-1", room_position=(0.95, 0.95),
                        spatial_observed_at=now, samples=deque([1, 2]))
        left_gid, _ = coordinator.identities.ensure_track("CAM-01", 1, descriptor, now)
        coordinator.identities.ensure_track("CAM-04", 2, descriptor, now)
        merged_gid, _ = coordinator.identities.merge_tracks("CAM-01", 1, "CAM-04", 2, 0.9, now)
        self.assertEqual(left_gid, merged_gid)
        a.global_id = merged_gid
        b.global_id = merged_gid
        coordinator._tracks["CAM-01"][1] = a
        coordinator._tracks["CAM-04"][2] = b

        coordinator._evaluate_pairs()

        self.assertEqual(a.global_id, merged_gid)
        self.assertIsNone(b.global_id)
        self.assertIsNone(coordinator.identities.lookup_track("CAM-04", 2))
        self.assertEqual(coordinator.metrics()["spatial_rejects"], 1)

    def test_room_map_deduplicates_same_global_identity_across_cameras(self):
        coordinator = ReIDCoordinator(
            {"CAM-01": object(), "CAM-04": object()}, None, {"enabled": False}, self.mapper
        )
        now = time.monotonic()
        coordinator._tracks["CAM-01"][1] = _LocalTrack(
            1, object(), now, global_id="G-007", room_id="ROOM-1",
            room_position=(0.40, 0.60), spatial_observed_at=now,
            samples=deque(),
        )
        coordinator._tracks["CAM-04"][2] = _LocalTrack(
            2, object(), now, global_id="G-007", room_id="ROOM-1",
            room_position=(0.44, 0.60), spatial_observed_at=now,
            samples=deque(),
        )
        people = coordinator.room_people()
        self.assertEqual(len(people), 1)
        self.assertEqual(people[0]["camera_count"], 2)
        self.assertEqual(people[0]["global_id"], "G-007")


if __name__ == "__main__":
    unittest.main()
