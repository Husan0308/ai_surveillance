from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from services.ml_service.heatmap.coordinator import FloorHeatmapCoordinator


class _Pose:
    def snapshot(self):
        return {}


class _FrameStore:
    def get(self):
        return SimpleNamespace(width=1000, height=500), 1


class _Mapper:
    def snapshot(self):
        return {
            "rooms": {"ROOM-1": {"cameras": ["CAM-01", "CAM-04"]}},
            "summary": {"active_rooms": ["ROOM-1"]},
        }

    def room_for_camera(self, camera_id):
        return "ROOM-1"

    def project_point(self, camera_id, point, source_size=None):
        width, height = source_size or (1000, 500)
        return {
            "room_id": "ROOM-1",
            "x": point[0] / width,
            "y": point[1] / height,
            "inside_overlap": True,
            "calibration_confidence": 1.0,
        }


def _person(left=(400.0, 420.0, 0.9), right=(600.0, 420.0, 0.8)):
    points = [SimpleNamespace(x=0.0, y=0.0, confidence=0.0) for _ in range(17)]
    points[15] = SimpleNamespace(x=left[0], y=left[1], confidence=left[2])
    points[16] = SimpleNamespace(x=right[0], y=right[1], confidence=right[2])
    return SimpleNamespace(
        bbox=(350.0, 100.0, 650.0, 430.0),
        keypoints=tuple(points),
    )


class FloorHeatmapTests(unittest.TestCase):
    def setUp(self):
        self.heatmap = FloorHeatmapCoordinator(
            _Pose(),
            {"CAM-01": _FrameStore(), "CAM-04": _FrameStore()},
            _Mapper(),
            {
                "enabled": True,
                "grid_width": 48,
                "grid_height": 32,
                "ankle_conf": 0.35,
                "hot_hold_sec": 3600,
                "cool_half_life_sec": 3600,
                "bucket_sec": 300,
                "dedupe_window_ms": 800,
                "dedupe_distance": 0.06,
            },
        )

    def test_ankle_midpoint_uses_valid_left_and_right_ankles(self):
        point = self.heatmap._ankle_point(_person())
        self.assertEqual(point, (500.0, 420.0))

    def test_single_valid_ankle_is_accepted(self):
        point = self.heatmap._ankle_point(
            _person(left=(410.0, 410.0, 0.9), right=(610.0, 410.0, 0.1))
        )
        self.assertEqual(point, (410.0, 410.0))

    def test_heat_stays_full_for_one_hour_then_cools_by_half_life(self):
        self.assertAlmostEqual(self.heatmap._bucket_weight(3599.0), 1.0)
        self.assertAlmostEqual(self.heatmap._bucket_weight(3600.0), 1.0)
        self.assertAlmostEqual(self.heatmap._bucket_weight(7200.0), 0.5, places=6)
        self.assertAlmostEqual(self.heatmap._bucket_weight(10800.0), 0.25, places=6)

    def test_same_room_overlap_duplicate_is_not_double_counted(self):
        self.assertFalse(self.heatmap._is_duplicate("ROOM-1", 100.0, 0.50, 0.50))
        self.assertTrue(self.heatmap._is_duplicate("ROOM-1", 100.2, 0.52, 0.51))
        self.assertFalse(self.heatmap._is_duplicate("ROOM-1", 101.0, 0.52, 0.51))

    def test_pose_ankles_project_to_floor_grid(self):
        result = SimpleNamespace(
            frame_captured_monotonic=1000.0,
            people=(_person(),),
        )
        self.heatmap._consume_pose_result("CAM-01", result)
        grid = self.heatmap.room_grid("ROOM-1", now=1000.0)
        self.assertGreater(float(grid.max()), 0.0)
        peak_y, peak_x = np.unravel_index(int(np.argmax(grid)), grid.shape)
        self.assertAlmostEqual(peak_x / (grid.shape[1] - 1), 0.5, delta=0.05)
        self.assertAlmostEqual(peak_y / (grid.shape[0] - 1), 0.84, delta=0.06)

    def test_transparent_png_is_available_for_known_room(self):
        png = self.heatmap.render_png("ROOM-1", now=1000.0)
        self.assertIsInstance(png, bytes)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
