from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

# When this file is executed directly with
#   python scripts/test_camera_v11_bbox_overlay_v1.py
# Python places the scripts/ directory at sys.path[0], not the repository root.
# Add the repo root explicitly so project packages such as services.* resolve
# without requiring callers to set PYTHONPATH manually.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v11.bbox_overlay_ipc_v1 import (
    BboxStateReader,
    BboxStateWriter,
    local_track_number,
    predict_bbox_norm,
    tracker_box_to_display,
)


class BboxOverlayV1Tests(unittest.TestCase):
    def test_local_track_number(self) -> None:
        self.assertEqual(local_track_number("CAM-01-T00037"), 37)
        self.assertEqual(local_track_number("CAM-06-T1"), 1)
        with self.assertRaises(ValueError):
            local_track_number("GID-00001")

    def test_prediction_is_bounded(self) -> None:
        box = predict_bbox_norm((0.10, 0.20, 0.30, 0.60), (0.20, 0.0, 0.0, 0.0), 5.0)
        # dt is capped to 0.45 s, so x translates by 0.09 rather than by 1.0.
        self.assertAlmostEqual(box[0], 0.19, places=6)
        self.assertAlmostEqual(box[2], 0.39, places=6)

    def test_detector_padding_is_removed_before_display_mapping(self) -> None:
        # Rows 3..381 in the 384 canvas are the full 378px content.
        mapped = tracker_box_to_display(
            (0.0, 3.0 / 384.0, 1.0, 381.0 / 384.0),
            (0.0, 0.0, 0.0, 0.0),
            0.0,
            640,
            360,
        )
        self.assertAlmostEqual(mapped[0], 0.0, places=4)
        self.assertAlmostEqual(mapped[1], 0.0, places=4)
        self.assertAlmostEqual(mapped[2], 640.0, places=4)
        self.assertAlmostEqual(mapped[3], 360.0, places=4)

    def test_atomic_latest_only_and_stale_expiry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bbox.json"
            writer = BboxStateWriter(path)
            reader = BboxStateReader(path)
            captured_ns = time.monotonic_ns()
            writer.publish(
                "CAM-01",
                captured_ns,
                [
                    {
                        "local_id": 7,
                        "confidence": 0.91,
                        "bbox_norm": [0.1, 0.1, 0.4, 0.8],
                        "velocity_norm_s": [0.0, 0.0, 0.0, 0.0],
                    }
                ],
            )
            rows = reader.camera_tracks(
                "CAM-01", now_ns=captured_ns + 100_000_000, stale_sec=1.1, width=640, height=360
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], 7)

            # A later publish replaces the slot instead of appending a queue.
            writer.publish("CAM-01", captured_ns + 200_000_000, [])
            self.assertEqual(
                reader.camera_tracks(
                    "CAM-01", now_ns=captured_ns + 250_000_000, stale_sec=1.1, width=640, height=360
                ),
                [],
            )

            writer.publish(
                "CAM-01",
                captured_ns,
                [
                    {
                        "local_id": 8,
                        "confidence": 0.9,
                        "bbox_norm": [0.2, 0.2, 0.5, 0.8],
                        "velocity_norm_s": [0.0, 0.0, 0.0, 0.0],
                    }
                ],
            )
            stale = reader.camera_tracks(
                "CAM-01", now_ns=captured_ns + 1_500_000_000, stale_sec=1.1, width=640, height=360
            )
            self.assertEqual(stale, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
