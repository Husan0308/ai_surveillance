from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

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
from services.camera_v11.bbox_single_target_lock_v1 import (
    SingleTargetBboxLockV1,
    SingleTargetLockConfigV1,
)
from services.camera_v11.step3_single_occupant_tracker_v1 import V11SingleOccupantTrackerV1


def _track(
    track_id: str,
    *,
    hits: int = 3,
    score: float = 0.90,
    bbox: tuple[float, float, float, float] = (0.10, 0.10, 0.40, 0.80),
    predicted: bool = False,
    state: str = "tracked",
    since_detection: float = 0.0,
) -> dict[str, object]:
    return {
        "track_id": track_id,
        "local_id": int(track_id.rsplit("T", 1)[-1]),
        "state": state,
        "predicted": predicted,
        "confidence": score,
        "hits": hits,
        "age_sec": 2.0,
        "bbox_norm": list(bbox),
        "velocity_norm_s": [0.0, 0.0, 0.0, 0.0],
        "since_detection_sec": since_detection,
    }


class BboxOverlayV1Tests(unittest.TestCase):
    def test_local_track_number(self) -> None:
        self.assertEqual(local_track_number("CAM-01-T00037"), 37)
        self.assertEqual(local_track_number("CAM-06-T1"), 1)
        with self.assertRaises(ValueError):
            local_track_number("GID-00001")

    def test_prediction_is_bounded(self) -> None:
        box = predict_bbox_norm((0.10, 0.20, 0.30, 0.60), (0.20, 0.0, 0.0, 0.0), 5.0)
        self.assertAlmostEqual(box[0], 0.19, places=6)
        self.assertAlmostEqual(box[2], 0.39, places=6)

    def test_detector_padding_is_removed_before_display_mapping(self) -> None:
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

    def test_single_target_lock_suppresses_competing_boxes(self) -> None:
        gate = SingleTargetBboxLockV1(SingleTargetLockConfigV1(acquire_updates=2))
        t0 = 1_000_000_000
        person = _track("CAM-01-T00001", hits=2, score=0.82)
        ghost = _track("CAM-01-T00002", hits=2, score=0.35, bbox=(0.72, 0.15, 0.82, 0.45))

        self.assertEqual(gate.select("CAM-01", [person, ghost], t0), [])
        selected = gate.select("CAM-01", [person, ghost], t0 + 500_000_000)
        self.assertEqual([row["track_id"] for row in selected], ["CAM-01-T00001"])

        louder_ghost = _track("CAM-01-T00002", hits=10, score=0.99, bbox=(0.65, 0.10, 0.88, 0.70))
        selected = gate.select("CAM-01", [person, louder_ghost], t0 + 1_000_000_000)
        self.assertEqual([row["track_id"] for row in selected], ["CAM-01-T00001"])
        stats = gate.stats("CAM-01")
        self.assertEqual(stats["output_max"], 1)
        self.assertEqual(stats["violations"], 0)
        self.assertGreater(stats["suppressed"], 0)

    def test_spatial_successor_handoff_is_repeated_and_one_box_only(self) -> None:
        gate = SingleTargetBboxLockV1(
            SingleTargetLockConfigV1(acquire_updates=2, handoff_updates=2, hold_sec=1.10)
        )
        t0 = 2_000_000_000
        person = _track("CAM-04-T00010", hits=2, bbox=(0.20, 0.10, 0.50, 0.85))
        gate.select("CAM-04", [person], t0)
        gate.select("CAM-04", [person], t0 + 500_000_000)

        successor = _track("CAM-04-T00018", hits=3, bbox=(0.22, 0.10, 0.52, 0.85))
        first = gate.select("CAM-04", [successor], t0 + 900_000_000)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["track_id"], "CAM-04-T00010")

        second = gate.select("CAM-04", [successor], t0 + 1_000_000_000)
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0]["track_id"], "CAM-04-T00018")
        stats = gate.stats("CAM-04")
        self.assertEqual(stats["handoff"], 1)
        self.assertEqual(stats["output_max"], 1)

    def test_far_false_positive_cannot_steal_locked_target(self) -> None:
        gate = SingleTargetBboxLockV1(
            SingleTargetLockConfigV1(acquire_updates=2, hold_sec=1.10, release_sec=1.60)
        )
        t0 = 3_000_000_000
        person = _track("CAM-01-T00007", hits=2, bbox=(0.15, 0.10, 0.45, 0.85))
        gate.select("CAM-01", [person], t0)
        gate.select("CAM-01", [person], t0 + 500_000_000)

        far = _track("CAM-01-T00099", hits=20, score=0.99, bbox=(0.75, 0.10, 0.95, 0.60))
        held = gate.select("CAM-01", [far], t0 + 1_000_000_000)
        self.assertEqual(len(held), 1)
        self.assertEqual(held[0]["track_id"], "CAM-01-T00007")

        blank = gate.select("CAM-01", [far], t0 + 1_800_000_000)
        self.assertEqual(blank, [])
        self.assertEqual(gate.stats("CAM-01")["violations"], 0)

    def test_single_occupant_tracker_blocks_second_birth_while_primary_is_recent(self) -> None:
        tracker = V11SingleOccupantTrackerV1(
            ["CAM-01"],
            new_track_thresh=0.50,
            confirm_hits=2,
            tentative_ttl_sec=1.5,
            max_lost_sec=4.5,
            single_occupant_block_sec=4.5,
        )
        primary = [100.0, 60.0, 260.0, 360.0, 0.92]
        ghost = [500.0, 80.0, 620.0, 250.0, 0.99]
        t0 = 10_000_000_000

        first = tracker.update("CAM-01", [primary], t0)
        self.assertEqual(first.created, 1)
        second = tracker.update("CAM-01", [primary], t0 + 500_000_000)
        self.assertTrue(any(s.confirmed for s in second.snapshots))

        third = tracker.update("CAM-01", [primary, ghost], t0 + 1_000_000_000)
        self.assertEqual(third.created, 0)
        self.assertGreaterEqual(tracker.blocked_new_total("CAM-01"), 1)
        confirmed = [s.track_id for s in third.snapshots if s.confirmed]
        self.assertEqual(len(set(confirmed)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
