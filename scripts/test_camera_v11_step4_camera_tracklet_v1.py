#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v11.step4_reid_camera_tracklet_v1 import (
    CameraTrackletConfigV1,
    CameraTrackletContinuityV1,
)
from services.camera_v11.step4_reid_gallery_v1 import GallerySampleV1, GalleryViewV1
from services.camera_v11.step4_reid_same_room_matcher_v1 import (
    MATCH_PROPOSED,
    SameRoomPairDiagnosticV1,
)

NS = 1_000_000_000


def unit(index: int) -> np.ndarray:
    row = np.zeros(256, dtype=np.float32)
    row[index] = 1.0
    row.setflags(write=False)
    return row


def mixed(first: int, second: int, weight: float = 0.08) -> np.ndarray:
    row = unit(first).copy()
    row.setflags(write=True)
    row[second] = weight
    row /= np.linalg.norm(row)
    row.setflags(write=False)
    return row


def view(camera: str, track: str, start_ns: int, embedding: np.ndarray) -> GalleryViewV1:
    samples = []
    for offset, sequence in enumerate((1, 2, 3)):
        timestamp = int(start_ns + offset * 100_000_000)
        samples.append(
            GallerySampleV1(
                camera_id=camera,
                local_track_id=track,
                timestamp_ns=timestamp,
                embedding=embedding,
                quality_score=0.9,
                detector_confidence=0.9,
                bbox_xyxy=(10.0, 10.0, 100.0, 200.0),
                sample_sequence=sequence,
            )
        )
    return GalleryViewV1(
        camera_id=camera,
        local_track_id=track,
        last_seen_ns=samples[-1].timestamp_ns,
        samples=tuple(samples),
    )


def proposal(track_a: str, track_b: str) -> SameRoomPairDiagnosticV1:
    return SameRoomPairDiagnosticV1(
        room="Devs",
        camera_a="CAM-01",
        track_a=track_a,
        camera_b="CAM-04",
        track_b=track_b,
        samples_a=3,
        samples_b=3,
        robust_score=0.95,
        max_score=0.99,
        top3_mean=0.96,
        median=0.94,
        a_best_mean=0.95,
        a_best_min=0.93,
        b_best_mean=0.95,
        b_best_min=0.93,
        support_ge_050=9,
        support_ge_055=9,
        support_ge_060=9,
        support_ge_065=9,
        support_ge_070=9,
        support_ge_075=9,
        support_ge_080=9,
        row_best=0.95,
        row_second=None,
        row_margin=None,
        column_best=0.95,
        column_second=None,
        column_margin=None,
        reciprocal=True,
        assigned=True,
        status=MATCH_PROPOSED,
        reason="test",
    )


class CameraTrackletContinuityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.views: list[GalleryViewV1] = []
        self.active: dict[str, frozenset[str]] = {
            "CAM-01": frozenset(),
            "CAM-04": frozenset(),
        }
        self.resolver = CameraTrackletContinuityV1(
            lambda: tuple(self.views),
            lambda: dict(self.active),
            config=CameraTrackletConfigV1(
                min_robust_score=0.72,
                min_margin=0.05,
                min_support_ge_065=3,
                confirm_cycles=2,
                active_overlap_grace_cycles=3,
            ),
        )

    def test_initial_tracks_allocate_stable_camera_tracklets(self) -> None:
        self.views = [
            view("CAM-01", "T1", 1 * NS, unit(0)),
            view("CAM-04", "T7", 1 * NS, unit(1)),
        ]
        self.active["CAM-01"] = frozenset({"T1"})
        self.active["CAM-04"] = frozenset({"T7"})
        self.resolver.refresh(1, 2 * NS)
        self.assertEqual(self.resolver.canonical_track_id("CAM-01", "T1"), "CAM-01-CT00001")
        self.assertEqual(self.resolver.canonical_track_id("CAM-04", "T7"), "CAM-04-CT00001")

    def test_recent_lost_successor_stitches_after_two_votes(self) -> None:
        old = view("CAM-01", "T1", 1 * NS, mixed(0, 1))
        self.views = [old]
        self.active["CAM-01"] = frozenset({"T1"})
        self.resolver.refresh(1, 2 * NS)
        stable = self.resolver.canonical_track_id("CAM-01", "T1")
        self.assertIsNotNone(stable)
        new = view("CAM-01", "T2", 2 * NS, mixed(0, 1))
        self.views = [old, new]
        self.active["CAM-01"] = frozenset({"T2"})
        self.resolver.refresh(2, 3 * NS)
        self.assertIsNone(self.resolver.canonical_track_id("CAM-01", "T2"))
        self.resolver.refresh(3, 4 * NS)
        self.assertEqual(self.resolver.canonical_track_id("CAM-01", "T2"), stable)
        self.assertEqual(self.resolver.snapshot()["stitched_total"], 1)

    def test_active_predecessor_race_is_deferred_then_stitched(self) -> None:
        old = view("CAM-01", "T1", 1 * NS, unit(3))
        self.views = [old]
        self.active["CAM-01"] = frozenset({"T1"})
        self.resolver.refresh(1, 2 * NS)
        first = self.resolver.canonical_track_id("CAM-01", "T1")
        new = view("CAM-01", "T2", 1 * NS + 400_000_000, unit(3))
        self.views = [old, new]
        self.active["CAM-01"] = frozenset({"T1", "T2"})
        self.resolver.refresh(2, 2 * NS + 400_000_000)
        self.assertIsNone(self.resolver.canonical_track_id("CAM-01", "T2"))
        self.active["CAM-01"] = frozenset({"T2"})
        self.resolver.refresh(3, 3 * NS)
        self.assertIsNone(self.resolver.canonical_track_id("CAM-01", "T2"))
        self.resolver.refresh(4, 4 * NS)
        self.assertEqual(self.resolver.canonical_track_id("CAM-01", "T2"), first)
        self.assertEqual(self.resolver.snapshot()["stitched_total"], 1)

    def test_simultaneously_visible_tracks_never_stitch_and_allocate_after_grace(self) -> None:
        old = view("CAM-01", "T1", 1 * NS, unit(3))
        self.views = [old]
        self.active["CAM-01"] = frozenset({"T1"})
        self.resolver.refresh(1, 2 * NS)
        first = self.resolver.canonical_track_id("CAM-01", "T1")
        new = view("CAM-01", "T2", 1 * NS + 300_000_000, unit(3))
        self.views = [old, new]
        self.active["CAM-01"] = frozenset({"T1", "T2"})
        for cycle in (2, 3, 4):
            self.resolver.refresh(cycle, cycle * NS)
            self.assertIsNone(self.resolver.canonical_track_id("CAM-01", "T2"))
        self.resolver.refresh(5, 5 * NS)
        second = self.resolver.canonical_track_id("CAM-01", "T2")
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first, second)
        self.assertEqual(self.resolver.snapshot()["stitched_total"], 0)
        self.assertEqual(
            self.resolver.snapshot()["active_overlap_fallback_allocated_total"], 1
        )

    def test_weak_recent_successor_stays_unresolved(self) -> None:
        old = view("CAM-01", "T1", 1 * NS, unit(4))
        self.views = [old]
        self.active["CAM-01"] = frozenset({"T1"})
        self.resolver.refresh(1, 2 * NS)
        new = view("CAM-01", "T2", 2 * NS, unit(5))
        self.views = [old, new]
        self.active["CAM-01"] = frozenset({"T2"})
        self.resolver.refresh(2, 3 * NS)
        self.resolver.refresh(3, 4 * NS)
        self.assertIsNone(self.resolver.canonical_track_id("CAM-01", "T2"))
        self.assertGreater(self.resolver.snapshot()["low_score_total"], 0)

    def test_canonicalized_cross_camera_proposal_uses_stable_ids(self) -> None:
        self.views = [
            view("CAM-01", "T1", 1 * NS, unit(7)),
            view("CAM-04", "T8", 1 * NS, unit(8)),
        ]
        self.active["CAM-01"] = frozenset({"T1"})
        self.active["CAM-04"] = frozenset({"T8"})
        self.resolver.refresh(1, 2 * NS)
        row = self.resolver.canonicalize_proposal(proposal("T1", "T8"))
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row.track_a, "CAM-01-CT00001")
        self.assertEqual(row.track_b, "CAM-04-CT00001")
        self.assertEqual(row.status, MATCH_PROPOSED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
