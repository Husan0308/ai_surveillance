#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.camera_v11.step4_reid_gallery_v1 import GallerySampleV1, GalleryViewV1
from services.camera_v11.step9_direct_global_reid_v1 import (
    DirectGlobalReIDConfigV1,
    DirectGlobalReIDResolverV1,
)


DIM = 256


def unit(index: int, *, mix: int | None = None, amount: float = 0.0) -> np.ndarray:
    vector = np.zeros(DIM, dtype=np.float32)
    vector[index] = 1.0
    if mix is not None and amount:
        vector[mix] = float(amount)
    vector /= np.linalg.norm(vector)
    vector.setflags(write=False)
    return vector


def view(camera: str, track: str, person: int, count: int, seq_start: int) -> GalleryViewV1:
    samples = []
    for offset in range(count):
        embedding = unit(person * 4, mix=person * 4 + 1, amount=0.02 * (offset + 1))
        samples.append(
            GallerySampleV1(
                camera_id=camera,
                local_track_id=track,
                timestamp_ns=1_000_000_000 + offset,
                embedding=embedding,
                quality_score=0.9,
                detector_confidence=0.95,
                bbox_xyxy=(10.0, 20.0, 100.0, 220.0),
                sample_sequence=seq_start + offset,
            )
        )
    return GalleryViewV1(camera, track, 2_000_000_000 + count, tuple(samples))


class DirectGlobalReIDV1Tests(unittest.TestCase):
    def make(self) -> DirectGlobalReIDResolverV1:
        return DirectGlobalReIDResolverV1(
            DirectGlobalReIDConfigV1(
                min_robust_score=0.70,
                min_top3_mean=0.70,
                min_median_best=0.70,
                min_support_ge_070=3,
                min_margin=0.05,
                confirm_evidence=2,
                new_identity_evidence=2,
            )
        )

    def test_same_person_two_cameras_gets_one_gid(self) -> None:
        resolver = self.make()
        active = {"CAM-01": frozenset({"A1"}), "CAM-04": frozenset({"A4"})}
        resolver.resolve((view("CAM-01", "A1", 0, 3, 1), view("CAM-04", "A4", 0, 3, 101)), active, now_ns=3_000_000_000)
        resolver.resolve((view("CAM-01", "A1", 0, 4, 1), view("CAM-04", "A4", 0, 4, 101)), active, now_ns=4_000_000_000)
        gid1 = resolver.global_for_track("CAM-01", "A1")
        gid4 = resolver.global_for_track("CAM-04", "A4")
        self.assertIsNotNone(gid1)
        self.assertEqual(gid1, gid4)
        self.assertEqual(resolver.snapshot()["global_ids"], 1)

    def test_different_people_cross_camera_do_not_share_gid(self) -> None:
        resolver = self.make()
        active = {"CAM-01": frozenset({"A1"}), "CAM-04": frozenset({"B4"})}
        resolver.resolve((view("CAM-01", "A1", 0, 3, 1), view("CAM-04", "B4", 1, 3, 101)), active, now_ns=3_000_000_000)
        resolver.resolve((view("CAM-01", "A1", 0, 4, 1), view("CAM-04", "B4", 1, 4, 101)), active, now_ns=4_000_000_000)
        gid1 = resolver.global_for_track("CAM-01", "A1")
        gid4 = resolver.global_for_track("CAM-04", "B4")
        self.assertIsNotNone(gid1)
        self.assertIsNotNone(gid4)
        self.assertNotEqual(gid1, gid4)
        self.assertEqual(resolver.snapshot()["global_ids"], 2)

    def test_local_track_switch_reuses_existing_gid(self) -> None:
        resolver = self.make()
        active = {"CAM-01": frozenset({"A1"}), "CAM-04": frozenset()}
        resolver.resolve((view("CAM-01", "A1", 0, 3, 1),), active, now_ns=3_000_000_000)
        resolver.resolve((view("CAM-01", "A1", 0, 4, 1),), active, now_ns=4_000_000_000)
        original = resolver.global_for_track("CAM-01", "A1")
        self.assertIsNotNone(original)

        active = {"CAM-01": frozenset({"A2"}), "CAM-04": frozenset()}
        resolver.resolve((view("CAM-01", "A2", 0, 3, 201),), active, now_ns=5_000_000_000)
        resolver.resolve((view("CAM-01", "A2", 0, 4, 201),), active, now_ns=6_000_000_000)
        self.assertEqual(resolver.global_for_track("CAM-01", "A2"), original)
        self.assertEqual(resolver.snapshot()["global_ids"], 1)
        self.assertEqual(resolver.snapshot()["reassociated_total"], 1)

    def test_two_simultaneous_same_camera_people_never_share_gid(self) -> None:
        resolver = self.make()
        active = {"CAM-01": frozenset({"A1", "B1"}), "CAM-04": frozenset()}
        first = (view("CAM-01", "A1", 0, 3, 1), view("CAM-01", "B1", 1, 3, 101))
        second = (view("CAM-01", "A1", 0, 4, 1), view("CAM-01", "B1", 1, 4, 101))
        resolver.resolve(first, active, now_ns=3_000_000_000)
        resolver.resolve(second, active, now_ns=4_000_000_000)
        gid_a = resolver.global_for_track("CAM-01", "A1")
        gid_b = resolver.global_for_track("CAM-01", "B1")
        self.assertIsNotNone(gid_a)
        self.assertIsNotNone(gid_b)
        self.assertNotEqual(gid_a, gid_b)

    def test_ambiguous_existing_match_stays_pending(self) -> None:
        resolver = self.make()
        active = {"CAM-01": frozenset({"A1", "B1"}), "CAM-04": frozenset()}
        resolver.resolve((view("CAM-01", "A1", 0, 3, 1), view("CAM-01", "B1", 1, 3, 101)), active, now_ns=3_000_000_000)
        resolver.resolve((view("CAM-01", "A1", 0, 4, 1), view("CAM-01", "B1", 1, 4, 101)), active, now_ns=4_000_000_000)

        vector = np.zeros(DIM, dtype=np.float32)
        vector[0] = 1.0
        vector[4] = 1.0
        vector /= np.linalg.norm(vector)
        vector.setflags(write=False)
        samples = tuple(
            GallerySampleV1("CAM-04", "X4", 5_000_000_000 + i, vector, 0.9, 0.95, (0, 0, 1, 2), 300 + i)
            for i in range(4)
        )
        x4 = GalleryViewV1("CAM-04", "X4", 6_000_000_000, samples)
        active = {"CAM-01": frozenset(), "CAM-04": frozenset({"X4"})}
        resolver.resolve((x4,), active, now_ns=6_000_000_000)
        self.assertIsNone(resolver.global_for_track("CAM-04", "X4"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
