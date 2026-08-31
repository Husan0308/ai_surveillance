#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v11.step4_reid_gallery_v1 import DiverseReIDGalleryV1
from services.camera_v11.step4_reid_scheduler_v1 import (
    ReIDCandidateV1,
    V11ReIDSchedulerV1,
)


def basis(index: int, scale: float = 1.0) -> np.ndarray:
    vector = np.zeros(256, dtype=np.float32)
    vector[index] = float(scale)
    return vector


def add(
    gallery: DiverseReIDGalleryV1,
    embedding: np.ndarray,
    *,
    camera: str = "CAM-01",
    track: str = "CAM-01-T000001",
    sequence: int = 1,
    quality: float = 0.50,
):
    return gallery.update(
        camera_id=camera,
        local_track_id=track,
        timestamp_ns=sequence * 1_000_000_000,
        embedding=embedding,
        quality_score=quality,
        detector_confidence=0.88,
        bbox_xyxy=(10.0, 20.0, 90.0, 220.0),
    )


def bootstrap(gallery: DiverseReIDGalleryV1, *, first_quality: float = 0.50) -> None:
    add(gallery, basis(0), sequence=1, quality=first_quality)
    add(gallery, basis(1), sequence=2, quality=0.50)
    add(gallery, basis(2), sequence=3, quality=0.50)


class GalleryTests(unittest.TestCase):
    def test_first_three_embeddings_bootstrap_even_when_identical(self) -> None:
        gallery = DiverseReIDGalleryV1()
        decisions = [add(gallery, basis(0), sequence=index) for index in range(1, 4)]
        self.assertEqual([row.action for row in decisions], ["bootstrap_add"] * 3)
        self.assertEqual(len(gallery.samples_for("CAM-01", "CAM-01-T000001")), 3)
        self.assertEqual(gallery.snapshot()["gallery_bootstrap_add"], 3)

    def test_gallery_never_exceeds_eight(self) -> None:
        gallery = DiverseReIDGalleryV1()
        for index in range(20):
            add(gallery, basis(index), sequence=index + 1, quality=0.35 + index * 0.01)
            self.assertLessEqual(
                len(gallery.samples_for("CAM-01", "CAM-01-T000001")), 8
            )
        self.assertEqual(len(gallery.samples_for("CAM-01", "CAM-01-T000001")), 8)
        self.assertGreater(gallery.snapshot()["gallery_full_reject_or_replace"], 0)

    def test_exact_embedding_after_bootstrap_is_dropped(self) -> None:
        gallery = DiverseReIDGalleryV1()
        bootstrap(gallery)
        decision = add(gallery, basis(0), sequence=4, quality=0.55)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.action, "duplicate_drop")
        self.assertEqual(len(gallery.samples_for("CAM-01", "CAM-01-T000001")), 3)

    def test_cosine_at_least_point_975_is_near_duplicate(self) -> None:
        gallery = DiverseReIDGalleryV1()
        bootstrap(gallery)
        cosine = 0.980
        vector = cosine * basis(0) + np.sqrt(1.0 - cosine**2) * basis(3)
        decision = add(gallery, vector, sequence=4, quality=0.55)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.action, "duplicate_drop")
        self.assertGreaterEqual(decision.max_cosine, 0.975)

    def test_better_quality_near_duplicate_replaces_worse_sample(self) -> None:
        gallery = DiverseReIDGalleryV1()
        bootstrap(gallery, first_quality=0.25)
        old_sequence = gallery.samples_for("CAM-01", "CAM-01-T000001")[0].sample_sequence
        decision = add(gallery, basis(0), sequence=4, quality=0.50)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.action, "quality_replace")
        self.assertEqual(decision.evicted_sequence, old_sequence)
        self.assertEqual(len(gallery.samples_for("CAM-01", "CAM-01-T000001")), 3)

    def test_sufficiently_different_embedding_is_added(self) -> None:
        gallery = DiverseReIDGalleryV1()
        bootstrap(gallery)
        decision = add(gallery, basis(3), sequence=4)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.action, "diverse_add")
        self.assertEqual(len(gallery.samples_for("CAM-01", "CAM-01-T000001")), 4)

    def test_embedding_is_l2_normalized_and_metadata_is_preserved(self) -> None:
        gallery = DiverseReIDGalleryV1()
        add(gallery, basis(4, scale=9.0), sequence=7, quality=0.72)
        sample = gallery.samples_for("CAM-01", "CAM-01-T000001")[0]
        self.assertAlmostEqual(float(np.linalg.norm(sample.embedding)), 1.0, places=6)
        self.assertEqual(sample.camera_id, "CAM-01")
        self.assertEqual(sample.local_track_id, "CAM-01-T000001")
        self.assertEqual(sample.timestamp_ns, 7_000_000_000)
        self.assertEqual(sample.bbox_xyxy, (10.0, 20.0, 90.0, 220.0))
        self.assertAlmostEqual(sample.quality_score, 0.72)
        self.assertAlmostEqual(sample.detector_confidence, 0.88)
        self.assertGreater(sample.sample_sequence, 0)

    def test_nan_and_inf_embeddings_are_rejected(self) -> None:
        gallery = DiverseReIDGalleryV1()
        for value in (float("nan"), float("inf"), float("-inf")):
            vector = basis(0)
            vector[4] = value
            with self.subTest(value=value):
                decision = add(gallery, vector)
                self.assertFalse(decision.accepted)
                self.assertEqual(decision.action, "invalid")
        self.assertEqual(gallery.snapshot()["gallery_tracks"], 0)
        self.assertEqual(gallery.snapshot()["gallery_invalid_reject"], 3)

    def test_camera_and_local_track_keys_are_isolated(self) -> None:
        gallery = DiverseReIDGalleryV1()
        add(gallery, basis(0), camera="CAM-01", track="CAM-01-T000001")
        add(gallery, basis(1), camera="CAM-01", track="CAM-01-T000002")
        add(gallery, basis(2), camera="CAM-04", track="CAM-04-T000001")
        self.assertEqual(gallery.snapshot()["gallery_tracks"], 3)
        self.assertEqual(
            gallery.samples_for("CAM-01", "CAM-01-T000001")[0].embedding.argmax(), 0
        )
        self.assertEqual(
            gallery.samples_for("CAM-01", "CAM-01-T000002")[0].embedding.argmax(), 1
        )
        self.assertEqual(
            gallery.samples_for("CAM-04", "CAM-04-T000001")[0].embedding.argmax(), 2
        )

    def test_terminated_and_expired_track_cleanup(self) -> None:
        gallery = DiverseReIDGalleryV1(expiry_sec=2.0)
        add(gallery, basis(0), track="CAM-01-T000001", sequence=1)
        add(gallery, basis(1), track="CAM-01-T000002", sequence=1)
        removed = gallery.touch_active(
            "CAM-01", frozenset({"CAM-01-T000001"}), 2_000_000_000
        )
        self.assertEqual(removed, 1)
        self.assertEqual(gallery.snapshot()["gallery_tracks"], 1)
        self.assertEqual(gallery.cleanup_expired(4_100_000_000), 1)
        self.assertEqual(gallery.snapshot()["gallery_tracks"], 0)

    def test_full_gallery_update_is_deterministic(self) -> None:
        rng = np.random.default_rng(1104)
        rows = []
        for index in range(18):
            vector = rng.normal(size=256).astype(np.float32)
            rows.append((vector, 0.30 + (index % 7) * 0.07))
        galleries = [DiverseReIDGalleryV1(), DiverseReIDGalleryV1()]
        for gallery in galleries:
            for index, (vector, quality) in enumerate(rows, start=1):
                add(gallery, vector.copy(), sequence=index, quality=quality)
        left = galleries[0].samples_for("CAM-01", "CAM-01-T000001")
        right = galleries[1].samples_for("CAM-01", "CAM-01-T000001")
        self.assertEqual([row.sample_sequence for row in left], [row.sample_sequence for row in right])
        for first, second in zip(left, right, strict=True):
            np.testing.assert_array_equal(first.embedding, second.embedding)
        left_stats = galleries[0].snapshot()
        right_stats = galleries[1].snapshot()
        for timing in ("gallery_update_p50_ms", "gallery_update_p95_ms"):
            left_stats.pop(timing)
            right_stats.pop(timing)
        self.assertEqual(left_stats, right_stats)


class _FakeReIDClient:
    def embed_crops(self, crops: list[np.ndarray]):
        rows = []
        for index, _crop in enumerate(crops):
            rows.append(basis(index))
        return np.stack(rows), {"inference_ms": 0.25}

    def close(self) -> None:
        return


class SchedulerTests(unittest.TestCase):
    def candidate(self, track: str, captured_ns: int) -> ReIDCandidateV1:
        return ReIDCandidateV1(
            camera_id="CAM-01",
            local_track_id=track,
            captured_ns=captured_ns,
            bbox_xyxy=(1.0, 2.0, 40.0, 100.0),
            detector_confidence=0.9,
            quality_score=0.7,
            crop_bgr=np.full((96, 40, 3), 127, dtype=np.uint8),
        )

    def test_pending_is_keyed_latest_only_and_bounded(self) -> None:
        scheduler = V11ReIDSchedulerV1(
            lambda _result: None, max_pending=2, client_factory=_FakeReIDClient
        )
        self.assertTrue(scheduler.submit(self.candidate("T1", 1)))
        self.assertTrue(scheduler.submit(self.candidate("T1", 2)))
        self.assertTrue(scheduler.submit(self.candidate("T2", 3)))
        row = scheduler.snapshot()
        self.assertEqual(row["reid_pending"], 2)
        self.assertEqual(row["reid_replaced_pending"], 1)
        scheduler.close(drain=False)

    def test_worker_callback_is_asynchronous_and_embedding_is_normalized(self) -> None:
        event = threading.Event()
        output = []

        def consume(result) -> None:
            output.append(result)
            event.set()

        scheduler = V11ReIDSchedulerV1(
            consume, max_batch=1, max_wait_ms=0.0, client_factory=_FakeReIDClient
        )
        scheduler.start()
        self.assertTrue(scheduler.submit(self.candidate("T1", time.monotonic_ns())))
        self.assertTrue(event.wait(timeout=2.0))
        scheduler.close(drain=True)
        self.assertEqual(len(output), 1)
        self.assertAlmostEqual(float(np.linalg.norm(output[0].embedding)), 1.0, places=6)
        row = scheduler.snapshot()
        self.assertEqual(row["reid_completed"], 1)
        self.assertEqual(row["reid_pending"], 0)
        self.assertEqual(row["reid_worker_errors"], 0)


class FrozenGuardTests(unittest.TestCase):
    def test_frozen_step123_guard(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_camera_v11_frozen_step123_guard.py")],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("V11_FROZEN_STEP123_GUARD RESULT=PASS", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
