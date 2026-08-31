#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v11.step4_reid_gallery_v1 import GallerySampleV1, GalleryViewV1
from services.camera_v11.step4_reid_pair_scorer_v1 import score_gallery_pair_v1
from services.camera_v11.step4_reid_pair_shadow_v1 import (
    TSV_COLUMNS,
    V11GalleryPairShadowWorkerV1,
)


def basis(index: int) -> np.ndarray:
    vector = np.zeros(256, dtype=np.float32)
    vector[index] = 1.0
    return vector


def normalized_random(seed: int, count: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(count, 256)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return [row.copy() for row in matrix]


def scalar_metrics(result):
    return {
        key: value
        for key, value in result.__dict__.items()
        if key not in ("a_best_mean", "a_best_min", "b_best_mean", "b_best_min")
    }


class PairScorerTests(unittest.TestCase):
    def test_identical_galleries_produce_very_high_scores(self) -> None:
        gallery = normalized_random(10, 5)
        result = score_gallery_pair_v1(gallery, gallery)
        self.assertEqual(result.status, "OK")
        self.assertEqual(result.pair_count, 25)
        self.assertGreater(result.max_score, 0.9999)
        self.assertGreater(result.top3_mean, 0.9999)
        self.assertGreater(result.a_best_min, 0.9999)
        self.assertGreater(result.b_best_min, 0.9999)
        self.assertGreater(result.robust_score, 0.75)

    def test_permuting_gallery_order_preserves_metrics(self) -> None:
        first = normalized_random(20, 5)
        second = normalized_random(21, 6)
        baseline = score_gallery_pair_v1(first, second)
        permuted = score_gallery_pair_v1(
            [first[index] for index in (4, 1, 3, 0, 2)],
            [second[index] for index in (2, 5, 1, 4, 0, 3)],
        )
        self.assertEqual(baseline.status, "OK")
        for key, value in baseline.__dict__.items():
            other = getattr(permuted, key)
            if isinstance(value, float):
                self.assertAlmostEqual(value, other, places=12, msg=key)
            else:
                self.assertEqual(value, other, key)

    def test_swapping_galleries_is_symmetric(self) -> None:
        first = normalized_random(30, 4)
        second = normalized_random(31, 7)
        forward = score_gallery_pair_v1(first, second)
        reverse = score_gallery_pair_v1(second, first)
        for key, value in scalar_metrics(forward).items():
            other = scalar_metrics(reverse)[key]
            if isinstance(value, float):
                self.assertAlmostEqual(value, other, places=12, msg=key)
            else:
                self.assertEqual(value, other, key)
        self.assertAlmostEqual(forward.a_best_mean, reverse.b_best_mean, places=12)
        self.assertAlmostEqual(forward.a_best_min, reverse.b_best_min, places=12)
        self.assertAlmostEqual(forward.b_best_mean, reverse.a_best_mean, places=12)
        self.assertAlmostEqual(forward.b_best_min, reverse.a_best_min, places=12)

    def test_one_lucky_identical_pair_does_not_dominate_consistency(self) -> None:
        first = [basis(0), basis(1), basis(2)]
        second = [basis(0), basis(3), basis(4)]
        result = score_gallery_pair_v1(first, second)
        self.assertEqual(result.status, "OK")
        self.assertAlmostEqual(result.max_score, 1.0)
        self.assertLess(result.a_best_mean, 0.34)
        self.assertLess(result.b_best_mean, 0.34)
        self.assertAlmostEqual(result.a_best_min, 0.0)
        self.assertAlmostEqual(result.b_best_min, 0.0)
        self.assertLess(result.robust_score, 0.35)

    def test_clearly_different_galleries_score_lower(self) -> None:
        same = score_gallery_pair_v1(
            [basis(0), basis(1), basis(2)],
            [basis(0), basis(1), basis(2)],
        )
        different = score_gallery_pair_v1(
            [basis(0), basis(1), basis(2)],
            [basis(3), basis(4), basis(5)],
        )
        self.assertEqual(different.status, "OK")
        self.assertLess(different.max_score, same.max_score)
        self.assertLess(different.robust_score, same.robust_score)
        self.assertEqual(different.support_ge_050, 0)

    def test_two_sample_gallery_is_insufficient(self) -> None:
        result = score_gallery_pair_v1(
            [basis(0), basis(1)], [basis(0), basis(1), basis(2)]
        )
        self.assertEqual(result.status, "INSUFFICIENT")
        self.assertEqual(result.pair_count, 6)
        self.assertIsNone(result.robust_score)

    def test_nan_and_inf_embeddings_are_rejected(self) -> None:
        for bad in (float("nan"), float("inf"), float("-inf")):
            gallery = [basis(0), basis(1), basis(2)]
            gallery[1][7] = bad
            with self.subTest(bad=bad):
                result = score_gallery_pair_v1(gallery, normalized_random(40, 3))
                self.assertEqual(result.status, "INVALID")
                self.assertIn("non-finite", result.reason)

    def test_input_galleries_remain_unchanged(self) -> None:
        first = normalized_random(50, 5)
        second = normalized_random(51, 6)
        first_before = [row.copy() for row in first]
        second_before = [row.copy() for row in second]
        score_gallery_pair_v1(first, second)
        for before, after in zip(first_before, first, strict=True):
            np.testing.assert_array_equal(before, after)
        for before, after in zip(second_before, second, strict=True):
            np.testing.assert_array_equal(before, after)

    def test_eight_by_eight_capacity_and_support_counts(self) -> None:
        gallery = normalized_random(60, 8)
        result = score_gallery_pair_v1(gallery, gallery)
        self.assertEqual(result.status, "OK")
        self.assertEqual(result.pair_count, 64)
        self.assertEqual(result.support_ge_080, 8)
        self.assertIsNotNone(result.top5_mean)

    def test_results_are_deterministic(self) -> None:
        first = normalized_random(70, 8)
        second = normalized_random(71, 8)
        results = [score_gallery_pair_v1(first, second) for _ in range(5)]
        self.assertTrue(all(result == results[0] for result in results[1:]))

    def test_fixed_robust_formula(self) -> None:
        result = score_gallery_pair_v1(
            normalized_random(80, 5), normalized_random(81, 5)
        )
        expected = (
            0.40 * result.top3_mean
            + 0.25 * result.median_of_best_matches
            + 0.20 * result.p75_score
            + 0.15 * result.max_score
        )
        self.assertAlmostEqual(result.robust_score, expected, places=15)


def view(camera: str, track: str, start_sequence: int) -> GalleryViewV1:
    samples = []
    now_ns = time.monotonic_ns()
    for offset, embedding in enumerate(normalized_random(start_sequence, 3)):
        embedding.setflags(write=False)
        samples.append(
            GallerySampleV1(
                camera_id=camera,
                local_track_id=track,
                timestamp_ns=now_ns,
                embedding=embedding,
                quality_score=0.7,
                detector_confidence=0.9,
                bbox_xyxy=(1.0, 2.0, 30.0, 90.0),
                sample_sequence=start_sequence + offset,
            )
        )
    return GalleryViewV1(camera, track, now_ns, tuple(samples))


class ShadowWorkerTests(unittest.TestCase):
    def test_shadow_worker_scores_only_cross_camera_and_writes_exact_tsv(self) -> None:
        views = (
            view("CAM-01", "CAM-01-T1", 100),
            view("CAM-04", "CAM-04-T1", 200),
            view("CAM-02", "CAM-02-T1", 300),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.tsv"
            worker = V11GalleryPairShadowWorkerV1(
                lambda: views,
                {"CAM-01": "Devs", "CAM-04": "Devs", "CAM-02": "Entrance"},
                tsv_path=path,
                max_candidates=24,
            )
            worker.start()
            worker.notify()
            deadline = time.monotonic() + 2.0
            while worker.snapshot()["pairs_scored"] < 3 and time.monotonic() < deadline:
                time.sleep(0.01)
            worker.close()
            row = worker.snapshot()
            self.assertEqual(row["pairs_considered"], 3)
            self.assertEqual(row["pairs_scored"], 3)
            self.assertEqual(row["pairs_insufficient"], 0)
            self.assertEqual(row["pairs_invalid"], 0)
            self.assertEqual(row["same_room_pairs"], 1)
            self.assertEqual(row["different_room_pairs"], 2)
            self.assertEqual(row["worker_errors"], 0)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(tuple(lines[0].split("\t")), TSV_COLUMNS)
            self.assertEqual(len(lines), 4)
            self.assertTrue(all(len(line.split("\t")) == len(TSV_COLUMNS) for line in lines))
            self.assertNotIn("embedding", "\n".join(lines).lower())


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
