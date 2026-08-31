#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v11.step4_reid_gallery_v1 import GallerySampleV1, GalleryViewV1
from services.camera_v11.step4_reid_pair_scorer_v1 import score_gallery_pair_v1
from services.camera_v11.step4_reid_same_room_evidence_padded_v1 import (
    score_gallery_matrix_step3_padded_exact_v1,
)


def view(camera: str, track: str, count: int, seed: int) -> GalleryViewV1:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(count, 256)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    now_ns = time.monotonic_ns()
    return GalleryViewV1(
        camera,
        track,
        now_ns,
        tuple(
            GallerySampleV1(
                camera,
                track,
                now_ns,
                row.copy(),
                0.8,
                0.9,
                (10.0, 10.0, 80.0, 190.0),
                seed * 100 + index,
            )
            for index, row in enumerate(matrix)
        ),
    )


class BatchedStep3EvidenceTests(unittest.TestCase):
    def test_random_variable_size_matrix_matches_authoritative_step3(self) -> None:
        rows = [view("CAM-01", f"A{index}", count, 100 + index) for index, count in enumerate((3, 4, 8))]
        columns = [view("CAM-04", f"B{index}", count, 200 + index) for index, count in enumerate((3, 5, 8))]
        batched = score_gallery_matrix_step3_padded_exact_v1(rows, columns)
        self.assertEqual(len(batched), 9)
        for row_index, row in enumerate(rows):
            for column_index, column in enumerate(columns):
                expected = score_gallery_pair_v1(
                    [sample.embedding for sample in row.samples],
                    [sample.embedding for sample in column.samples],
                )
                actual = batched[(row_index, column_index)]
                for field, expected_value in expected.__dict__.items():
                    actual_value = getattr(actual, field)
                    if isinstance(expected_value, float):
                        self.assertAlmostEqual(expected_value, actual_value, places=12, msg=field)
                    else:
                        self.assertEqual(expected_value, actual_value, field)

    def test_insufficient_status_and_pair_count_match_step3(self) -> None:
        rows = [view("CAM-01", "A1", 2, 300)]
        columns = [view("CAM-04", "B1", 8, 301)]
        actual = score_gallery_matrix_step3_padded_exact_v1(rows, columns)[(0, 0)]
        expected = score_gallery_pair_v1(
            [sample.embedding for sample in rows[0].samples],
            [sample.embedding for sample in columns[0].samples],
        )
        self.assertEqual(actual, expected)

    def test_nonfinite_status_matches_step3_and_is_never_valid(self) -> None:
        rows = [view("CAM-01", "A1", 3, 400)]
        columns = [view("CAM-04", "B1", 3, 401)]
        rows[0].samples[1].embedding[4] = np.nan
        actual = score_gallery_matrix_step3_padded_exact_v1(rows, columns)[(0, 0)]
        expected = score_gallery_pair_v1(
            [sample.embedding for sample in rows[0].samples],
            [sample.embedding for sample in columns[0].samples],
        )
        self.assertEqual(actual.status, "INVALID")
        self.assertEqual(actual.status, expected.status)
        self.assertEqual(actual.pair_count, expected.pair_count)
        self.assertIn("non-finite", actual.reason)

    def test_fixed_robust_formula_is_unchanged(self) -> None:
        result = score_gallery_matrix_step3_padded_exact_v1(
            [view("CAM-01", "A1", 8, 500)],
            [view("CAM-04", "B1", 8, 501)],
        )[(0, 0)]
        expected = (
            0.40 * result.top3_mean
            + 0.25 * result.median_of_best_matches
            + 0.20 * result.p75_score
            + 0.15 * result.max_score
        )
        self.assertAlmostEqual(result.robust_score, expected, places=15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
