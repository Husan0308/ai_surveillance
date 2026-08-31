#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
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
from services.camera_v11.step4_reid_pair_scorer_v1 import GalleryPairScoreV1
from services.camera_v11.step4_reid_same_room_matcher_v1 import (
    INSUFFICIENT,
    INVALID,
    LOW_MARGIN,
    MATCH_PROPOSED,
    NON_RECIPROCAL,
    STALE,
    SameRoomMatcherConfigV1,
    match_same_room_camera_pair_v1,
)
from services.camera_v11.step4_reid_same_room_shadow_v1 import (
    TSV_COLUMNS,
    V11SameRoomMatcherShadowWorkerV1,
)


ROOMS = {"CAM-01": "Devs", "CAM-04": "Devs", "CAM-02": "Entrance"}


def basis(index: int) -> np.ndarray:
    value = np.zeros(256, dtype=np.float32)
    value[index] = 1.0
    return value


def view(
    camera: str,
    track: str,
    *,
    count: int = 3,
    last_seen_ns: int | None = None,
    first_basis: int = 0,
) -> GalleryViewV1:
    timestamp_ns = time.monotonic_ns() if last_seen_ns is None else int(last_seen_ns)
    samples = tuple(
        GallerySampleV1(
            camera_id=camera,
            local_track_id=track,
            timestamp_ns=timestamp_ns,
            embedding=basis((first_basis + offset) % 256),
            quality_score=0.8,
            detector_confidence=0.9,
            bbox_xyxy=(10.0, 10.0, 80.0, 190.0),
            sample_sequence=offset + 1,
        )
        for offset in range(count)
    )
    return GalleryViewV1(camera, track, timestamp_ns, samples)


def score(value: float, status: str = "OK") -> GalleryPairScoreV1:
    if status != "OK":
        return GalleryPairScoreV1(status=status, pair_count=6, reason=status.lower())
    return GalleryPairScoreV1(
        status="OK",
        pair_count=9,
        max_score=value,
        top2_mean=value,
        top3_mean=value,
        top5_mean=value,
        median_score=value,
        mean_score=value,
        p75_score=value,
        p90_score=value,
        support_ge_050=int(value >= 0.50) * 9,
        support_ge_055=int(value >= 0.55) * 9,
        support_ge_060=int(value >= 0.60) * 9,
        support_ge_065=int(value >= 0.65) * 9,
        support_ge_070=int(value >= 0.70) * 9,
        support_ge_075=int(value >= 0.75) * 9,
        support_ge_080=int(value >= 0.80) * 9,
        a_best_mean=value,
        a_best_min=value,
        b_best_mean=value,
        b_best_min=value,
        median_of_best_matches=value,
        robust_score=value,
    )


def matrix_scorer(values: dict[tuple[str, str], float | str]):
    def scorer(first: GalleryViewV1, second: GalleryViewV1) -> GalleryPairScoreV1:
        value = values[(first.local_track_id, second.local_track_id)]
        return score(0.0, value) if isinstance(value, str) else score(value)

    return scorer


def proposals(result) -> set[tuple[str, str]]:
    return {(row.track_a, row.track_b) for row in result.proposals}


class SameRoomMatcherTests(unittest.TestCase):
    def run_matrix(self, rows, columns, values, **kwargs):
        return match_same_room_camera_pair_v1(
            rows,
            columns,
            ROOMS,
            pair_scorer=matrix_scorer(values),
            **kwargs,
        )

    def test_1x1_valid_matrix_produces_one_reciprocal_proposal(self) -> None:
        result = self.run_matrix(
            [view("CAM-01", "A1")], [view("CAM-04", "B1")], {("A1", "B1"): 0.8}
        )
        self.assertEqual(proposals(result), {("A1", "B1")})
        row = result.proposals[0]
        self.assertTrue(row.reciprocal)
        self.assertTrue(row.assigned)
        self.assertIsNone(row.row_second)
        self.assertIsNone(row.row_margin)

    def test_clear_2x2_diagonal_produces_two_matches(self) -> None:
        result = self.run_matrix(
            [view("CAM-01", "A1"), view("CAM-01", "A2")],
            [view("CAM-04", "B1"), view("CAM-04", "B2")],
            {("A1", "B1"): 0.90, ("A1", "B2"): 0.20, ("A2", "B1"): 0.10, ("A2", "B2"): 0.80},
        )
        self.assertEqual(proposals(result), {("A1", "B1"), ("A2", "B2")})

    def test_two_rows_preferring_same_column_never_duplicate_assignment(self) -> None:
        result = self.run_matrix(
            [view("CAM-01", "A1"), view("CAM-01", "A2")],
            [view("CAM-04", "B1"), view("CAM-04", "B2")],
            {("A1", "B1"): 0.90, ("A1", "B2"): 0.10, ("A2", "B1"): 0.80, ("A2", "B2"): 0.20},
        )
        self.assertEqual(proposals(result), {("A1", "B1")})
        self.assertEqual(len({row.track_b for row in result.proposals}), len(result.proposals))

    def test_nonreciprocal_pair_is_rejected(self) -> None:
        result = self.run_matrix(
            [view("CAM-01", "A1"), view("CAM-01", "A2")],
            [view("CAM-04", "B1")],
            {("A1", "B1"): 0.80, ("A2", "B1"): 0.90},
        )
        rows = {(row.track_a, row.track_b): row for row in result.diagnostics}
        self.assertEqual(rows[("A1", "B1")].status, NON_RECIPROCAL)
        self.assertFalse(rows[("A1", "B1")].assigned)

    def test_row_and_column_permutation_preserves_semantic_assignment(self) -> None:
        rows = [view("CAM-01", "A1"), view("CAM-01", "A2")]
        columns = [view("CAM-04", "B1"), view("CAM-04", "B2")]
        values = {("A1", "B1"): 0.91, ("A1", "B2"): 0.21, ("A2", "B1"): 0.22, ("A2", "B2"): 0.92}
        normal = self.run_matrix(rows, columns, values)
        permuted = self.run_matrix(list(reversed(rows)), list(reversed(columns)), values)
        self.assertEqual(proposals(normal), proposals(permuted))

    def test_rectangular_3x2_matrix(self) -> None:
        result = self.run_matrix(
            [view("CAM-01", "A1"), view("CAM-01", "A2"), view("CAM-01", "A3")],
            [view("CAM-04", "B1"), view("CAM-04", "B2")],
            {
                ("A1", "B1"): 0.90, ("A1", "B2"): 0.10,
                ("A2", "B1"): 0.20, ("A2", "B2"): 0.80,
                ("A3", "B1"): 0.30, ("A3", "B2"): 0.40,
            },
        )
        self.assertEqual(proposals(result), {("A1", "B1"), ("A2", "B2")})

    def test_insufficient_pair_is_excluded(self) -> None:
        result = self.run_matrix(
            [view("CAM-01", "A1", count=2)],
            [view("CAM-04", "B1")],
            {("A1", "B1"): "INSUFFICIENT"},
        )
        self.assertEqual(result.diagnostics[0].status, INSUFFICIENT)
        self.assertFalse(result.diagnostics[0].assigned)

    def test_stale_pair_is_excluded_without_scoring(self) -> None:
        now_ns = time.monotonic_ns()
        called = []

        def should_not_run(_first, _second):
            called.append(True)
            return score(0.9)

        result = match_same_room_camera_pair_v1(
            [view("CAM-01", "A1", last_seen_ns=now_ns - 13_000_000_000)],
            [view("CAM-04", "B1", last_seen_ns=now_ns)],
            ROOMS,
            now_ns=now_ns,
            config=SameRoomMatcherConfigV1(recent_age_sec=12.0),
            pair_scorer=should_not_run,
        )
        self.assertEqual(result.diagnostics[0].status, STALE)
        self.assertEqual(called, [])

    def test_nan_and_inf_evidence_is_invalid_and_excluded(self) -> None:
        for bad in (float("nan"), float("inf"), float("-inf")):
            first = view("CAM-01", "A1")
            first.samples[0].embedding[7] = bad
            with self.subTest(bad=bad):
                result = match_same_room_camera_pair_v1(
                    [first], [view("CAM-04", "B1")], ROOMS
                )
                self.assertEqual(result.diagnostics[0].status, INVALID)
                self.assertFalse(result.diagnostics[0].assigned)
            first.samples[0].embedding[7] = 0.0

    def test_row_and_column_margins_are_computed_correctly(self) -> None:
        result = self.run_matrix(
            [view("CAM-01", "A1"), view("CAM-01", "A2")],
            [view("CAM-04", "B1"), view("CAM-04", "B2")],
            {("A1", "B1"): 0.80, ("A1", "B2"): 0.70, ("A2", "B1"): 0.60, ("A2", "B2"): 0.90},
        )
        row = next(item for item in result.diagnostics if (item.track_a, item.track_b) == ("A1", "B1"))
        self.assertAlmostEqual(row.row_best, 0.80)
        self.assertAlmostEqual(row.row_second, 0.70)
        self.assertAlmostEqual(row.row_margin, 0.10)
        self.assertAlmostEqual(row.column_best, 0.80)
        self.assertAlmostEqual(row.column_second, 0.60)
        self.assertAlmostEqual(row.column_margin, 0.20)

    def test_equal_scores_are_deterministic(self) -> None:
        rows = [view("CAM-01", "A2"), view("CAM-01", "A1")]
        columns = [view("CAM-04", "B2"), view("CAM-04", "B1")]
        values = {(a, b): 0.8 for a in ("A1", "A2") for b in ("B1", "B2")}
        observed = [proposals(self.run_matrix(rows, columns, values)) for _ in range(5)]
        self.assertTrue(all(item == {("A1", "B1")} for item in observed))

    def test_no_local_track_appears_twice(self) -> None:
        result = self.run_matrix(
            [view("CAM-01", f"A{i}") for i in range(1, 4)],
            [view("CAM-04", f"B{i}") for i in range(1, 4)],
            {(f"A{i}", f"B{j}"): (0.9 if i == j else 0.4) for i in range(1, 4) for j in range(1, 4)},
        )
        endpoints = [(row.camera_a, row.track_a) for row in result.proposals]
        endpoints += [(row.camera_b, row.track_b) for row in result.proposals]
        self.assertEqual(len(endpoints), len(set(endpoints)))

    def test_matcher_does_not_mutate_galleries(self) -> None:
        rows = [view("CAM-01", "A2"), view("CAM-01", "A1")]
        columns = [view("CAM-04", "B2"), view("CAM-04", "B1")]
        identities = [id(sample.embedding) for item in rows + columns for sample in item.samples]
        values = {(a, b): 0.8 for a in ("A1", "A2") for b in ("B1", "B2")}
        self.run_matrix(rows, columns, values)
        self.assertEqual([item.local_track_id for item in rows], ["A2", "A1"])
        self.assertEqual([item.local_track_id for item in columns], ["B2", "B1"])
        self.assertEqual(identities, [id(sample.embedding) for item in rows + columns for sample in item.samples])

    def test_matcher_has_no_global_or_room_identity_assignment(self) -> None:
        field_names = {field.name for field in dataclasses.fields(type(self.run_matrix(
            [view("CAM-01", "A1")], [view("CAM-04", "B1")], {("A1", "B1"): 0.8}
        ).diagnostics[0]))}
        self.assertNotIn("global_id", field_names)
        self.assertNotIn("room_id", field_names)

    def test_ambiguous_clothes_exposes_small_margins(self) -> None:
        rows = [view("CAM-01", "A1"), view("CAM-01", "A2")]
        columns = [view("CAM-04", "B1"), view("CAM-04", "B2")]
        values = {("A1", "B1"): 0.76, ("A1", "B2"): 0.75, ("A2", "B1"): 0.74, ("A2", "B2"): 0.77}
        diagnostic = self.run_matrix(rows, columns, values)
        by_pair = {(row.track_a, row.track_b): row for row in diagnostic.diagnostics}
        self.assertAlmostEqual(by_pair[("A1", "B1")].row_margin, 0.01)
        self.assertAlmostEqual(by_pair[("A1", "B1")].column_margin, 0.02)
        gated = self.run_matrix(
            rows,
            columns,
            values,
            config=SameRoomMatcherConfigV1(min_row_margin=0.02),
        )
        gated_rows = {(row.track_a, row.track_b): row for row in gated.diagnostics}
        self.assertEqual(gated_rows[("A1", "B1")].status, LOW_MARGIN)
        self.assertFalse(gated_rows[("A1", "B1")].assigned)

    def test_same_camera_and_cross_room_matrices_are_forbidden(self) -> None:
        with self.assertRaisesRegex(ValueError, "same-camera"):
            match_same_room_camera_pair_v1(
                [view("CAM-01", "A1")], [view("CAM-01", "A2")], ROOMS
            )
        with self.assertRaisesRegex(ValueError, "cross-room"):
            match_same_room_camera_pair_v1(
                [view("CAM-01", "A1")], [view("CAM-02", "C1")], ROOMS
            )


class SameRoomShadowWorkerTests(unittest.TestCase):
    def test_stability_and_tsv_have_no_embeddings(self) -> None:
        views = (view("CAM-01", "A1", first_basis=20), view("CAM-04", "B1", first_basis=20))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matches.tsv"
            worker = V11SameRoomMatcherShadowWorkerV1(
                lambda: views,
                {"CAM-01": "Devs", "CAM-04": "Devs"},
                tsv_path=path,
            )
            worker.start()
            worker.notify()
            deadline = time.monotonic() + 3.0
            while worker.snapshot()["proposals"] < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            worker.close()
            snapshot = worker.snapshot()
            self.assertGreaterEqual(snapshot["cycles"], 1)
            self.assertGreaterEqual(snapshot["proposals"], 1)
            self.assertEqual(snapshot["unique_proposals"], 1)
            self.assertEqual(snapshot["worker_errors"], 0)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(tuple(lines[0].split("\t")), TSV_COLUMNS)
            self.assertNotIn("embedding", lines[0].lower())


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
