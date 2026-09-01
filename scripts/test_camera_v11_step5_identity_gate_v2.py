#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v11.step4_reid_same_room_matcher_v1 import (
    MATCH_PROPOSED,
    SameRoomPairDiagnosticV1,
)
from services.camera_v11.step5_same_room_shadow_tap_v1 import (
    Step5IdentityGateConfigV2,
    proposal_passes_step5_identity_gate_v2,
)


def row(
    *,
    robust: float = 0.84,
    row_second: float | None = 0.70,
    row_margin: float | None = 0.14,
    column_second: float | None = 0.71,
    column_margin: float | None = 0.13,
    support_ge_065: int = 5,
) -> SameRoomPairDiagnosticV1:
    return SameRoomPairDiagnosticV1(
        room="Devs",
        camera_a="CAM-01",
        track_a="CAM-01-CT00001",
        camera_b="CAM-04",
        track_b="CAM-04-CT00001",
        samples_a=5,
        samples_b=5,
        robust_score=robust,
        max_score=0.91,
        top3_mean=0.88,
        median=0.72,
        a_best_mean=0.84,
        a_best_min=0.73,
        b_best_mean=0.83,
        b_best_min=0.72,
        support_ge_050=12,
        support_ge_055=10,
        support_ge_060=8,
        support_ge_065=support_ge_065,
        support_ge_070=4,
        support_ge_075=3,
        support_ge_080=2,
        row_best=robust,
        row_second=row_second,
        row_margin=row_margin,
        column_best=robust,
        column_second=column_second,
        column_margin=column_margin,
        reciprocal=True,
        assigned=True,
        status=MATCH_PROPOSED,
        reason="test",
    )


class Step5IdentityGateV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Step5IdentityGateConfigV2(
            min_robust_score=0.72,
            min_row_margin=0.05,
            min_column_margin=0.05,
            min_support_ge_065=3,
        )

    def assert_gate(self, item, expected: bool, reason: str) -> None:
        accepted, actual_reason = proposal_passes_step5_identity_gate_v2(item, self.cfg)
        self.assertEqual(accepted, expected)
        self.assertEqual(actual_reason, reason)

    def test_clear_reciprocal_pair_passes(self) -> None:
        self.assert_gate(row(), True, "accepted")

    def test_near_tie_row_margin_is_no_match(self) -> None:
        self.assert_gate(
            row(row_second=0.839576, row_margin=0.000424),
            False,
            "low_row_margin",
        )

    def test_near_tie_column_margin_is_no_match(self) -> None:
        self.assert_gate(
            row(column_second=0.835, column_margin=0.005),
            False,
            "low_column_margin",
        )

    def test_low_absolute_similarity_is_no_match(self) -> None:
        self.assert_gate(row(robust=0.69), False, "low_robust_score")

    def test_low_high_similarity_support_is_no_match(self) -> None:
        self.assert_gate(row(support_ge_065=2), False, "low_support_ge_065")

    def test_single_candidate_can_pass_without_margin(self) -> None:
        self.assert_gate(
            row(
                row_second=None,
                row_margin=None,
                column_second=None,
                column_margin=None,
            ),
            True,
            "accepted",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
