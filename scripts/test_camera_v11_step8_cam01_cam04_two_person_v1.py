#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_camera_v11_step8_cam01_cam04_two_person_v1 import (
    _isolation_reasons,
    _verified_ids_at,
)


def global_observe(shadow_id: str) -> dict[str, str]:
    return {
        "event": "GLOBAL_SHADOW_OBSERVE",
        "shadow_global_id": shadow_id,
    }


def verify(event: str, shadow_id: str) -> dict[str, str]:
    return {"event": event, "shadow_global_id": shadow_id}


class Step8TwoPersonPhaseCheckerTests(unittest.TestCase):
    def test_preverified_identity_with_only_observations_in_window_is_valid(self) -> None:
        global_rows = [global_observe("GSH-A") for _ in range(3)]
        verify_rows = [verify("GLOBAL_VERIFY_PASS", "GSH-A")]
        reasons = _isolation_reasons(
            label="d",
            global_rows=global_rows,
            verify_rows=verify_rows,
            global_start=0,
            global_end=3,
            verify_start=1,
            expected_id="GSH-A",
            min_observations=3,
        )
        self.assertEqual(reasons, [])

    def test_hidden_id_swap_after_crossing_is_rejected(self) -> None:
        global_rows = [global_observe("GSH-B") for _ in range(3)]
        verify_rows = [
            verify("GLOBAL_VERIFY_PASS", "GSH-A"),
            verify("GLOBAL_VERIFY_PASS", "GSH-B"),
        ]
        reasons = _isolation_reasons(
            label="d",
            global_rows=global_rows,
            verify_rows=verify_rows,
            global_start=0,
            global_end=3,
            verify_start=2,
            expected_id="GSH-A",
            min_observations=3,
        )
        self.assertIn(
            "id_swap_after_crossing_phase_d_seen=GSH-B/expected=GSH-A",
            reasons,
        )

    def test_identity_in_hold_at_window_start_is_not_treated_as_verified(self) -> None:
        verify_rows = [
            verify("GLOBAL_VERIFY_PASS", "GSH-A"),
            verify("GLOBAL_VERIFY_HOLD", "GSH-A"),
        ]
        reasons = _isolation_reasons(
            label="d",
            global_rows=[global_observe("GSH-A") for _ in range(3)],
            verify_rows=verify_rows,
            global_start=0,
            global_end=3,
            verify_start=2,
            expected_id="GSH-A",
            min_observations=3,
        )
        self.assertIn("phase_d_expected_not_verified_at_start=GSH-A", reasons)

    def test_recover_restores_carried_verification_state(self) -> None:
        rows = [
            verify("GLOBAL_VERIFY_PASS", "GSH-A"),
            verify("GLOBAL_VERIFY_HOLD", "GSH-A"),
            verify("GLOBAL_VERIFY_RECOVER", "GSH-A"),
        ]
        self.assertEqual(_verified_ids_at(rows, 3), {"GSH-A"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
