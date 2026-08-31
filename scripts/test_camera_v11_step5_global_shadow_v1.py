#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v11.step5_global_shadow_v1 import (
    CONFIRMED_SHADOW,
    CONFLICT_PENDING,
    EXPIRED_SHADOW,
    GLOBAL_SHADOW_CONFIRM,
    GLOBAL_SHADOW_CONFLICT,
    PROVISIONAL,
    GlobalShadowStateMachineV1,
)


class Step5GlobalShadowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.machine = GlobalShadowStateMachineV1(
            confirm_observations=3,
            confirm_consecutive=3,
            expire_provisional_after_missed_cycles=3,
        )

    def observe(
        self,
        cycle: int,
        *,
        a: tuple[str, str] = ("CAM-01", "A1"),
        b: tuple[str, str] = ("CAM-04", "B1"),
        score: float = 0.8,
    ):
        self.machine.begin_cycle(cycle)
        events = self.machine.observe_proposal(
            cycle=cycle,
            timestamp_ns=cycle * 1_000_000,
            room="Devs",
            camera_a=a[0],
            track_a=a[1],
            camera_b=b[0],
            track_b=b[1],
            robust_score=score,
            reciprocal=True,
            assigned=True,
            status="MATCH_PROPOSED",
        )
        self.machine.end_cycle(cycle=cycle, timestamp_ns=cycle * 1_000_000 + 1)
        return events

    def test_first_proposal_creates_provisional(self) -> None:
        events = self.observe(1)
        self.assertEqual(len(events), 1)
        record = self.machine.records[0]
        self.assertEqual(record.shadow_global_id, "GSH-000001")
        self.assertEqual(record.state, PROVISIONAL)
        self.assertEqual(record.proposal_count, 1)
        self.assertEqual(record.consecutive_count, 1)

    def test_three_consecutive_observations_confirm_exactly_on_third(self) -> None:
        self.observe(1)
        self.observe(2)
        self.assertEqual(self.machine.records[0].state, PROVISIONAL)
        events = self.observe(3)
        self.assertEqual(self.machine.records[0].state, CONFIRMED_SHADOW)
        self.assertTrue(any(event.event == GLOBAL_SHADOW_CONFIRM for event in events))
        self.assertEqual(self.machine.records[0].proposal_count, 3)
        self.assertEqual(self.machine.records[0].consecutive_count, 3)

    def test_two_observations_do_not_confirm(self) -> None:
        self.observe(1)
        self.observe(2)
        self.assertEqual(self.machine.records[0].state, PROVISIONAL)

    def test_missing_cycle_breaks_consecutive_count(self) -> None:
        self.observe(1)
        self.machine.begin_cycle(2)
        self.machine.end_cycle(cycle=2, timestamp_ns=2_000_001)
        self.observe(3)
        record = self.machine.records[0]
        self.assertEqual(record.proposal_count, 2)
        self.assertEqual(record.consecutive_count, 1)
        self.assertEqual(record.state, PROVISIONAL)

    def test_different_pair_gets_new_id_after_expiry_not_old_id(self) -> None:
        self.observe(1)
        for cycle in (2, 3, 4):
            self.machine.begin_cycle(cycle)
            self.machine.end_cycle(cycle=cycle, timestamp_ns=cycle * 1_000_000)
        self.assertEqual(self.machine.records[0].state, EXPIRED_SHADOW)
        self.observe(5, a=("CAM-01", "A2"), b=("CAM-04", "B2"))
        self.assertEqual(self.machine.records[-1].shadow_global_id, "GSH-000002")

    def test_one_local_track_cannot_join_two_active_ids(self) -> None:
        self.observe(1)
        self.machine.begin_cycle(2)
        events = self.machine.observe_proposal(
            cycle=2,
            timestamp_ns=2_000_000,
            room="Devs",
            camera_a="CAM-01",
            track_a="A1",
            camera_b="CAM-04",
            track_b="B2",
            robust_score=0.81,
            reciprocal=True,
            assigned=True,
            status="MATCH_PROPOSED",
        )
        self.machine.end_cycle(cycle=2, timestamp_ns=2_000_001)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event, GLOBAL_SHADOW_CONFLICT)
        self.assertEqual(events[0].state, CONFLICT_PENDING)
        self.assertEqual(len(self.machine.records), 1)

    def test_conflict_does_not_merge_ids(self) -> None:
        self.observe(1, a=("CAM-01", "A1"), b=("CAM-04", "B1"))
        self.machine.begin_cycle(2)
        events = self.machine.observe_proposal(
            cycle=2,
            timestamp_ns=2_000_000,
            room="Devs",
            camera_a="CAM-01",
            track_a="A1",
            camera_b="CAM-04",
            track_b="B2",
            robust_score=0.82,
            reciprocal=True,
            assigned=True,
            status="MATCH_PROPOSED",
        )
        self.machine.end_cycle(cycle=2, timestamp_ns=2_000_001)
        self.assertEqual(events[0].status, CONFLICT_PENDING)
        self.assertEqual(len(self.machine.records), 1)

    def test_nonproposal_is_ignored(self) -> None:
        self.machine.begin_cycle(1)
        events = self.machine.observe_proposal(
            cycle=1,
            timestamp_ns=1,
            room="Devs",
            camera_a="CAM-01",
            track_a="A1",
            camera_b="CAM-04",
            track_b="B1",
            robust_score=0.8,
            reciprocal=False,
            assigned=False,
            status="NON_RECIPROCAL",
        )
        self.machine.end_cycle(cycle=1, timestamp_ns=2)
        self.assertEqual(events, ())
        self.assertEqual(self.machine.records, ())

    def test_provisional_expiry_is_bounded(self) -> None:
        self.observe(1)
        for cycle in (2, 3, 4):
            self.machine.begin_cycle(cycle)
            self.machine.end_cycle(cycle=cycle, timestamp_ns=cycle)
        record = self.machine.records[0]
        self.assertEqual(record.state, EXPIRED_SHADOW)
        snapshot = self.machine.snapshot()
        self.assertEqual(snapshot["global_shadow_active"], 0)
        self.assertEqual(snapshot["global_shadow_member_tracks"], 0)

    def test_deterministic_id_allocation(self) -> None:
        self.observe(1, a=("CAM-01", "A1"), b=("CAM-04", "B1"))
        self.machine.begin_cycle(2)
        self.machine.end_cycle(cycle=2, timestamp_ns=2)
        self.machine.begin_cycle(3)
        self.machine.end_cycle(cycle=3, timestamp_ns=3)
        self.machine.begin_cycle(4)
        self.machine.end_cycle(cycle=4, timestamp_ns=4)
        self.observe(5, a=("CAM-01", "A2"), b=("CAM-04", "B2"))
        self.assertEqual(
            [record.shadow_global_id for record in self.machine.records],
            ["GSH-000001", "GSH-000002"],
        )

    def test_best_score_is_monotonic(self) -> None:
        self.observe(1, score=0.7)
        self.observe(2, score=0.9)
        self.observe(3, score=0.8)
        record = self.machine.records[0]
        self.assertAlmostEqual(record.first_score, 0.7)
        self.assertAlmostEqual(record.last_score, 0.8)
        self.assertAlmostEqual(record.best_score, 0.9)

    def test_no_production_identity_fields_exist(self) -> None:
        self.observe(1)
        record = self.machine.records[0]
        self.assertFalse(hasattr(record, "global_id"))
        self.assertFalse(hasattr(record, "room_id"))
        self.assertFalse(hasattr(record, "face_id"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
