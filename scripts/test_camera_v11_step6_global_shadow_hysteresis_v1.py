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
    EXPIRED_SHADOW,
    GLOBAL_SHADOW_AMBIGUITY,
    GLOBAL_SHADOW_CONFIRM,
    GLOBAL_SHADOW_CONFLICT,
    GLOBAL_SHADOW_EXPIRE,
    GLOBAL_SHADOW_OBSERVE,
    GlobalShadowEventV1,
    GlobalShadowStateMachineV1,
)
from services.camera_v11.step6_global_shadow_hysteresis_v1 import (
    CONFLICT_HOLD_SHADOW,
    EXPIRED_VERIFY_SHADOW,
    GLOBAL_VERIFY_CONFLICT_PERSISTENT,
    GLOBAL_VERIFY_HOLD,
    GLOBAL_VERIFY_PASS,
    GLOBAL_VERIFY_RECOVER,
    VERIFIED_SHADOW,
    VERIFY_PENDING,
    GlobalShadowHysteresisV1,
)


def event(
    kind: str,
    *,
    shadow_id: str = "GSH-000001",
    pair: tuple[tuple[str, str], tuple[str, str]] = (
        ("CAM-01", "CAM-01-T00001"),
        ("CAM-04", "CAM-04-T00001"),
    ),
    state: str = CONFIRMED_SHADOW,
    score: float = 0.72,
    timestamp_ns: int = 1,
) -> GlobalShadowEventV1:
    (camera_a, track_a), (camera_b, track_b) = pair
    return GlobalShadowEventV1(
        timestamp_ns=timestamp_ns,
        event=kind,
        shadow_global_id=shadow_id,
        room="Devs",
        camera_a=camera_a,
        track_a=track_a,
        camera_b=camera_b,
        track_b=track_b,
        proposal_count=3,
        consecutive_count=3,
        state=state,
        robust_score=score,
        status=state,
    )


class Step6GlobalShadowHysteresisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.machine = GlobalShadowHysteresisV1(
            verify_clean_observations=3,
            recover_clean_observations=3,
            persistent_conflict_observations=3,
        )

    def confirm(self, shadow_id: str = "GSH-000001") -> None:
        out = self.machine.observe_step5_event(event(GLOBAL_SHADOW_CONFIRM, shadow_id=shadow_id))
        self.assertEqual(1, len(out))
        self.assertEqual(VERIFY_PENDING, out[0].state)

    def observe(self, shadow_id: str = "GSH-000001", n: int = 1) -> tuple:
        output = []
        for index in range(n):
            output.extend(
                self.machine.observe_step5_event(
                    event(GLOBAL_SHADOW_OBSERVE, shadow_id=shadow_id, timestamp_ns=10 + index)
                )
            )
        return tuple(output)

    def test_confirm_creates_verify_pending(self) -> None:
        self.confirm()
        row = self.machine.records[0]
        self.assertEqual(VERIFY_PENDING, row.state)
        self.assertEqual(0, row.clean_observations)

    def test_third_clean_observation_verifies(self) -> None:
        self.confirm()
        self.observe(n=2)
        self.assertEqual(VERIFY_PENDING, self.machine.records[0].state)
        out = self.observe(n=1)
        self.assertTrue(any(item.event == GLOBAL_VERIFY_PASS for item in out))
        self.assertEqual(VERIFIED_SHADOW, self.machine.records[0].state)

    def test_successor_pair_follows_same_shadow_id_and_verifies(self) -> None:
        first = (("CAM-01", "A1"), ("CAM-04", "B1"))
        second = (("CAM-01", "A2"), ("CAM-04", "B1"))
        third = (("CAM-01", "A2"), ("CAM-04", "B2"))
        self.machine.observe_step5_event(event(GLOBAL_SHADOW_CONFIRM, pair=first))
        self.machine.observe_step5_event(event(GLOBAL_SHADOW_OBSERVE, pair=second, timestamp_ns=2))
        self.machine.observe_step5_event(event(GLOBAL_SHADOW_OBSERVE, pair=third, timestamp_ns=3))
        out = self.machine.observe_step5_event(event(GLOBAL_SHADOW_OBSERVE, pair=third, timestamp_ns=4))
        row = self.machine.records[0]
        self.assertEqual(row.shadow_global_id, "GSH-000001")
        self.assertEqual(row.pair_key, third)
        self.assertEqual(row.state, VERIFIED_SHADOW)
        self.assertTrue(any(item.event == GLOBAL_VERIFY_PASS for item in out))
        self.assertEqual(self.machine.snapshot()["verify_hold_events"], 0)

    def test_conflict_moves_verified_identity_to_hold(self) -> None:
        self.confirm()
        self.observe(n=3)
        out = self.machine.observe_step5_event(
            event(
                GLOBAL_SHADOW_CONFLICT,
                shadow_id="",
                pair=(("CAM-01", "CAM-01-T00001"), ("CAM-04", "CAM-04-T99999")),
                state="CONFLICT_PENDING",
            )
        )
        self.assertTrue(any(item.event == GLOBAL_VERIFY_HOLD for item in out))
        self.assertEqual(CONFLICT_HOLD_SHADOW, self.machine.records[0].state)
        self.assertEqual(0, self.machine.records[0].clean_observations)

    def test_unrelated_conflict_does_not_hold_identity(self) -> None:
        self.confirm()
        self.observe(n=3)
        self.machine.observe_step5_event(
            event(
                GLOBAL_SHADOW_CONFLICT,
                shadow_id="",
                pair=(("CAM-01", "CAM-01-T77777"), ("CAM-04", "CAM-04-T88888")),
                state="CONFLICT_PENDING",
            )
        )
        self.assertEqual(VERIFIED_SHADOW, self.machine.records[0].state)

    def test_hold_requires_three_clean_observations_to_recover(self) -> None:
        self.confirm()
        self.observe(n=3)
        self.machine.observe_step5_event(
            event(
                GLOBAL_SHADOW_CONFLICT,
                shadow_id="",
                pair=(("CAM-01", "CAM-01-T00001"), ("CAM-04", "CAM-04-T00002")),
                state="CONFLICT_PENDING",
            )
        )
        self.observe(n=2)
        self.assertEqual(CONFLICT_HOLD_SHADOW, self.machine.records[0].state)
        out = self.observe(n=1)
        self.assertTrue(any(item.event == GLOBAL_VERIFY_RECOVER for item in out))
        self.assertEqual(VERIFIED_SHADOW, self.machine.records[0].state)

    def test_three_same_conflicts_mark_persistent_without_reassignment(self) -> None:
        self.confirm()
        conflict = event(
            GLOBAL_SHADOW_CONFLICT,
            shadow_id="",
            pair=(("CAM-01", "CAM-01-T00001"), ("CAM-04", "CAM-04-T00002")),
            state="CONFLICT_PENDING",
        )
        output = []
        for _ in range(3):
            output.extend(self.machine.observe_step5_event(conflict))
        self.assertTrue(any(item.event == GLOBAL_VERIFY_CONFLICT_PERSISTENT for item in output))
        row = self.machine.records[0]
        self.assertEqual("GSH-000001", row.shadow_global_id)
        self.assertEqual(CONFLICT_HOLD_SHADOW, row.state)

    def test_different_conflict_pair_resets_conflict_streak(self) -> None:
        self.confirm()
        first = event(
            GLOBAL_SHADOW_CONFLICT,
            shadow_id="",
            pair=(("CAM-01", "CAM-01-T00001"), ("CAM-04", "CAM-04-T00002")),
            state="CONFLICT_PENDING",
        )
        second = event(
            GLOBAL_SHADOW_CONFLICT,
            shadow_id="",
            pair=(("CAM-01", "CAM-01-T00001"), ("CAM-04", "CAM-04-T00003")),
            state="CONFLICT_PENDING",
        )
        self.machine.observe_step5_event(first)
        self.machine.observe_step5_event(first)
        self.machine.observe_step5_event(second)
        self.assertEqual(1, self.machine.records[0].conflict_streak)

    def test_two_people_verify_and_cross_owner_ambiguity_never_holds(self) -> None:
        step5 = GlobalShadowStateMachineV1(
            confirm_observations=3,
            confirm_consecutive=3,
            expire_provisional_after_missed_cycles=6,
        )
        verifier = GlobalShadowHysteresisV1(
            verify_clean_observations=3,
            recover_clean_observations=3,
            persistent_conflict_observations=3,
        )
        pairs = (
            (("CAM-01", "A1"), ("CAM-04", "A4")),
            (("CAM-01", "B1"), ("CAM-04", "B4")),
        )
        for cycle in range(1, 7):
            step5.begin_cycle(cycle)
            for a, b in pairs:
                events = step5.observe_proposal(
                    cycle=cycle,
                    timestamp_ns=cycle,
                    room="Devs",
                    camera_a=a[0],
                    track_a=a[1],
                    camera_b=b[0],
                    track_b=b[1],
                    robust_score=0.8,
                    reciprocal=True,
                    assigned=True,
                    status="MATCH_PROPOSED",
                )
                for source in events:
                    verifier.observe_step5_event(source)
            for source in step5.end_cycle(cycle=cycle, timestamp_ns=cycle):
                verifier.observe_step5_event(source)

        before = verifier.snapshot()
        self.assertEqual(before["verify_records_created"], 2)
        self.assertEqual(before["verify_verified"], 2)
        self.assertEqual(before["verify_verified_total"], 2)

        step5.begin_cycle(7)
        ambiguity = step5.observe_proposal(
            cycle=7,
            timestamp_ns=7,
            room="Devs",
            camera_a="CAM-01",
            track_a="A1",
            camera_b="CAM-04",
            track_b="B4",
            robust_score=0.81,
            reciprocal=True,
            assigned=True,
            status="MATCH_PROPOSED",
        )
        step5.end_cycle(cycle=7, timestamp_ns=7)
        self.assertEqual(ambiguity[0].event, GLOBAL_SHADOW_AMBIGUITY)
        for source in ambiguity:
            self.assertEqual(verifier.observe_step5_event(source), ())

        after = verifier.snapshot()
        self.assertEqual(after["verify_verified"], 2)
        self.assertEqual(after["verify_hold"], 0)
        self.assertEqual(after["verify_hold_events"], 0)
        self.assertEqual(after["verify_recovered_total"], 0)

    def test_expiry_marks_verification_expired(self) -> None:
        self.confirm()
        out = self.machine.observe_step5_event(
            event(GLOBAL_SHADOW_EXPIRE, state=EXPIRED_SHADOW, timestamp_ns=999)
        )
        self.assertEqual(EXPIRED_VERIFY_SHADOW, self.machine.records[0].state)
        self.assertEqual(1, len(out))

    def test_raw_score_never_controls_verification_threshold(self) -> None:
        self.machine.observe_step5_event(event(GLOBAL_SHADOW_CONFIRM, score=-1000.0))
        for index in range(3):
            self.machine.observe_step5_event(
                event(GLOBAL_SHADOW_OBSERVE, score=-1000.0, timestamp_ns=20 + index)
            )
        self.assertEqual(VERIFIED_SHADOW, self.machine.records[0].state)

    def test_no_production_identity_mutation_fields_exist(self) -> None:
        self.confirm()
        row = self.machine.records[0]
        self.assertFalse(hasattr(row, "production_global_id"))
        self.assertFalse(hasattr(row, "room_id"))
        self.assertFalse(hasattr(row, "tracker_id"))

    def test_snapshot_accounting(self) -> None:
        self.confirm()
        self.observe(n=3)
        snap = self.machine.snapshot()
        self.assertEqual(1, snap["verify_records_created"])
        self.assertEqual(1, snap["verify_verified"])
        self.assertEqual(0, snap["verify_hold"])
        self.assertLess(float(snap["verify_p95_ms"]), 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
