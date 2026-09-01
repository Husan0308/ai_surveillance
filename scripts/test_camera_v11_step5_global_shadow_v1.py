#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v11.step5_global_shadow_v1 import (
    CONFIRMED_SHADOW,
    CONFLICT_PENDING,
    EXPIRED_SHADOW,
    GLOBAL_SHADOW_AMBIGUITY,
    GLOBAL_SHADOW_CONFIRM,
    GLOBAL_SHADOW_CONFLICT,
    GLOBAL_SHADOW_NO_ACTION,
    GLOBAL_SHADOW_OBSERVE,
    PROVISIONAL,
    GlobalShadowStateMachineV1,
)
from services.camera_v11.step5_global_shadow_worker_v1 import (
    TSV_COLUMNS,
    V11GlobalShadowWorkerV1,
)


class Step5GlobalShadowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.machine = GlobalShadowStateMachineV1(
            confirm_observations=3,
            confirm_consecutive=3,
            expire_provisional_after_missed_cycles=3,
            successor_confirm_observations=2,
            successor_max_gap_cycles=2,
        )

    def observe(
        self,
        cycle: int,
        *,
        a: tuple[str, str] = ("CAM-01", "A1"),
        b: tuple[str, str] = ("CAM-04", "B1"),
        score: float = 0.8,
        room: str = "Devs",
    ):
        self.machine.begin_cycle(cycle)
        events = self.machine.observe_proposal(
            cycle=cycle,
            timestamp_ns=cycle * 1_000_000,
            room=room,
            camera_a=a[0],
            track_a=a[1],
            camera_b=b[0],
            track_b=b[1],
            robust_score=score,
            reciprocal=True,
            assigned=True,
            status="MATCH_PROPOSED",
        )
        expiry = self.machine.end_cycle(
            cycle=cycle, timestamp_ns=cycle * 1_000_000 + 1
        )
        return events + expiry

    def empty_cycle(self, cycle: int):
        self.machine.begin_cycle(cycle)
        return self.machine.end_cycle(cycle=cycle, timestamp_ns=cycle)

    def confirm_two_people(self) -> tuple[str, str]:
        pairs = (
            (("CAM-01", "A1"), ("CAM-04", "A4")),
            (("CAM-01", "B1"), ("CAM-04", "B4")),
        )
        for cycle in (1, 2, 3):
            self.machine.begin_cycle(cycle)
            for a, b in pairs:
                self.machine.observe_proposal(
                    cycle=cycle,
                    timestamp_ns=cycle * 1_000_000,
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
            self.machine.end_cycle(cycle=cycle, timestamp_ns=cycle * 1_000_000 + 1)
        by_pair = {row.pair_key: row for row in self.machine.records}
        self.assertEqual(len(by_pair), 2)
        self.assertTrue(all(row.state == CONFIRMED_SHADOW for row in by_pair.values()))
        return (
            by_pair[pairs[0]].shadow_global_id,
            by_pair[pairs[1]].shadow_global_id,
        )

    def confirm_pair(
        self,
        cycles: tuple[int, int, int],
        *,
        a: tuple[str, str],
        b: tuple[str, str],
    ) -> str:
        for cycle in cycles:
            self.observe(cycle, a=a, b=b)
        record = next(row for row in self.machine.records if row.pair_key == (a, b))
        self.assertEqual(record.state, CONFIRMED_SHADOW)
        return record.shadow_global_id

    def test_first_proposal_creates_provisional_with_decision_context(self) -> None:
        events = self.observe(1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].decision, "create_new")
        self.assertEqual(events[0].cycle, 1)
        self.assertEqual(events[0].owner_a, "GSH-000001")
        record = self.machine.records[0]
        self.assertEqual(record.state, PROVISIONAL)
        self.assertEqual(record.proposal_count, 1)
        self.assertEqual(record.consecutive_count, 1)

    def test_clean_observations_survive_cache_pending_cycle_and_confirm(self) -> None:
        self.observe(1)
        self.empty_cycle(2)
        self.observe(3)
        events = self.observe(5)
        self.assertEqual(self.machine.records[0].state, CONFIRMED_SHADOW)
        self.assertTrue(any(event.event == GLOBAL_SHADOW_CONFIRM for event in events))
        self.assertEqual(self.machine.records[0].consecutive_count, 3)

    def test_two_independent_people_create_two_independent_confirmed_globals(self) -> None:
        self.confirm_two_people()
        snap = self.machine.snapshot()
        self.assertEqual(snap["global_shadow_created"], 2)
        self.assertEqual(snap["global_shadow_confirmed"], 2)
        self.assertEqual(snap["global_shadow_active"], 2)
        self.assertEqual(snap["global_shadow_member_tracks"], 4)
        self.assertEqual(snap["global_shadow_conflicts"], 0)
        self.assertEqual(snap["global_shadow_expired"], 0)

    def test_successor_requires_repeated_anchor_then_reuses_global_id(self) -> None:
        self.confirm_pair(
            (1, 2, 3), a=("CAM-01", "A1"), b=("CAM-04", "A4")
        )
        first = self.observe(4, a=("CAM-01", "A2"), b=("CAM-04", "A4"))
        self.assertEqual(first[0].event, GLOBAL_SHADOW_NO_ACTION)
        self.assertEqual(first[0].decision, "successor_evidence_pending")
        events = self.observe(5, a=("CAM-01", "A2"), b=("CAM-04", "A4"))
        record = self.machine.records[0]
        self.assertEqual(record.shadow_global_id, "GSH-000001")
        self.assertEqual(
            record.pair_key, (("CAM-01", "A2"), ("CAM-04", "A4"))
        )
        self.assertTrue(
            any(
                event.event == GLOBAL_SHADOW_OBSERVE
                and event.decision == "successor_attach"
                for event in events
            )
        )
        self.assertEqual(self.machine.snapshot()["global_shadow_successor_attaches"], 1)

    def test_person_a_and_b_successors_reuse_their_own_globals(self) -> None:
        a_id, b_id = self.confirm_two_people()
        self.observe(4, a=("CAM-01", "A2"), b=("CAM-04", "A4"))
        self.observe(5, a=("CAM-01", "A2"), b=("CAM-04", "A4"))
        self.observe(6, a=("CAM-01", "B2"), b=("CAM-04", "B4"))
        self.observe(7, a=("CAM-01", "B2"), b=("CAM-04", "B4"))
        by_id = {row.shadow_global_id: row for row in self.machine.records}
        self.assertEqual(by_id[a_id].pair_key[0], ("CAM-01", "A2"))
        self.assertEqual(by_id[b_id].pair_key[0], ("CAM-01", "B2"))
        self.assertNotEqual(a_id, b_id)
        self.assertEqual(self.machine.snapshot()["global_shadow_member_tracks"], 4)

    def test_same_cycle_unknown_overlap_is_harmless_ambiguity(self) -> None:
        self.observe(1)
        self.machine.begin_cycle(2)
        self.machine.observe_proposal(
            cycle=2,
            timestamp_ns=2_000_000,
            room="Devs",
            camera_a="CAM-01",
            track_a="A1",
            camera_b="CAM-04",
            track_b="B1",
            robust_score=0.80,
            reciprocal=True,
            assigned=True,
            status="MATCH_PROPOSED",
        )
        events = self.machine.observe_proposal(
            cycle=2,
            timestamp_ns=2_000_001,
            room="Devs",
            camera_a="CAM-01",
            track_a="A2",
            camera_b="CAM-04",
            track_b="B1",
            robust_score=0.81,
            reciprocal=True,
            assigned=True,
            status="MATCH_PROPOSED",
        )
        self.machine.end_cycle(cycle=2, timestamp_ns=2_000_002)
        self.assertEqual(events[0].event, GLOBAL_SHADOW_AMBIGUITY)
        self.assertEqual(events[0].decision, "same_camera_overlap_reject")
        self.assertEqual(self.machine.snapshot()["global_shadow_conflicts"], 0)
        self.assertEqual(
            self.machine.records[0].pair_key,
            (("CAM-01", "A1"), ("CAM-04", "B1")),
        )

    def test_active_current_member_rejects_second_same_camera_member(self) -> None:
        self.confirm_pair(
            (1, 2, 3), a=("CAM-01", "A1"), b=("CAM-04", "A4")
        )
        for cycle in (4, 5):
            self.machine.begin_cycle(cycle)
            events = self.machine.observe_proposal(
                cycle=cycle,
                timestamp_ns=cycle,
                room="Devs",
                camera_a="CAM-01",
                track_a="A2",
                camera_b="CAM-04",
                track_b="A4",
                robust_score=0.99,
                reciprocal=True,
                assigned=True,
                status="MATCH_PROPOSED",
                active_camera_members=(
                    ("CAM-01", "A1"),
                    ("CAM-01", "A2"),
                    ("CAM-04", "A4"),
                ),
            )
            self.machine.end_cycle(cycle=cycle, timestamp_ns=cycle)
            self.assertEqual(events[0].event, GLOBAL_SHADOW_AMBIGUITY)
            self.assertEqual(
                events[0].decision, "global_same_camera_member_reject"
            )
        self.assertEqual(
            self.machine.records[0].pair_key,
            (("CAM-01", "A1"), ("CAM-04", "A4")),
        )
        snapshot = self.machine.snapshot()
        self.assertEqual(snapshot["global_same_camera_member_reject"], 2)
        self.assertEqual(snapshot["global_shadow_successor_attaches"], 0)
        self.assertEqual(snapshot["global_shadow_conflicts"], 0)

    def test_crossing_noise_between_two_owners_never_merges_or_conflicts(self) -> None:
        self.confirm_two_people()
        events = self.observe(4, a=("CAM-01", "A1"), b=("CAM-04", "B4"))
        self.assertEqual(events[0].event, GLOBAL_SHADOW_AMBIGUITY)
        self.assertEqual(events[0].state, "AMBIGUITY_NO_ACTION")
        self.assertEqual(events[0].decision, "two_owner_ambiguity")
        self.assertIn("GSH-000001", events[0].current_members)
        self.assertIn("GSH-000002", events[0].current_members)
        self.assertEqual(len(self.machine.records), 2)
        self.assertEqual(self.machine.snapshot()["global_shadow_conflicts"], 0)

    def test_wrong_cross_pair_does_not_expire_two_confirmed_identities(self) -> None:
        self.confirm_two_people()
        self.observe(4, a=("CAM-01", "B1"), b=("CAM-04", "A4"))
        for cycle in (5, 6, 7, 8):
            self.empty_cycle(cycle)
        snap = self.machine.snapshot()
        self.assertEqual(snap["global_shadow_confirmed"], 2)
        self.assertEqual(snap["global_shadow_expired"], 0)
        self.assertEqual(snap["global_shadow_conflicts"], 0)

    def test_provisional_expiry_cannot_steal_or_expire_confirmed_identity(self) -> None:
        self.confirm_pair(
            (1, 2, 3), a=("CAM-01", "A1"), b=("CAM-04", "A4")
        )
        self.observe(4, a=("CAM-01", "B1"), b=("CAM-04", "B4"))
        self.observe(5, a=("CAM-01", "A1"), b=("CAM-04", "A4"))
        self.observe(6, a=("CAM-01", "A1"), b=("CAM-04", "A4"))
        expiry = self.observe(7, a=("CAM-01", "A1"), b=("CAM-04", "A4"))
        expiry_event = next(
            event for event in expiry if event.decision == "provisional_expire"
        )
        self.assertEqual(expiry_event.cycle, 7)
        self.assertIn("history:GSH-000002", expiry_event.owner_a)
        self.assertIn("GSH-000002:", expiry_event.current_members)
        self.assertIn("CAM-01/B1=4", expiry_event.last_seen_cycles)
        events = self.observe(8, a=("CAM-01", "B1"), b=("CAM-04", "A4"))
        self.assertEqual(events[0].decision, "historical_alias_reject")
        confirmed = [
            row for row in self.machine.records if row.state == CONFIRMED_SHADOW
        ]
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(
            confirmed[0].pair_key, (("CAM-01", "A1"), ("CAM-04", "A4"))
        )

    def test_expired_exact_pair_can_form_new_bounded_hypothesis(self) -> None:
        self.observe(1)
        for cycle in (2, 3, 4):
            self.empty_cycle(cycle)
        self.assertEqual(self.machine.records[0].state, EXPIRED_SHADOW)
        events = self.observe(5)
        self.assertEqual(events[0].decision, "create_new")
        self.assertEqual(self.machine.records[-1].shadow_global_id, "GSH-000002")

    def test_stale_alias_cannot_steal_current_member_back(self) -> None:
        self.confirm_pair(
            (1, 2, 3), a=("CAM-01", "A1"), b=("CAM-04", "A4")
        )
        self.observe(4, a=("CAM-01", "A2"), b=("CAM-04", "A4"))
        self.observe(5, a=("CAM-01", "A2"), b=("CAM-04", "A4"))
        events = self.observe(6, a=("CAM-01", "A1"), b=("CAM-04", "A4"))
        self.assertEqual(events[0].decision, "historical_alias_reject")
        self.assertEqual(
            self.machine.records[0].pair_key,
            (("CAM-01", "A2"), ("CAM-04", "A4")),
        )

    def test_both_cameras_fragment_but_one_current_member_per_camera_remains(self) -> None:
        self.confirm_two_people()
        cycle = 4
        for old_a, old_b, new_a, new_b in (
            ("A1", "A4", "A2", "A5"),
            ("B1", "B4", "B2", "B5"),
        ):
            for _ in range(2):
                self.observe(
                    cycle,
                    a=("CAM-01", new_a),
                    b=("CAM-04", old_b),
                )
                cycle += 1
            for _ in range(2):
                self.observe(
                    cycle,
                    a=("CAM-01", new_a),
                    b=("CAM-04", new_b),
                )
                cycle += 1
        for record in self.machine.records:
            self.assertEqual({member[0] for member in record.pair_key}, {"CAM-01", "CAM-04"})
            self.assertEqual(len(record.pair_key), 2)
        self.assertEqual(self.machine.snapshot()["global_shadow_member_tracks"], 4)

    def test_exact_live_failure_timeline_confirms_b_and_ignores_cross_pair(self) -> None:
        machine = GlobalShadowStateMachineV1(
            confirm_observations=3,
            confirm_consecutive=3,
            expire_provisional_after_missed_cycles=6,
            successor_confirm_observations=2,
        )
        proposals = {
            5: (("CAM-01", "A1"), ("CAM-04", "X1")),
            11: (("CAM-01", "A1"), ("CAM-04", "X1")),
            12: (("CAM-01", "A1"), ("CAM-04", "X1")),
            22: (("CAM-01", "A1"), ("CAM-04", "X3")),
            23: (("CAM-01", "A1"), ("CAM-04", "X3")),
            24: (("CAM-01", "A1"), ("CAM-04", "X3")),
            41: (("CAM-01", "B1"), ("CAM-04", "Y1")),
            42: (("CAM-01", "B1"), ("CAM-04", "Y1")),
            44: (("CAM-01", "B1"), ("CAM-04", "Y1")),
            45: (("CAM-01", "B1"), ("CAM-04", "Y1")),
            47: (("CAM-01", "B1"), ("CAM-04", "X3")),
            52: (("CAM-01", "B1"), ("CAM-04", "X3")),
        }
        decisions = []
        for cycle in range(1, 55):
            machine.begin_cycle(cycle)
            if cycle in proposals:
                a, b = proposals[cycle]
                decisions.extend(
                    machine.observe_proposal(
                        cycle=cycle,
                        timestamp_ns=cycle,
                        room="Devs",
                        camera_a=a[0],
                        track_a=a[1],
                        camera_b=b[0],
                        track_b=b[1],
                        robust_score=0.5,
                        reciprocal=True,
                        assigned=True,
                        status="MATCH_PROPOSED",
                    )
                )
            decisions.extend(machine.end_cycle(cycle=cycle, timestamp_ns=cycle))
        snap = machine.snapshot()
        self.assertEqual(snap["global_shadow_created"], 2)
        self.assertEqual(snap["global_shadow_confirmed"], 2)
        self.assertEqual(snap["global_shadow_conflicts"], 0)
        self.assertEqual(snap["global_shadow_expired"], 0)
        self.assertEqual(snap["global_shadow_active"], 2)
        self.assertEqual(snap["global_shadow_member_tracks"], 4)
        self.assertTrue(any(row.decision == "two_owner_ambiguity" for row in decisions))

    def test_real_room_mismatch_still_emits_conflict(self) -> None:
        self.observe(1)
        events = self.observe(2, room="Other")
        self.assertEqual(events[0].event, GLOBAL_SHADOW_CONFLICT)
        self.assertEqual(events[0].state, CONFLICT_PENDING)
        self.assertEqual(events[0].cycle, 2)
        self.assertEqual(events[0].owner_a, "GSH-000001")
        self.assertEqual(self.machine.records[0].consecutive_count, 0)
        self.assertIn("GSH-000001:", events[0].current_members)
        self.assertIn("CAM-01/A1=1", events[0].last_seen_cycles)
        self.assertEqual(self.machine.snapshot()["global_shadow_conflicts"], 1)

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

    def test_best_score_is_monotonic_across_successor(self) -> None:
        self.observe(1, score=0.7)
        self.observe(2, a=("CAM-01", "A2"), score=0.9)
        self.observe(3, a=("CAM-01", "A2"), score=0.9)
        self.observe(4, a=("CAM-01", "A2"), b=("CAM-04", "B2"), score=0.8)
        self.observe(5, a=("CAM-01", "A2"), b=("CAM-04", "B2"), score=0.8)
        record = self.machine.records[0]
        self.assertAlmostEqual(record.first_score, 0.7)
        self.assertAlmostEqual(record.last_score, 0.8)
        self.assertAlmostEqual(record.best_score, 0.9)

    def test_bounded_decision_context_is_serialized_without_embeddings(self) -> None:
        events = self.observe(1)
        with tempfile.TemporaryDirectory() as directory:
            tsv = Path(directory) / "step5.tsv"
            worker = V11GlobalShadowWorkerV1(tsv_path=tsv)
            worker.start()
            worker._write_events(events)
            worker.close()
            with tsv.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                self.assertEqual(tuple(reader.fieldnames or ()), TSV_COLUMNS)
                rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["decision"], "create_new")
        self.assertEqual(rows[0]["cycle"], "1")
        self.assertLessEqual(len(rows[0]["current_members"]), 512)
        self.assertLessEqual(len(rows[0]["last_seen_cycles"]), 512)
        self.assertFalse(any("embedding" in field.lower() for field in TSV_COLUMNS))

    def test_no_production_identity_fields_exist(self) -> None:
        self.observe(1)
        record = self.machine.records[0]
        self.assertFalse(hasattr(record, "global_id"))
        self.assertFalse(hasattr(record, "room_id"))
        self.assertFalse(hasattr(record, "face_id"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
