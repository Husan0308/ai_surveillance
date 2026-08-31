#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = ROOT / "scripts/check_camera_v11_step7_cam01_cam04_one_person_v1.py"
SPEC = importlib.util.spec_from_file_location("step7_checker", CHECKER_PATH)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)

GLOBAL_COLUMNS = (
    "timestamp",
    "event",
    "shadow_global_id",
    "room",
    "camera_a",
    "track_a",
    "camera_b",
    "track_b",
    "proposal_count",
    "consecutive_count",
    "state",
    "robust_score",
    "status",
)
VERIFY_COLUMNS = (
    "timestamp",
    "event",
    "shadow_global_id",
    "room",
    "camera_a",
    "track_a",
    "camera_b",
    "track_b",
    "state",
    "clean_observations",
    "total_observations",
    "conflict_events",
    "conflict_streak",
    "robust_score",
    "status",
)


def write_tsv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class Step7OnePersonCheckerTests(unittest.TestCase):
    def _run(self, *, conflict: bool = False, second_verified: bool = False) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            display_log = root / "display.log"
            match_log = root / "match.log"
            pair_tsv = root / "pair.tsv"
            match_tsv = root / "match.tsv"
            global_tsv = root / "global.tsv"
            verify_tsv = root / "verify.tsv"
            for path in (display_log, pair_tsv, match_tsv):
                path.write_text("placeholder\n", encoding="utf-8")

            created = 2 if second_verified else 1
            confirmed = 2 if second_verified else 1
            active = 2 if second_verified else 1
            member_tracks = 4 if second_verified else 2
            conflicts = 1 if conflict else 0
            match_log.write_text(
                "CAMERA_V11_STEP5_GLOBAL_SHADOW_V1 "
                f"created={created} provisional=0 confirmed={confirmed} observations=8 "
                f"conflicts={conflicts} expired=0 active={active} member_tracks={member_tracks} "
                "queue_pending=0 queue_dropped=0 events_written=8 state_p50=0.010ms "
                "state_p95=0.020ms worker_errors=0\n"
                "CAMERA_V11_STEP6_GLOBAL_VERIFY_V1 "
                f"records_created={created} pending=0 verified={confirmed} hold=0 expired=0 "
                f"verified_total={confirmed} hold_events=0 recovered_total=0 persistent_conflicts=0 "
                "verify_events=2 events_written=2 verify_p50=0.005ms verify_p95=0.010ms "
                "verify_worker_errors=0 geometry_enabled=0 production_global_id=0 room_id=0 face=0 handoff=0\n",
                encoding="utf-8",
            )

            global_rows = [
                {
                    "timestamp": "1",
                    "event": "GLOBAL_SHADOW_CONFIRM",
                    "shadow_global_id": "GSH-000001",
                    "room": "Devs",
                    "camera_a": "CAM-01",
                    "track_a": "CAM-01-T00001",
                    "camera_b": "CAM-04",
                    "track_b": "CAM-04-T00001",
                    "proposal_count": "3",
                    "consecutive_count": "3",
                    "state": "CONFIRMED_SHADOW",
                    "robust_score": "0.72",
                    "status": "CONFIRMED_SHADOW",
                }
            ]
            if conflict:
                global_rows.append(
                    {
                        **global_rows[0],
                        "timestamp": "2",
                        "event": "GLOBAL_SHADOW_CONFLICT",
                        "shadow_global_id": "",
                        "track_b": "CAM-04-T00002",
                        "state": "CONFLICT_PENDING",
                        "status": "CONFLICT_PENDING",
                    }
                )
            if second_verified:
                global_rows.append(
                    {
                        **global_rows[0],
                        "timestamp": "3",
                        "shadow_global_id": "GSH-000002",
                        "track_a": "CAM-01-T00002",
                        "track_b": "CAM-04-T00002",
                    }
                )
            write_tsv(global_tsv, GLOBAL_COLUMNS, global_rows)

            verify_rows = [
                {
                    "timestamp": "2",
                    "event": "GLOBAL_VERIFY_PASS",
                    "shadow_global_id": "GSH-000001",
                    "room": "Devs",
                    "camera_a": "CAM-01",
                    "track_a": "CAM-01-T00001",
                    "camera_b": "CAM-04",
                    "track_b": "CAM-04-T00001",
                    "state": "VERIFIED_SHADOW",
                    "clean_observations": "3",
                    "total_observations": "3",
                    "conflict_events": "0",
                    "conflict_streak": "0",
                    "robust_score": "0.72",
                    "status": "VERIFIED_SHADOW",
                }
            ]
            if second_verified:
                verify_rows.append(
                    {
                        **verify_rows[0],
                        "timestamp": "4",
                        "shadow_global_id": "GSH-000002",
                        "track_a": "CAM-01-T00002",
                        "track_b": "CAM-04-T00002",
                    }
                )
            write_tsv(verify_tsv, VERIFY_COLUMNS, verify_rows)

            argv = [
                str(CHECKER_PATH),
                "--display-log",
                str(display_log),
                "--match-log",
                str(match_log),
                "--pair-tsv",
                str(pair_tsv),
                "--match-tsv",
                str(match_tsv),
                "--global-tsv",
                str(global_tsv),
                "--verify-tsv",
                str(verify_tsv),
            ]
            fake_prior = mock.Mock(returncode=0, stdout="V11_STEP6_GLOBAL_VERIFY_V1 RESULT=PASS\n")
            output = io.StringIO()
            with mock.patch.object(checker.subprocess, "run", return_value=fake_prior), mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
                result = checker.main()
            return result, output.getvalue()

    def test_clean_one_person_pair_passes(self) -> None:
        result, output = self._run()
        self.assertEqual(result, 0, output)
        self.assertIn("V11_STEP7_CAM01_CAM04_ONE_PERSON_V1 RESULT=PASS", output)
        self.assertIn("verified_ids=1", output)

    def test_conflict_fails(self) -> None:
        result, output = self._run(conflict=True)
        self.assertEqual(result, 1)
        self.assertIn("global_conflict_rows=1/expected0", output)

    def test_two_verified_ids_fail(self) -> None:
        result, output = self._run(second_verified=True)
        self.assertEqual(result, 1)
        self.assertIn("verified_ids=2/expected1", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
