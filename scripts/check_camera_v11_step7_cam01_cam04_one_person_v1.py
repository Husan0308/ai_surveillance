#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEP5_PREFIX = "CAMERA_V11_STEP5_GLOBAL_SHADOW_V1 "
STEP6_PREFIX = "CAMERA_V11_STEP6_GLOBAL_VERIFY_V1 "
INTEGER = re.compile(r"\b([a-z][a-z0-9_]+)=(\d+)\b")

TARGET_CAMERAS = {"CAM-01", "CAM-04"}
TARGET_ROOM = "Devs"


def _final_metrics(text: str, prefix: str) -> dict[str, int]:
    lines = [line for line in text.splitlines() if line.startswith(prefix)]
    if not lines:
        return {}
    return {name: int(value) for name, value in INTEGER.findall(lines[-1])}


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--display-log", required=True, type=Path)
    parser.add_argument("--match-log", required=True, type=Path)
    parser.add_argument("--pair-tsv", required=True, type=Path)
    parser.add_argument("--match-tsv", required=True, type=Path)
    parser.add_argument("--global-tsv", required=True, type=Path)
    parser.add_argument("--verify-tsv", required=True, type=Path)
    parser.add_argument("--warmup-windows", type=int, default=2)
    args = parser.parse_args()

    reasons: list[str] = []

    # Step7 is an acceptance layer only. First require the full Step6 chain to pass.
    prior = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_camera_v11_step6_global_shadow_v1_log.py"),
            "--display-log",
            str(args.display_log),
            "--match-log",
            str(args.match_log),
            "--pair-tsv",
            str(args.pair_tsv),
            "--match-tsv",
            str(args.match_tsv),
            "--global-tsv",
            str(args.global_tsv),
            "--verify-tsv",
            str(args.verify_tsv),
            "--warmup-windows",
            str(args.warmup_windows),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(prior.stdout, end="")
    if prior.returncode != 0:
        reasons.append("step6_or_prior_regression")

    if not args.match_log.is_file():
        text = ""
        reasons.append("match_log_missing")
    else:
        text = args.match_log.read_text(encoding="utf-8", errors="replace")

    step5 = _final_metrics(text, STEP5_PREFIX)
    step6 = _final_metrics(text, STEP6_PREFIX)
    if not step5:
        reasons.append("step5_metrics_missing")
    if not step6:
        reasons.append("step6_metrics_missing")

    # One physical person visible in both Devs cameras must collapse to one stable
    # Step5 shadow identity with exactly two active member tracks.
    if step5:
        expected_step5 = {
            "global_shadow_created": 1,
            "global_shadow_provisional": 0,
            "global_shadow_confirmed": 1,
            "global_shadow_conflicts": 0,
            "global_shadow_expired": 0,
            "global_shadow_active": 1,
            "global_shadow_member_tracks": 2,
            "queue_dropped": 0,
            "worker_errors": 0,
        }
        # Runtime prints short field names, not internal snapshot names.
        aliases = {
            "global_shadow_created": "created",
            "global_shadow_provisional": "provisional",
            "global_shadow_confirmed": "confirmed",
            "global_shadow_conflicts": "conflicts",
            "global_shadow_expired": "expired",
            "global_shadow_active": "active",
            "global_shadow_member_tracks": "member_tracks",
            "queue_dropped": "queue_dropped",
            "worker_errors": "worker_errors",
        }
        for internal, expected in expected_step5.items():
            field = aliases[internal]
            actual = step5.get(field)
            if actual is None:
                reasons.append(f"step5_missing:{field}")
            elif actual != expected:
                reasons.append(f"step5_{field}={actual}/expected{expected}")

    # Step6 must end with exactly one verified record and no hold/recovery path.
    if step6:
        expected_step6 = {
            "records_created": 1,
            "pending": 0,
            "verified": 1,
            "hold": 0,
            "expired": 0,
            "verified_total": 1,
            "hold_events": 0,
            "recovered_total": 0,
            "persistent_conflicts": 0,
            "verify_worker_errors": 0,
        }
        for field, expected in expected_step6.items():
            actual = step6.get(field)
            if actual is None:
                reasons.append(f"step6_missing:{field}")
            elif actual != expected:
                reasons.append(f"step6_{field}={actual}/expected{expected}")

    global_rows: list[dict[str, str]] = []
    verify_rows: list[dict[str, str]] = []
    if not args.global_tsv.is_file():
        reasons.append("global_tsv_missing")
    else:
        global_rows = _read_tsv(args.global_tsv)
    if not args.verify_tsv.is_file():
        reasons.append("verify_tsv_missing")
    else:
        verify_rows = _read_tsv(args.verify_tsv)

    # Strict ground-truth acceptance invariant: every Step5/6 identity event in this
    # dedicated run must be Devs CAM-01<->CAM-04. Any other pair means the run did
    # not represent one stable person/pair.
    for label, rows in (("step5", global_rows), ("step6", verify_rows)):
        for row in rows:
            cameras = {row.get("camera_a", ""), row.get("camera_b", "")}
            if row.get("room") != TARGET_ROOM:
                reasons.append(f"{label}_unexpected_room:{row.get('room','')}")
                break
            if cameras != TARGET_CAMERAS:
                reasons.append(
                    f"{label}_unexpected_camera_pair:{row.get('camera_a','')}+{row.get('camera_b','')}"
                )
                break

    confirm_rows = [row for row in global_rows if row.get("event") == "GLOBAL_SHADOW_CONFIRM"]
    conflict_rows = [row for row in global_rows if row.get("event") == "GLOBAL_SHADOW_CONFLICT"]
    expire_rows = [row for row in global_rows if row.get("event") == "GLOBAL_SHADOW_EXPIRE"]
    if len(confirm_rows) != 1:
        reasons.append(f"global_confirm_rows={len(confirm_rows)}/expected1")
    if conflict_rows:
        reasons.append(f"global_conflict_rows={len(conflict_rows)}/expected0")
    if expire_rows:
        reasons.append(f"global_expire_rows={len(expire_rows)}/expected0")

    pass_rows = [row for row in verify_rows if row.get("event") == "GLOBAL_VERIFY_PASS"]
    hold_rows = [row for row in verify_rows if row.get("event") == "GLOBAL_VERIFY_HOLD"]
    recover_rows = [row for row in verify_rows if row.get("event") == "GLOBAL_VERIFY_RECOVER"]
    persistent_rows = [
        row for row in verify_rows if row.get("event") == "GLOBAL_VERIFY_CONFLICT_PERSISTENT"
    ]
    if len(pass_rows) != 1:
        reasons.append(f"verify_pass_rows={len(pass_rows)}/expected1")
    if hold_rows:
        reasons.append(f"verify_hold_rows={len(hold_rows)}/expected0")
    if recover_rows:
        reasons.append(f"verify_recover_rows={len(recover_rows)}/expected0")
    if persistent_rows:
        reasons.append(f"verify_persistent_rows={len(persistent_rows)}/expected0")

    confirmed_ids = {row.get("shadow_global_id", "") for row in confirm_rows if row.get("shadow_global_id")}
    verified_ids = {row.get("shadow_global_id", "") for row in pass_rows if row.get("shadow_global_id")}
    if len(confirmed_ids) != 1:
        reasons.append(f"confirmed_ids={len(confirmed_ids)}/expected1")
    if len(verified_ids) != 1:
        reasons.append(f"verified_ids={len(verified_ids)}/expected1")
    if confirmed_ids and verified_ids and confirmed_ids != verified_ids:
        reasons.append("confirmed_verified_id_mismatch")

    # The one accepted ID must keep one canonical local-track pair throughout the run.
    target_ids = confirmed_ids | verified_ids
    pair_keys: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    for row in global_rows + verify_rows:
        if row.get("shadow_global_id") not in target_ids:
            continue
        pair = tuple(
            sorted(
                (
                    (row.get("camera_a", ""), row.get("track_a", "")),
                    (row.get("camera_b", ""), row.get("track_b", "")),
                )
            )
        )
        pair_keys.add(pair)  # type: ignore[arg-type]
    if len(pair_keys) != 1:
        reasons.append(f"canonical_pairs={len(pair_keys)}/expected1")

    if reasons:
        print(
            "V11_STEP7_CAM01_CAM04_ONE_PERSON_V1 RESULT=FAIL reasons="
            + ";".join(dict.fromkeys(reasons))
            + " physical_people_expected=1 verified_ids_expected=1 "
            "production_global_id=0 identity_accuracy_proven=0"
        )
        return 1

    shadow_id = next(iter(verified_ids))
    pair = next(iter(pair_keys))
    print(
        "V11_STEP7_CAM01_CAM04_ONE_PERSON_V1 RESULT=PASS "
        f"shadow_global_id={shadow_id} "
        f"pair={pair[0][0]}:{pair[0][1]}+{pair[1][0]}:{pair[1][1]} "
        "physical_people_expected=1 verified_ids=1 confirmed_ids=1 "
        "conflicts=0 holds=0 recoveries=0 expiries=0 canonical_pairs=1 "
        "production_global_id=0 identity_accuracy_proven=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
