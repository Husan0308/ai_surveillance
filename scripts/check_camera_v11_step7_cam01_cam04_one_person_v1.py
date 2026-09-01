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
    prior = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_camera_v11_step6_global_shadow_v1_log.py"),
            "--display-log", str(args.display_log),
            "--match-log", str(args.match_log),
            "--pair-tsv", str(args.pair_tsv),
            "--match-tsv", str(args.match_tsv),
            "--global-tsv", str(args.global_tsv),
            "--verify-tsv", str(args.verify_tsv),
            "--warmup-windows", str(args.warmup_windows),
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

    expected_step5 = {
        "created": 1,
        "provisional": 0,
        "confirmed": 1,
        "conflicts": 0,
        "expired": 0,
        "active": 1,
        "member_tracks": 2,
        "queue_dropped": 0,
        "worker_errors": 0,
    }
    for field, expected in expected_step5.items():
        actual = step5.get(field)
        if actual is None:
            reasons.append(f"step5_missing:{field}")
        elif actual != expected:
            reasons.append(f"step5_{field}={actual}/expected{expected}")

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

    global_rows = _read_tsv(args.global_tsv) if args.global_tsv.is_file() else []
    verify_rows = _read_tsv(args.verify_tsv) if args.verify_tsv.is_file() else []
    if not args.global_tsv.is_file():
        reasons.append("global_tsv_missing")
    if not args.verify_tsv.is_file():
        reasons.append("verify_tsv_missing")

    for label, rows in (("step5", global_rows), ("step6", verify_rows)):
        for row in rows:
            if row.get("room") != TARGET_ROOM:
                reasons.append(f"{label}_unexpected_room:{row.get('room','')}")
                break
            cameras = {row.get("camera_a", ""), row.get("camera_b", "")}
            if cameras != TARGET_CAMERAS:
                reasons.append(
                    f"{label}_unexpected_camera_pair:{row.get('camera_a','')}+{row.get('camera_b','')}"
                )
                break

    confirm_rows = [r for r in global_rows if r.get("event") == "GLOBAL_SHADOW_CONFIRM"]
    conflict_rows = [r for r in global_rows if r.get("event") == "GLOBAL_SHADOW_CONFLICT"]
    expire_rows = [r for r in global_rows if r.get("event") == "GLOBAL_SHADOW_EXPIRE"]
    if len(confirm_rows) != 1:
        reasons.append(f"global_confirm_rows={len(confirm_rows)}/expected1")
    if conflict_rows:
        reasons.append(f"global_conflict_rows={len(conflict_rows)}/expected0")
    if expire_rows:
        reasons.append(f"global_expire_rows={len(expire_rows)}/expected0")

    pass_rows = [r for r in verify_rows if r.get("event") == "GLOBAL_VERIFY_PASS"]
    hold_rows = [r for r in verify_rows if r.get("event") == "GLOBAL_VERIFY_HOLD"]
    recover_rows = [r for r in verify_rows if r.get("event") == "GLOBAL_VERIFY_RECOVER"]
    persistent_rows = [
        r for r in verify_rows if r.get("event") == "GLOBAL_VERIFY_CONFLICT_PERSISTENT"
    ]
    if len(pass_rows) != 1:
        reasons.append(f"verify_pass_rows={len(pass_rows)}/expected1")
    if hold_rows:
        reasons.append(f"verify_hold_rows={len(hold_rows)}/expected0")
    if recover_rows:
        reasons.append(f"verify_recover_rows={len(recover_rows)}/expected0")
    if persistent_rows:
        reasons.append(f"verify_persistent_rows={len(persistent_rows)}/expected0")

    confirmed_ids = {
        r.get("shadow_global_id", "") for r in confirm_rows if r.get("shadow_global_id")
    }
    verified_ids = {
        r.get("shadow_global_id", "") for r in pass_rows if r.get("shadow_global_id")
    }
    if len(confirmed_ids) != 1:
        reasons.append(f"confirmed_ids={len(confirmed_ids)}/expected1")
    if len(verified_ids) != 1:
        reasons.append(f"verified_ids={len(verified_ids)}/expected1")
    if confirmed_ids and verified_ids and confirmed_ids != verified_ids:
        reasons.append("confirmed_verified_id_mismatch")

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
        if any(not camera or not track for camera, track in pair):
            reasons.append("empty_track_pair_for_target_id")
            continue
        pair_keys.add(pair)  # type: ignore[arg-type]
    if target_ids and not pair_keys:
        reasons.append("target_pair_evidence_missing")

    if reasons:
        print(
            "V11_STEP7_CAM01_CAM04_ONE_PERSON_V1 RESULT=FAIL reasons="
            + ";".join(dict.fromkeys(reasons))
            + " physical_people_expected=1 verified_ids_expected=1 "
            "production_global_id=0 identity_accuracy_proven=0"
        )
        return 1

    shadow_id = next(iter(verified_ids))
    print(
        "V11_STEP7_CAM01_CAM04_ONE_PERSON_V1 RESULT=PASS "
        f"shadow_global_id={shadow_id} physical_people_expected=1 "
        f"verified_ids=1 confirmed_ids=1 local_pair_aliases={len(pair_keys)} "
        "conflicts=0 holds=0 recoveries=0 expiries=0 current_members=2 "
        "production_global_id=0 identity_accuracy_proven=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
