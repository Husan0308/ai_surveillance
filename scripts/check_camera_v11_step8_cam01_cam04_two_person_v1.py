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


def _read_phase_markers(path: Path) -> dict[str, tuple[int, int]]:
    markers: dict[str, tuple[int, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != ("marker", "global_rows", "verify_rows"):
            raise ValueError("phase_marker_columns_invalid")
        for row in reader:
            marker = str(row.get("marker", "")).strip()
            if not marker:
                continue
            markers[marker] = (int(row["global_rows"]), int(row["verify_rows"]))
    return markers


def _slice(rows: list[dict[str, str]], start: int, end: int) -> list[dict[str, str]]:
    start = max(0, int(start))
    end = max(start, int(end))
    return rows[start:end]


def _ids(rows: list[dict[str, str]], event: str | None = None) -> set[str]:
    result: set[str] = set()
    for row in rows:
        if event is not None and row.get("event") != event:
            continue
        value = str(row.get("shadow_global_id", "")).strip()
        if value:
            result.add(value)
    return result


def _event_rows(rows: list[dict[str, str]], event: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("event") == event]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--display-log", required=True, type=Path)
    parser.add_argument("--match-log", required=True, type=Path)
    parser.add_argument("--pair-tsv", required=True, type=Path)
    parser.add_argument("--match-tsv", required=True, type=Path)
    parser.add_argument("--global-tsv", required=True, type=Path)
    parser.add_argument("--verify-tsv", required=True, type=Path)
    parser.add_argument("--phase-markers", required=True, type=Path)
    parser.add_argument("--warmup-windows", type=int, default=2)
    parser.add_argument("--min-isolation-observations", type=int, default=3)
    args = parser.parse_args()

    reasons: list[str] = []

    # First keep every generic Step1-6 regression/safety check. Step8 only adds
    # controlled two-person ground-truth invariants on top of the existing chain.
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

    # Two physical people must produce exactly two live global hypotheses. Any
    # third identity, expiry, hold, or persistent conflict is a real failure; do
    # not hide it by relaxing expected counts.
    expected_step5 = {
        "created": 2,
        "provisional": 0,
        "confirmed": 2,
        "conflicts": 0,
        "expired": 0,
        "active": 2,
        "member_tracks": 4,
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
        "records_created": 2,
        "pending": 0,
        "verified": 2,
        "hold": 0,
        "expired": 0,
        "verified_total": 2,
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

    if not args.global_tsv.is_file():
        reasons.append("global_tsv_missing")
        global_rows: list[dict[str, str]] = []
    else:
        global_rows = _read_tsv(args.global_tsv)
    if not args.verify_tsv.is_file():
        reasons.append("verify_tsv_missing")
        verify_rows: list[dict[str, str]] = []
    else:
        verify_rows = _read_tsv(args.verify_tsv)

    try:
        markers = _read_phase_markers(args.phase_markers)
    except Exception as exc:
        markers = {}
        reasons.append(f"phase_markers_invalid:{type(exc).__name__}")

    required_markers = (
        "A_END",
        "B_END",
        "C_END",
        "D_START",
        "D_END",
        "E_START",
        "E_END",
    )
    for marker in required_markers:
        if marker not in markers:
            reasons.append(f"phase_marker_missing:{marker}")

    # Every Step5/6 identity event in this dedicated run must be the Devs overlap
    # pair. Other rooms/people would invalidate the manual ground truth protocol.
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

    conflict_rows = _event_rows(global_rows, "GLOBAL_SHADOW_CONFLICT")
    expire_rows = _event_rows(global_rows, "GLOBAL_SHADOW_EXPIRE")
    hold_rows = _event_rows(verify_rows, "GLOBAL_VERIFY_HOLD")
    recover_rows = _event_rows(verify_rows, "GLOBAL_VERIFY_RECOVER")
    persistent_rows = _event_rows(verify_rows, "GLOBAL_VERIFY_CONFLICT_PERSISTENT")
    if conflict_rows:
        reasons.append(f"global_conflict_rows={len(conflict_rows)}/expected0")
    if expire_rows:
        reasons.append(f"global_expire_rows={len(expire_rows)}/expected0")
    if hold_rows:
        reasons.append(f"verify_hold_rows={len(hold_rows)}/expected0")
    if recover_rows:
        reasons.append(f"verify_recover_rows={len(recover_rows)}/expected0")
    if persistent_rows:
        reasons.append(f"verify_persistent_rows={len(persistent_rows)}/expected0")

    person_a_id = ""
    person_b_id = ""

    if all(marker in markers for marker in required_markers):
        a_g_end, a_v_end = markers["A_END"]
        b_g_end, b_v_end = markers["B_END"]
        c_g_end, c_v_end = markers["C_END"]
        d_g_start, d_v_start = markers["D_START"]
        d_g_end, d_v_end = markers["D_END"]
        e_g_start, e_v_start = markers["E_START"]
        e_g_end, e_v_end = markers["E_END"]

        ordered = [a_g_end, b_g_end, c_g_end, d_g_start, d_g_end, e_g_start, e_g_end]
        if ordered != sorted(ordered):
            reasons.append("global_phase_markers_not_monotonic")
        ordered_v = [a_v_end, b_v_end, c_v_end, d_v_start, d_v_end, e_v_start, e_v_end]
        if ordered_v != sorted(ordered_v):
            reasons.append("verify_phase_markers_not_monotonic")
        if e_g_end > len(global_rows) or e_v_end > len(verify_rows):
            reasons.append("phase_marker_exceeds_tsv_rows")

        # Phase A: only Person A exists. This uniquely labels the first confirmed
        # and verified global ID as Person A without using face recognition.
        a_global = _slice(global_rows, 0, a_g_end)
        a_verify = _slice(verify_rows, 0, a_v_end)
        a_confirm = _ids(a_global, "GLOBAL_SHADOW_CONFIRM")
        a_pass = _ids(a_verify, "GLOBAL_VERIFY_PASS")
        if len(a_confirm) != 1:
            reasons.append(f"phase_a_confirmed_ids={len(a_confirm)}/expected1")
        if len(a_pass) != 1:
            reasons.append(f"phase_a_verified_ids={len(a_pass)}/expected1")
        if len(a_confirm) == 1 and len(a_pass) == 1:
            if a_confirm != a_pass:
                reasons.append("phase_a_confirm_verify_mismatch")
            else:
                person_a_id = next(iter(a_confirm))

        # Phase B: Person A remains visible and Person B joins, separated. The one
        # newly confirmed/verified ID is therefore ground-truth Person B. Keeping A
        # present also blocks unsafe same-camera successor reuse for B.
        b_global = _slice(global_rows, a_g_end, b_g_end)
        b_verify = _slice(verify_rows, a_v_end, b_v_end)
        b_new_confirm = _ids(b_global, "GLOBAL_SHADOW_CONFIRM") - ({person_a_id} if person_a_id else set())
        b_new_pass = _ids(b_verify, "GLOBAL_VERIFY_PASS") - ({person_a_id} if person_a_id else set())
        if len(b_new_confirm) != 1:
            reasons.append(f"phase_b_new_confirmed_ids={len(b_new_confirm)}/expected1")
        if len(b_new_pass) != 1:
            reasons.append(f"phase_b_new_verified_ids={len(b_new_pass)}/expected1")
        if len(b_new_confirm) == 1 and len(b_new_pass) == 1:
            if b_new_confirm != b_new_pass:
                reasons.append("phase_b_confirm_verify_mismatch")
            else:
                person_b_id = next(iter(b_new_confirm))
        if person_a_id and person_b_id and person_a_id == person_b_id:
            reasons.append("wrong_merge_person_a_person_b_same_global")

        expected_ids = {value for value in (person_a_id, person_b_id) if value}

        # Phase C is the stress section: both people cross/occlude. It is allowed
        # to change local tracker IDs, but it must not create a third global ID or
        # drive either verified identity into HOLD.
        c_global = _slice(global_rows, b_g_end, c_g_end)
        c_seen_ids = _ids(c_global)
        unexpected_c = c_seen_ids - expected_ids
        if unexpected_c:
            reasons.append("phase_c_new_global_ids=" + ",".join(sorted(unexpected_c)))

        # Phase D/E turn the manual scene into ground truth after the crossing.
        # When only A remains, every clean Step5 observation must still be A's ID;
        # then when only B remains, every clean observation must be B's ID. This
        # catches an ID swap that a simple final count of two IDs cannot detect.
        d_global = _slice(global_rows, d_g_start, d_g_end)
        d_observe = _event_rows(d_global, "GLOBAL_SHADOW_OBSERVE")
        d_ids = _ids(d_observe)
        if len(d_observe) < args.min_isolation_observations:
            reasons.append(
                f"phase_d_observations={len(d_observe)}/min{args.min_isolation_observations}"
            )
        if person_a_id and d_ids != {person_a_id}:
            reasons.append(
                "id_swap_after_crossing_phase_d_seen=" + ",".join(sorted(d_ids or {"NONE"}))
                + f"/expected={person_a_id}"
            )

        e_global = _slice(global_rows, e_g_start, e_g_end)
        e_observe = _event_rows(e_global, "GLOBAL_SHADOW_OBSERVE")
        e_ids = _ids(e_observe)
        if len(e_observe) < args.min_isolation_observations:
            reasons.append(
                f"phase_e_observations={len(e_observe)}/min{args.min_isolation_observations}"
            )
        if person_b_id and e_ids != {person_b_id}:
            reasons.append(
                "id_swap_after_crossing_phase_e_seen=" + ",".join(sorted(e_ids or {"NONE"}))
                + f"/expected={person_b_id}"
            )

        all_global_ids = _ids(global_rows)
        if len(expected_ids) == 2 and all_global_ids != expected_ids:
            reasons.append(
                "global_id_set=" + ",".join(sorted(all_global_ids))
                + "/expected=" + ",".join(sorted(expected_ids))
            )

        confirm_ids = _ids(global_rows, "GLOBAL_SHADOW_CONFIRM")
        pass_ids = _ids(verify_rows, "GLOBAL_VERIFY_PASS")
        if len(confirm_ids) != 2:
            reasons.append(f"global_confirmed_ids={len(confirm_ids)}/expected2")
        if len(pass_ids) != 2:
            reasons.append(f"global_verified_ids={len(pass_ids)}/expected2")
        if confirm_ids and pass_ids and confirm_ids != pass_ids:
            reasons.append("confirmed_verified_id_set_mismatch")

    if reasons:
        print(
            "V11_STEP8_CAM01_CAM04_TWO_PERSON_V1 RESULT=FAIL reasons="
            + ";".join(dict.fromkeys(reasons))
            + " physical_people_expected=2 verified_ids_expected=2 "
            "wrong_merge_expected=0 id_swap_expected=0 cross_person_successor_expected=0 "
            "production_global_id=0 identity_accuracy_proven=0"
        )
        return 1

    aliases: dict[str, set[tuple[tuple[str, str], tuple[str, str]]]] = {
        person_a_id: set(),
        person_b_id: set(),
    }
    for row in global_rows:
        shadow_id = str(row.get("shadow_global_id", "")).strip()
        if shadow_id not in aliases:
            continue
        pair = tuple(
            sorted(
                (
                    (str(row.get("camera_a", "")), str(row.get("track_a", ""))),
                    (str(row.get("camera_b", "")), str(row.get("track_b", ""))),
                )
            )
        )
        aliases[shadow_id].add(pair)  # type: ignore[arg-type]

    print(
        "V11_STEP8_CAM01_CAM04_TWO_PERSON_V1 RESULT=PASS "
        f"person_a_shadow_id={person_a_id} person_b_shadow_id={person_b_id} "
        "physical_people_expected=2 verified_ids=2 confirmed_ids=2 "
        f"person_a_pair_aliases={len(aliases[person_a_id])} "
        f"person_b_pair_aliases={len(aliases[person_b_id])} "
        "wrong_merge=0 id_swap=0 cross_person_successor=0 conflicts=0 holds=0 "
        "recoveries=0 expiries=0 current_members=4 production_global_id=0 "
        "identity_accuracy_proven=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
