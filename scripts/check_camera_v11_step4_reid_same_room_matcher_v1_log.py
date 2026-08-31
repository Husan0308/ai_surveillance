#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v11.step4_reid_same_room_matcher_v1 import (
    ASSIGNMENT_CONFLICT,
    INSUFFICIENT,
    INVALID,
    LOW_MARGIN,
    LOW_SCORE,
    MATCH_PROPOSED,
    MATCH_STATUSES,
    NON_RECIPROCAL,
    STALE,
)
from services.camera_v11.step4_reid_same_room_shadow_v1 import TSV_COLUMNS


PREFIX = "CAMERA_V11_STEP4_REID_SAME_ROOM_MATCHER_V1 "
INTEGER = re.compile(r"\b([a-z][a-z0-9_]+)=(\d+)\b")
MILLISECOND = re.compile(r"\b(match_p50|match_p95)=([0-9.]+)ms")
COUNTERS = (
    "cycles",
    "matrices_built",
    "pairs_considered",
    "pairs_valid",
    "pairs_insufficient",
    "nonreciprocal",
    "low_margin",
    "low_score",
    "assignment_conflicts",
    "proposals",
    "unique_proposals",
    "proposal_changes",
    "stale",
    "invalid",
    "worker_errors",
)
EXPECTED_ROOMS = {
    frozenset(("CAM-01", "CAM-04")): "Devs",
    frozenset(("CAM-02", "CAM-05")): "Entrance",
    frozenset(("CAM-03", "CAM-06")): "Main Rooms",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--display-log", required=True, type=Path)
    parser.add_argument("--match-log", required=True, type=Path)
    parser.add_argument("--pair-tsv", required=True, type=Path)
    parser.add_argument("--match-tsv", required=True, type=Path)
    parser.add_argument("--warmup-windows", type=int, default=2)
    parser.add_argument("--match-p95-max-ms", type=float, default=2.0)
    args = parser.parse_args()
    if not args.display_log.is_file() or not args.match_log.is_file():
        print("V11_STEP4_REID_SAME_ROOM_MATCHER_V1 RESULT=FAIL reasons=missing_log global_id=0 room_id=0 face=0 handoff=0")
        return 2

    reasons: list[str] = []
    prior = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_camera_v11_step4_reid_pair_scorer_v1_log.py"),
            "--display-log",
            str(args.display_log),
            "--pair-log",
            str(args.match_log),
            "--tsv",
            str(args.pair_tsv),
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
        reasons.append("step3_pair_scorer_or_prior_regression")

    text = args.match_log.read_text(encoding="utf-8", errors="replace")
    markers = (
        "CAMERA_V11_STEP4_REID_SAME_ROOM_MATCHER_V1_ARCH",
        "mode=shadow diagnostic_no_merge=1 same_room_only=1 different_camera_only=1",
        "evidence=step3_robust_score step3_formula_changed=0 step3_status_required=VALID",
        "reciprocal_before_proposal=1 assignment=scipy_linear_sum_assignment_maximize",
        "assignment_eligible_only=1 one_to_one=1 deterministic=1",
        "worker=one async=1 dirty_slot=latest-only cadence=2.0s camera_queue=0",
        "camera_display_block=0 tracker_mutation=0 local_track_id_mutation=0",
        "global_id=0 room_id=0 face=0 handoff=0 hysteresis=0 identity_state=0",
        "CAMERA_V11_STEP4_REID_SAME_ROOM_MATCHER_V1_CONFIG",
        "thresholds=robust=off,row_margin=off,column_margin=off",
        "live_threshold_tuning=0 max_tracks_per_camera=8",
        "priority=CAM-01+CAM-04 other_rooms_production=0",
        "raw_embeddings_tsv=0",
    )
    for marker in markers:
        if marker not in text:
            reasons.append(f"missing_marker:{marker.split()[0]}")
    if "Traceback (most recent call last)" in text:
        reasons.append("runtime_traceback")
    if "V11_STEP4_REID_SAME_ROOM_MATCHER_SHADOW_ERROR" in text:
        reasons.append("matcher_worker_error")

    metric_lines = [line for line in text.splitlines() if line.startswith(PREFIX)]
    final_line = metric_lines[-1] if metric_lines else ""
    if not final_line:
        reasons.append("matcher_metrics_missing")
    counters = {name: int(value) for name, value in INTEGER.findall(final_line)}
    timings = {name: float(value) for name, value in MILLISECOND.findall(final_line)}
    for name in COUNTERS:
        if name not in counters:
            reasons.append(f"missing_counter:{name}")
    for name in ("match_p50", "match_p95"):
        if name not in timings:
            reasons.append(f"missing_timing:{name}")

    cycles = counters.get("cycles", 0)
    matrices = counters.get("matrices_built", 0)
    considered = counters.get("pairs_considered", 0)
    valid = counters.get("pairs_valid", 0)
    insufficient = counters.get("pairs_insufficient", 0)
    stale = counters.get("stale", 0)
    invalid = counters.get("invalid", -1)
    proposals = counters.get("proposals", 0)
    unique = counters.get("unique_proposals", 0)
    worker_errors = counters.get("worker_errors", -1)
    if cycles <= 0 or matrices <= 0 or considered <= 0 or valid <= 0:
        reasons.append("no_live_same_room_matrix_evidence")
    if considered != valid + insufficient + stale + max(0, invalid):
        reasons.append(
            f"structural_accounting={considered}!={valid}+{insufficient}+{stale}+{invalid}"
        )
    classified_valid = (
        counters.get("nonreciprocal", 0)
        + counters.get("low_margin", 0)
        + counters.get("low_score", 0)
        + counters.get("assignment_conflicts", 0)
        + proposals
    )
    if valid != classified_valid:
        reasons.append(f"valid_accounting={valid}!={classified_valid}")
    if counters.get("low_margin", -1) != 0 or counters.get("low_score", -1) != 0:
        reasons.append("diagnostic_thresholds_not_disabled")
    if proposals <= 0 or unique <= 0:
        reasons.append("no_live_shadow_proposals")
    if unique > proposals:
        reasons.append(f"unique_proposals={unique}/proposals={proposals}")
    if invalid != 0:
        reasons.append(f"pairs_invalid={invalid}")
    if worker_errors != 0:
        reasons.append(f"worker_errors={worker_errors}")
    if timings.get("match_p50", 0.0) <= 0.0:
        reasons.append("no_match_timing")
    if timings.get("match_p95", 1e9) > args.match_p95_max_ms:
        reasons.append(
            f"match_p95={timings.get('match_p95', -1):.3f}ms/max{args.match_p95_max_ms:.3f}"
        )

    rows: list[dict[str, str]] = []
    fieldnames: tuple[str, ...] = ()
    if not args.match_tsv.is_file():
        reasons.append("match_tsv_missing")
    else:
        with args.match_tsv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = tuple(reader.fieldnames or ())
            if fieldnames != TSV_COLUMNS:
                reasons.append("match_tsv_columns_invalid")
            rows = list(reader)
    if len(rows) != considered:
        reasons.append(f"match_tsv_rows={len(rows)}/considered={considered}")
    if any("embedding" in name.lower() for name in fieldnames):
        reasons.append("raw_embedding_column_forbidden")

    status_counts: defaultdict[str, int] = defaultdict(int)
    endpoints_by_cycle: defaultdict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        status = row.get("status", "")
        status_counts[status] += 1
        if status not in MATCH_STATUSES:
            reasons.append(f"invalid_status:{status}")
        camera_a = row.get("camera_a", "")
        camera_b = row.get("camera_b", "")
        pair = frozenset((camera_a, camera_b))
        expected_room = EXPECTED_ROOMS.get(pair)
        if camera_a == camera_b:
            reasons.append("same_camera_pair_in_tsv")
        if expected_room is None or row.get("room") != expected_room:
            reasons.append(f"cross_room_or_unknown_pair:{camera_a}+{camera_b}")
        if status == MATCH_PROPOSED:
            if row.get("reciprocal") != "1" or row.get("assigned") != "1":
                reasons.append("proposal_without_reciprocal_assignment")
            if int(row.get("proposal_seen", "0")) < 1 or int(row.get("proposal_consecutive", "0")) < 1:
                reasons.append("proposal_stability_missing")
            matrix_cycle = (row.get("cycle", ""), row.get("room", ""), "+".join(sorted(pair)))
            endpoints_by_cycle[matrix_cycle].extend(
                [(camera_a, row.get("track_a", "")), (camera_b, row.get("track_b", ""))]
            )
        elif row.get("assigned") == "1":
            reasons.append("nonproposal_marked_assigned")
    for endpoints in endpoints_by_cycle.values():
        if len(endpoints) != len(set(endpoints)):
            reasons.append("duplicate_endpoint_in_cycle")
            break
    if rows and not any(
        {row.get("camera_a"), row.get("camera_b")} == {"CAM-01", "CAM-04"}
        for row in rows
    ):
        reasons.append("no_cam01_cam04_matrix")
    expected_status_counts = {
        INSUFFICIENT: insufficient,
        NON_RECIPROCAL: counters.get("nonreciprocal", 0),
        LOW_MARGIN: counters.get("low_margin", 0),
        LOW_SCORE: counters.get("low_score", 0),
        ASSIGNMENT_CONFLICT: counters.get("assignment_conflicts", 0),
        MATCH_PROPOSED: proposals,
        STALE: stale,
        INVALID: max(0, invalid),
    }
    for status, expected in expected_status_counts.items():
        if status_counts[status] != expected:
            reasons.append(f"status_count:{status}={status_counts[status]}/{expected}")
    observed_unique = len(
        {
            (row.get("camera_a"), row.get("track_a"), row.get("camera_b"), row.get("track_b"))
            for row in rows
            if row.get("status") == MATCH_PROPOSED
        }
    )
    if observed_unique != unique:
        reasons.append(f"unique_proposal_rows={observed_unique}/counter={unique}")

    if reasons:
        print(
            "V11_STEP4_REID_SAME_ROOM_MATCHER_V1 RESULT=FAIL reasons="
            + ";".join(dict.fromkeys(reasons))
            + " global_id=0 room_id=0 face=0 handoff=0"
        )
        return 1
    print(
        "V11_STEP4_REID_SAME_ROOM_MATCHER_V1 RESULT=PASS "
        f"cycles={cycles} matrices_built={matrices} pairs_considered={considered} "
        f"pairs_valid={valid} nonreciprocal={counters['nonreciprocal']} "
        f"proposals={proposals} unique_proposals={unique} "
        f"proposal_changes={counters['proposal_changes']} "
        f"match_p50={timings['match_p50']:.3f}ms match_p95={timings['match_p95']:.3f}ms "
        "mechanics_only=1 identity_accuracy_proven=0 global_id=0 room_id=0 face=0 handoff=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
