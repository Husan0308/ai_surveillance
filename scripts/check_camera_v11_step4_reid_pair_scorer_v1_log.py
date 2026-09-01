#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v11.step4_reid_pair_shadow_v1 import TSV_COLUMNS


PREFIX = "CAMERA_V11_STEP4_REID_PAIR_SCORER_V1 "
INTEGER = re.compile(r"\b([a-z][a-z0-9_]+)=(\d+)\b")
MILLISECOND = re.compile(r"\b(score_p50|score_p95)=([0-9.]+)ms")
COUNTERS = (
    "pairs_considered",
    "pairs_scored",
    "pairs_insufficient",
    "pairs_invalid",
    "same_room_pairs",
    "different_room_pairs",
    "worker_errors",
)


def _env_truthy(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "disabled",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--display-log", required=True, type=Path)
    parser.add_argument("--pair-log", required=True, type=Path)
    parser.add_argument("--tsv", required=True, type=Path)
    parser.add_argument("--warmup-windows", type=int, default=2)
    parser.add_argument("--score-p95-max-ms", type=float, default=2.0)
    args = parser.parse_args()
    if not args.display_log.is_file() or not args.pair_log.is_file():
        print("V11_STEP4_REID_PAIR_SCORER_V1 RESULT=FAIL reasons=missing_log")
        return 2

    # General Step3 acceptance remains strict and expects both same-room and
    # different-room diagnostics. The dedicated Step7 CAM-01/CAM-04 one-person
    # run intentionally contains no unrelated people in the other rooms, so its
    # wrapper can disable only this unrelated context requirement via environment.
    require_different_room = _env_truthy("V11_STEP4_PAIR_REQUIRE_DIFFERENT_ROOM", "1")

    root = ROOT
    reasons: list[str] = []
    step2 = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/check_camera_v11_step4_reid_gallery_v1_log.py"),
            "--display-log",
            str(args.display_log),
            "--gallery-log",
            str(args.pair_log),
            "--warmup-windows",
            str(args.warmup_windows),
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(step2.stdout, end="")
    if step2.returncode != 0:
        reasons.append("step2_gallery_or_prior_regression")

    text = args.pair_log.read_text(encoding="utf-8", errors="replace")
    markers = (
        "CAMERA_V11_STEP4_REID_PAIR_SCORER_V1_ARCH",
        "input=independent-local-track-galleries samples=3..8 matrix=max8x8",
        "compute=cpu-numpy gpu_inference=0 worker=one async=1 dirty_slot=latest-only",
        "camera_display_block=0 tracker_mutation=0 identity_decision=0 threshold=0",
        "candidate_scope=active-recent-cross-camera max_candidates=24",
        "priority=CAM-01+CAM-04",
        "CAMERA_V11_STEP4_REID_PAIR_SCORER_V1_FORMULA",
        "robust_score=0.40*top3_mean+0.25*median_of_best_matches+0.20*p75_score+0.15*max_score",
        "diagnostics_only=1",
        "raw_embeddings_tsv=0",
        "reciprocal=0 one_to_one=0 room_id_assignment=0 global_id=0",
        "provisional_confirmed=0 face=0 cross_room_handoff=0",
    )
    for marker in markers:
        if marker not in text:
            reasons.append(f"missing_marker:{marker.split()[0]}")
    if "Traceback (most recent call last)" in text:
        reasons.append("runtime_traceback")
    if "V11_STEP4_REID_PAIR_SHADOW_ERROR" in text:
        reasons.append("shadow_worker_error")

    lines = [line for line in text.splitlines() if line.startswith(PREFIX)]
    final_line = lines[-1] if lines else ""
    if not final_line:
        reasons.append("pair_metrics_missing")
    counters = {name: int(value) for name, value in INTEGER.findall(final_line)}
    timings = {name: float(value) for name, value in MILLISECOND.findall(final_line)}
    for counter in COUNTERS:
        if counter not in counters:
            reasons.append(f"missing_counter:{counter}")
    for timing in ("score_p50", "score_p95"):
        if timing not in timings:
            reasons.append(f"missing_timing:{timing}")

    considered = counters.get("pairs_considered", 0)
    scored = counters.get("pairs_scored", 0)
    insufficient = counters.get("pairs_insufficient", 0)
    invalid = counters.get("pairs_invalid", -1)
    same_room = counters.get("same_room_pairs", 0)
    different_room = counters.get("different_room_pairs", 0)
    worker_errors = counters.get("worker_errors", -1)
    if considered <= 0 or scored <= 0:
        reasons.append("no_live_pair_scores")
    if considered != scored + insufficient + max(0, invalid):
        reasons.append(
            f"pair_accounting={considered}!={scored}+{insufficient}+{invalid}"
        )
    if same_room + different_room != considered:
        reasons.append(
            f"context_accounting={same_room}+{different_room}!={considered}"
        )
    if same_room <= 0:
        reasons.append("no_same_room_pairs")
    if require_different_room and different_room <= 0:
        reasons.append("no_different_room_pairs")
    if invalid != 0:
        reasons.append(f"pairs_invalid={invalid}")
    if worker_errors != 0:
        reasons.append(f"worker_errors={worker_errors}")
    if timings.get("score_p50", 0.0) <= 0.0:
        reasons.append("no_score_timing")
    if timings.get("score_p95", 1e9) > args.score_p95_max_ms:
        reasons.append(
            f"score_p95={timings.get('score_p95', -1):.3f}ms/"
            f"max{args.score_p95_max_ms:.3f}"
        )

    tsv_rows: list[dict[str, str]] = []
    if not args.tsv.is_file():
        reasons.append("pair_tsv_missing")
    else:
        with args.tsv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != TSV_COLUMNS:
                reasons.append("pair_tsv_columns_invalid")
            tsv_rows = list(reader)
        if len(tsv_rows) != scored:
            reasons.append(f"pair_tsv_rows={len(tsv_rows)}/scored={scored}")
        if any(row.get("camera_a") == row.get("camera_b") for row in tsv_rows):
            reasons.append("same_camera_pair_in_tsv")
        if any(row.get("context") not in ("same_room", "different_room") for row in tsv_rows):
            reasons.append("invalid_context_in_tsv")
        if not any(
            {row.get("camera_a"), row.get("camera_b")} == {"CAM-01", "CAM-04"}
            for row in tsv_rows
        ):
            reasons.append("no_cam01_cam04_diagnostic")
        if any("embedding" in column.lower() for column in (reader.fieldnames or ())):
            reasons.append("raw_embedding_column_forbidden")

    if reasons:
        print(
            "V11_STEP4_REID_PAIR_SCORER_V1 RESULT=FAIL reasons="
            + ";".join(reasons)
            + f" different_room_required={int(require_different_room)}"
        )
        return 1
    print(
        "V11_STEP4_REID_PAIR_SCORER_V1 RESULT=PASS "
        f"pairs_considered={considered} pairs_scored={scored} "
        f"pairs_insufficient={insufficient} pairs_invalid=0 "
        f"same_room_pairs={same_room} different_room_pairs={different_room} "
        f"different_room_required={int(require_different_room)} "
        f"score_p50={timings['score_p50']:.3f}ms "
        f"score_p95={timings['score_p95']:.3f}ms "
        f"tsv_rows={len(tsv_rows)} cam01_cam04=1 "
        "threshold=0 assignment=0 reciprocal=0 one_to_one=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
