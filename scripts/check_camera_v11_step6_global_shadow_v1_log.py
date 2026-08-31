#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v11.step6_step5_worker_tap_v1 import VERIFY_TSV_COLUMNS

PREFIX = "CAMERA_V11_STEP6_GLOBAL_VERIFY_V1 "
INTEGER = re.compile(r"\b([a-z][a-z0-9_]+)=(\d+)\b")
MILLISECOND = re.compile(r"\b(verify_p50|verify_p95)=([0-9.]+)ms")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--display-log", required=True, type=Path)
    parser.add_argument("--match-log", required=True, type=Path)
    parser.add_argument("--pair-tsv", required=True, type=Path)
    parser.add_argument("--match-tsv", required=True, type=Path)
    parser.add_argument("--global-tsv", required=True, type=Path)
    parser.add_argument("--verify-tsv", required=True, type=Path)
    parser.add_argument("--warmup-windows", type=int, default=2)
    parser.add_argument("--verify-p95-max-ms", type=float, default=1.0)
    args = parser.parse_args()

    reasons: list[str] = []
    prior = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_camera_v11_step5_global_shadow_v1_log.py"),
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
        reasons.append("step5_or_prior_regression")

    if not args.match_log.is_file():
        reasons.append("match_log_missing")
        text = ""
    else:
        text = args.match_log.read_text(encoding="utf-8", errors="replace")

    markers = (
        "CAMERA_V11_STEP6_GLOBAL_VERIFY_V1_ARCH",
        "verification=time_consistency+conflict_hysteresis",
        "conflict_action=hold_no_reassign",
        "geometry=disabled_requires_common_world_calibration",
        "raw_pixel_geometry=forbidden",
        "production_global_id=0 room_id=0 tracker_mutation=0",
        "face=0 handoff=0 identity_accuracy_proven=0",
        "matcher_blocking_verify_work=0",
        "CAMERA_V11_STEP6_GLOBAL_VERIFY_V1_CONFIG",
        "reid_threshold_added=0 geometry_threshold_added=0",
    )
    for marker in markers:
        if marker not in text:
            reasons.append(f"missing_marker:{marker.split()[0]}")
    if "Traceback (most recent call last)" in text:
        reasons.append("runtime_traceback")
    if "V11_STEP6_GLOBAL_VERIFY_WORKER_ERROR" in text:
        reasons.append("step6_worker_error")

    lines = [line for line in text.splitlines() if line.startswith(PREFIX)]
    final_line = lines[-1] if lines else ""
    if not final_line:
        reasons.append("step6_metrics_missing")
    counters = {name: int(value) for name, value in INTEGER.findall(final_line)}
    timings = {name: float(value) for name, value in MILLISECOND.findall(final_line)}
    required = (
        "records_created",
        "pending",
        "verified",
        "hold",
        "expired",
        "verified_total",
        "hold_events",
        "recovered_total",
        "persistent_conflicts",
        "verify_events",
        "events_written",
        "verify_worker_errors",
        "geometry_enabled",
        "production_global_id",
        "room_id",
        "face",
        "handoff",
    )
    for name in required:
        if name not in counters:
            reasons.append(f"missing_counter:{name}")
    for name in ("verify_p50", "verify_p95"):
        if name not in timings:
            reasons.append(f"missing_timing:{name}")

    if counters.get("records_created", 0) <= 0:
        reasons.append("no_live_step6_records")
    if counters.get("verify_worker_errors", -1) != 0:
        reasons.append(f"verify_worker_errors={counters.get('verify_worker_errors', -1)}")
    if counters.get("geometry_enabled", -1) != 0:
        reasons.append("geometry_must_remain_disabled_without_calibration")
    for name in ("production_global_id", "room_id", "face", "handoff"):
        if counters.get(name, -1) != 0:
            reasons.append(f"forbidden_mutation:{name}={counters.get(name, -1)}")
    if timings.get("verify_p95", 1e9) > args.verify_p95_max_ms:
        reasons.append(
            f"verify_p95={timings.get('verify_p95', -1):.3f}ms/max{args.verify_p95_max_ms:.3f}"
        )

    rows: list[dict[str, str]] = []
    if not args.verify_tsv.is_file():
        reasons.append("verify_tsv_missing")
    else:
        with args.verify_tsv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != VERIFY_TSV_COLUMNS:
                reasons.append("verify_tsv_columns_invalid")
            rows = list(reader)
    if any("embedding" in name.lower() for name in VERIFY_TSV_COLUMNS):
        reasons.append("raw_embedding_column_forbidden")
    if counters.get("events_written", -1) != len(rows):
        reasons.append(
            f"events_written={counters.get('events_written', -1)}/tsv_rows={len(rows)}"
        )
    if rows and not all(row.get("room") == "Devs" for row in rows):
        reasons.append("unexpected_room_in_step6_tsv")

    if reasons:
        print(
            "V11_STEP6_GLOBAL_VERIFY_V1 RESULT=FAIL reasons="
            + ";".join(dict.fromkeys(reasons))
            + " production_global_id=0 room_id=0 face=0 handoff=0 "
            "geometry_enabled=0 identity_accuracy_proven=0"
        )
        return 1

    print(
        "V11_STEP6_GLOBAL_VERIFY_V1 RESULT=PASS "
        f"records_created={counters['records_created']} pending={counters['pending']} "
        f"verified={counters['verified']} hold={counters['hold']} expired={counters['expired']} "
        f"verified_total={counters['verified_total']} hold_events={counters['hold_events']} "
        f"recovered_total={counters['recovered_total']} "
        f"persistent_conflicts={counters['persistent_conflicts']} "
        f"verify_p50={timings['verify_p50']:.3f}ms "
        f"verify_p95={timings['verify_p95']:.3f}ms "
        "production_global_id=0 room_id=0 face=0 handoff=0 "
        "geometry_enabled=0 identity_accuracy_proven=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
