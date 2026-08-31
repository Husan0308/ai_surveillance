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

from services.camera_v11.step5_global_shadow_worker_v1 import TSV_COLUMNS

PREFIX = "CAMERA_V11_STEP5_GLOBAL_SHADOW_V1 "
INTEGER = re.compile(r"\b([a-z][a-z0-9_]+)=(\d+)\b")
MILLISECOND = re.compile(r"\b(state_p50|state_p95)=([0-9.]+)ms")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--display-log", required=True, type=Path)
    parser.add_argument("--match-log", required=True, type=Path)
    parser.add_argument("--pair-tsv", required=True, type=Path)
    parser.add_argument("--match-tsv", required=True, type=Path)
    parser.add_argument("--global-tsv", required=True, type=Path)
    parser.add_argument("--warmup-windows", type=int, default=2)
    parser.add_argument("--state-p95-max-ms", type=float, default=1.0)
    args = parser.parse_args()

    reasons: list[str] = []
    prior = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_camera_v11_step4_reid_same_room_matcher_v1_log.py"),
            "--display-log",
            str(args.display_log),
            "--match-log",
            str(args.match_log),
            "--pair-tsv",
            str(args.pair_tsv),
            "--match-tsv",
            str(args.match_tsv),
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
        reasons.append("step4_matcher_or_prior_regression")

    if not args.match_log.is_file():
        reasons.append("match_log_missing")
        text = ""
    else:
        text = args.match_log.read_text(encoding="utf-8", errors="replace")
    for marker in (
        "CAMERA_V11_STEP5_GLOBAL_SHADOW_V1_ARCH",
        "mode=shadow input=step4_MATCH_PROPOSED",
        "confirm_observations=3 confirm_consecutive=3",
        "different_pair_reuse=0",
        "conflict_resolution=0 hysteresis=0",
        "production_global_id=0 room_id=0",
        "face=0 handoff=0 identity_accuracy_proven=0",
        "queue=bounded async=1 matcher_blocking_state_work=0",
        "CAMERA_V11_STEP5_GLOBAL_SHADOW_V1_CONFIG",
    ):
        if marker not in text:
            reasons.append(f"missing_marker:{marker.split()[0]}")
    if "Traceback (most recent call last)" in text:
        reasons.append("runtime_traceback")
    if "V11_STEP5_GLOBAL_SHADOW_WORKER_ERROR" in text:
        reasons.append("global_shadow_worker_error")

    lines = [line for line in text.splitlines() if line.startswith(PREFIX)]
    final_line = lines[-1] if lines else ""
    if not final_line:
        reasons.append("global_shadow_metrics_missing")
    counters = {name: int(value) for name, value in INTEGER.findall(final_line)}
    timings = {name: float(value) for name, value in MILLISECOND.findall(final_line)}
    required = (
        "created",
        "provisional",
        "confirmed",
        "observations",
        "conflicts",
        "expired",
        "active",
        "member_tracks",
        "queue_pending",
        "queue_dropped",
        "events_written",
        "worker_errors",
    )
    for name in required:
        if name not in counters:
            reasons.append(f"missing_counter:{name}")
    for name in ("state_p50", "state_p95"):
        if name not in timings:
            reasons.append(f"missing_timing:{name}")

    created = counters.get("created", 0)
    observations = counters.get("observations", 0)
    active = counters.get("active", 0)
    provisional = counters.get("provisional", 0)
    confirmed = counters.get("confirmed", 0)
    if created <= 0 or observations <= 0:
        reasons.append("no_live_global_shadow_evidence")
    if active != provisional + confirmed:
        reasons.append(f"active_accounting={active}!={provisional}+{confirmed}")
    if counters.get("member_tracks", -1) != active * 2:
        reasons.append("member_track_accounting_invalid")
    if counters.get("queue_pending", -1) != 0:
        reasons.append(f"queue_pending={counters.get('queue_pending', -1)}")
    if counters.get("queue_dropped", -1) != 0:
        reasons.append(f"queue_dropped={counters.get('queue_dropped', -1)}")
    if counters.get("worker_errors", -1) != 0:
        reasons.append(f"worker_errors={counters.get('worker_errors', -1)}")
    if timings.get("state_p95", 1e9) > args.state_p95_max_ms:
        reasons.append(
            f"state_p95={timings.get('state_p95', -1):.3f}ms/max{args.state_p95_max_ms:.3f}"
        )

    rows: list[dict[str, str]] = []
    if not args.global_tsv.is_file():
        reasons.append("global_tsv_missing")
    else:
        with args.global_tsv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != TSV_COLUMNS:
                reasons.append("global_tsv_columns_invalid")
            rows = list(reader)
    if any("embedding" in name.lower() for name in TSV_COLUMNS):
        reasons.append("raw_embedding_column_forbidden")
    if counters.get("events_written", -1) != len(rows):
        reasons.append(
            f"events_written={counters.get('events_written', -1)}/tsv_rows={len(rows)}"
        )
    if rows and not all(row.get("room") == "Devs" for row in rows):
        reasons.append("unexpected_room_in_step5_tsv")
    if any(row.get("event") == "GLOBAL_SHADOW_CONFIRM" for row in rows):
        if not any(row.get("state") == "CONFIRMED_SHADOW" for row in rows):
            reasons.append("confirm_event_without_confirmed_state")

    if reasons:
        print(
            "V11_STEP5_GLOBAL_SHADOW_V1 RESULT=FAIL reasons="
            + ";".join(dict.fromkeys(reasons))
            + " production_global_id=0 room_id=0 face=0 handoff=0 identity_accuracy_proven=0"
        )
        return 1
    print(
        "V11_STEP5_GLOBAL_SHADOW_V1 RESULT=PASS "
        f"created={created} provisional={provisional} confirmed={confirmed} "
        f"observations={observations} conflicts={counters['conflicts']} "
        f"expired={counters['expired']} active={active} member_tracks={counters['member_tracks']} "
        f"state_p50={timings['state_p50']:.3f}ms state_p95={timings['state_p95']:.3f}ms "
        "production_global_id=0 room_id=0 face=0 handoff=0 identity_accuracy_proven=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
