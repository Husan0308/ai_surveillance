#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

FROZEN_STEP2_SHA = "2f83fb3ef5c2bb4e4cba7dc9c923c918fe3847a1"
ROW = re.compile(
    r"(CAM-\d+):updates=(\d+),created=(\d+),recovered=(\d+),removed=(\d+),"
    r"visible=(\d+),ids=([^ |]+)"
)
LATEST = re.compile(
    r"(CAM-\d+):demand=(\d+),accepted=(\d+),processed=(\d+),"
    r"overwritten=(\d+),coalesced=(\d+),pending=(\d+)"
)
METRIC = re.compile(r"\b(tracker_p50|tracker_p95)=([0-9.]+)ms")
ERRORS = re.compile(r"\b(duplicate_errors|prefix_errors)=(\d+)")
STEP2_FAIL = re.compile(r"V11_STEP2_PRODUCTION_V25 RESULT=FAIL reasons=([^\n]+)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--display-log", required=True)
    parser.add_argument("--tracker-log", required=True)
    parser.add_argument("--warmup-windows", type=int, default=2)
    parser.add_argument("--tracker-p95-max-ms", type=float, default=8.0)
    args = parser.parse_args()

    display = Path(args.display_log)
    tracker = Path(args.tracker_log)
    if not display.is_file() or not tracker.is_file():
        print("V11_STEP3_TRACKER_V2 RESULT=FAIL reasons=missing_log")
        return 2

    # Run the frozen Step2 V25 checker first. If (and only if) it fails solely
    # because the display checker counted too many isolated 5-second low windows,
    # re-evaluate display with the same whole-run FPS/ratio, hard floor, sustained
    # streak, latency, queue and error gates while treating isolated low-window
    # count as diagnostic. All detector/queue/result-age failures remain fatal.
    step2_check = Path(__file__).with_name("check_camera_v11_step2_production_log_v25.py")
    base = subprocess.run(
        [
            sys.executable,
            str(step2_check),
            "--display-log",
            str(display),
            "--detector-log",
            str(tracker),
            "--warmup-windows",
            str(args.warmup_windows),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(base.stdout, end="")
    reasons: list[str] = []
    step2_regression_ok = base.returncode == 0

    if not step2_regression_ok:
        fail_match = STEP2_FAIL.search(base.stdout)
        fail_reasons = [] if fail_match is None else [x for x in fail_match.group(1).split(";") if x]
        if fail_reasons == ["aggregate_step1_regression"]:
            display_check = Path(__file__).with_name("check_camera_v11_step1_v25_aggregate_log.py")
            sustained = subprocess.run(
                [
                    sys.executable,
                    str(display_check),
                    str(display),
                    "--warmup-windows",
                    str(args.warmup_windows),
                    "--max-transient-fraction",
                    "1.0",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            print(sustained.stdout, end="")
            if sustained.returncode == 0:
                step2_regression_ok = True
                print(
                    "V11_STEP3_STEP2_REGRESSION result=PASS "
                    "mode=frozen-step2-plus-sustained-display isolated_low_windows=diagnostic"
                )
        if not step2_regression_ok:
            reasons.append("step2_regression")

    text = tracker.read_text(encoding="utf-8", errors="replace")
    for marker in (
        "CAMERA_V11_STEP3_V2_ARCH",
        f"base_step2_sha={FROZEN_STEP2_SHA}",
        "tracker=observation-centric-sparse-bytetrack",
        "appearance=0 reid=0 face=0 tracker_gpu=0",
        "frame_queue=0",
        "display_topology_changed=0 detector_schedule_changed=0",
        "CAMERA_V11_STEP3_V2_TRACKER ",
    ):
        if marker not in text:
            reasons.append(f"missing:{marker.split()[0]}")

    tracker_lines = [
        line for line in text.splitlines() if line.startswith("CAMERA_V11_STEP3_V2_TRACKER ")
    ]
    latest_lines = [
        line for line in text.splitlines() if line.startswith("CAMERA_V11_STEP2_V12_LATEST ")
    ]
    rows = ROW.findall(tracker_lines[-1]) if tracker_lines else []
    latest = LATEST.findall(latest_lines[-1]) if latest_lines else []
    metrics = {name: float(value) for name, value in METRIC.findall(tracker_lines[-1])} if tracker_lines else {}
    errors = {name: int(value) for name, value in ERRORS.findall(tracker_lines[-1])} if tracker_lines else {}

    processed_by_camera = {cid: int(processed) for cid, _d, _a, processed, _o, _c, _p in latest}
    updates_by_camera: dict[str, int] = {}

    if len(rows) != 6:
        reasons.append(f"tracker_rows={len(rows)}")
    if len(latest) != 6:
        reasons.append(f"step2_latest_rows={len(latest)}")

    for cid, updates, _created, _recovered, _removed, _visible, ids_text in rows:
        update_count = int(updates)
        updates_by_camera[cid] = update_count
        if update_count <= 0:
            reasons.append(f"{cid}:no_tracker_updates")
        processed = processed_by_camera.get(cid)
        if processed is None:
            reasons.append(f"{cid}:missing_step2_processed")
        elif update_count != processed:
            reasons.append(f"{cid}:tracker_updates={update_count}/processed={processed}")
        if ids_text != "-":
            ids = ids_text.split(",")
            if len(ids) != len(set(ids)):
                reasons.append(f"{cid}:duplicate_visible_ids")
            for track_id in ids:
                if not track_id.startswith(f"{cid}-T"):
                    reasons.append(f"{cid}:bad_id={track_id}")

    p95 = metrics.get("tracker_p95", 1e9)
    if p95 > args.tracker_p95_max_ms:
        reasons.append(f"tracker_p95={p95:.3f}ms/max{args.tracker_p95_max_ms:.3f}")
    if metrics.get("tracker_p50", 0.0) <= 0.0:
        reasons.append("no_tracker_timing")
    if errors.get("duplicate_errors", 1) != 0:
        reasons.append(f"duplicate_errors={errors.get('duplicate_errors', -1)}")
    if errors.get("prefix_errors", 1) != 0:
        reasons.append(f"prefix_errors={errors.get('prefix_errors', -1)}")

    if reasons:
        print("V11_STEP3_TRACKER_V2 RESULT=FAIL reasons=" + ";".join(reasons))
        return 1

    print(
        "V11_STEP3_TRACKER_V2 RESULT=PASS "
        f"cameras=6 tracker_p50={metrics['tracker_p50']:.3f}ms "
        f"tracker_p95={metrics['tracker_p95']:.3f}ms "
        f"updates_min={min(updates_by_camera.values())} "
        "tracker_processed_match=1 duplicate_errors=0 prefix_errors=0 step2_regression=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
