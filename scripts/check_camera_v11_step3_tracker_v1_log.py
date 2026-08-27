#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROW = re.compile(
    r"(CAM-\d+):updates=(\d+),created=(\d+),recovered=(\d+),removed=(\d+),"
    r"visible=(\d+),ids=([^ |]+)"
)
METRIC = re.compile(r"\b(tracker_p50|tracker_p95)=([0-9.]+)ms")
ERRORS = re.compile(r"\b(duplicate_errors|prefix_errors)=(\d+)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--display-log", required=True)
    parser.add_argument("--tracker-log", required=True)
    parser.add_argument("--tracker-p95-max-ms", type=float, default=8.0)
    args = parser.parse_args()

    display = Path(args.display_log)
    tracker = Path(args.tracker_log)
    if not display.is_file() or not tracker.is_file():
        print("V11_STEP3_TRACKER_V1 RESULT=FAIL reasons=missing_log")
        return 2

    step2_check = Path(__file__).with_name("check_camera_v11_step2_production_log_v15.py")
    base = subprocess.run(
        [
            sys.executable,
            str(step2_check),
            "--display-log",
            str(display),
            "--detector-log",
            str(tracker),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(base.stdout, end="")
    reasons: list[str] = []
    if base.returncode != 0:
        reasons.append("step2_regression")

    text = tracker.read_text(encoding="utf-8", errors="replace")
    for marker in (
        "CAMERA_V11_STEP3_ARCH",
        "tracker=cpu-time-aware-bytetrack-style",
        "appearance=0 reid=0 face=0",
        "CAMERA_V11_STEP3_TRACKER ",
    ):
        if marker not in text:
            reasons.append(f"missing:{marker.split()[0]}")

    lines = [line for line in text.splitlines() if line.startswith("CAMERA_V11_STEP3_TRACKER ")]
    rows = ROW.findall(lines[-1]) if lines else []
    metrics = {name: float(value) for name, value in METRIC.findall(lines[-1])} if lines else {}
    errors = {name: int(value) for name, value in ERRORS.findall(lines[-1])} if lines else {}

    if len(rows) != 6:
        reasons.append(f"tracker_rows={len(rows)}")
    for cid, updates, _created, _recovered, _removed, _visible, ids_text in rows:
        if int(updates) <= 0:
            reasons.append(f"{cid}:no_tracker_updates")
        if ids_text != "-":
            for track_id in ids_text.split(","):
                if not track_id.startswith(f"{cid}-T"):
                    reasons.append(f"{cid}:bad_id={track_id}")

    p95 = metrics.get("tracker_p95", 1e9)
    if p95 > args.tracker_p95_max_ms:
        reasons.append(f"tracker_p95={p95:.3f}ms")
    if metrics.get("tracker_p50", 0.0) <= 0.0:
        reasons.append("no_tracker_timing")
    if errors.get("duplicate_errors", 1) != 0:
        reasons.append(f"duplicate_errors={errors.get('duplicate_errors', -1)}")
    if errors.get("prefix_errors", 1) != 0:
        reasons.append(f"prefix_errors={errors.get('prefix_errors', -1)}")

    if reasons:
        print("V11_STEP3_TRACKER_V1 RESULT=FAIL reasons=" + ";".join(reasons))
        return 1

    print(
        "V11_STEP3_TRACKER_V1 RESULT=PASS "
        f"cameras=6 tracker_p50={metrics['tracker_p50']:.3f}ms "
        f"tracker_p95={metrics['tracker_p95']:.3f}ms "
        "duplicate_errors=0 prefix_errors=0 step2_regression=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
