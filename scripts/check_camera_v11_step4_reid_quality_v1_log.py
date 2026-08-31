#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


QUALITY_PREFIX = "CAMERA_V11_STEP4_REID_QUALITY_V1 "
INTEGER = re.compile(r"\b([a-z_]+)=(\d+)\b")
MILLISECOND = re.compile(r"\b(gate_p50|gate_p95)=([0-9.]+)ms")
TRACKER_P95 = re.compile(r"\btracker_p95=([0-9.]+)ms")
LATEST = re.compile(
    r"(CAM-\d+):demand=(\d+),accepted=(\d+),processed=(\d+),"
    r"overwritten=(\d+),coalesced=(\d+),pending=(\d+)"
)
CREDIT = re.compile(r"(CAM-\d+):credit=(\d+),overflow=(\d+)")
NATIVE = re.compile(r"CAM-\d+:\d+x\d+")
TERMINAL = (
    "accepted",
    "reject_predicted",
    "reject_score",
    "reject_size",
    "reject_edge",
    "reject_aspect",
    "reject_blur",
    "reject_invalid",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--display-log", required=True, type=Path)
    parser.add_argument("--quality-log", required=True, type=Path)
    parser.add_argument("--warmup-windows", type=int, default=2)
    parser.add_argument("--gate-p95-max-ms", type=float, default=3.0)
    parser.add_argument("--tracker-p95-max-ms", type=float, default=1.5)
    args = parser.parse_args()
    if not args.display_log.is_file() or not args.quality_log.is_file():
        print("V11_STEP4_REID_QUALITY_V1 RESULT=FAIL reasons=missing_log")
        return 2

    reasons: list[str] = []
    root = Path(__file__).resolve().parents[1]
    guard = subprocess.run(
        [sys.executable, str(root / "scripts/check_camera_v11_frozen_step123_guard.py")],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(guard.stdout, end="")
    if guard.returncode != 0:
        reasons.append("frozen_files_changed")

    step3 = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/check_camera_v11_step3_tracker_v2_log.py"),
            "--display-log",
            str(args.display_log),
            "--tracker-log",
            str(args.quality_log),
            "--warmup-windows",
            str(args.warmup_windows),
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(step3.stdout, end="")
    if step3.returncode != 0:
        reasons.append("frozen_step3_regression")

    text = args.quality_log.read_text(encoding="utf-8", errors="replace")
    required_markers = (
        "CAMERA_V11_STEP4_REID_QUALITY_V1_ARCH",
        "source=native-decoded-before-detector-resize",
        "camera_thread=metadata-only gate_worker=async-cpu",
        "camera_queue=0 python_frame_queue=0",
        "display_topology_changed=0 detector_schedule_changed=0 tracker_changed=0",
        "reid_inference=0 gallery=0 pair_scoring=0 room_id=0 global_id=0 face=0 handoff=0",
        "CAMERA_V11_STEP4_REID_QUALITY_V1_POLICY",
    )
    for marker in required_markers:
        if marker not in text:
            reasons.append(f"missing:{marker.split()[0]}")
    if "Traceback (most recent call last)" in text:
        reasons.append("runtime_traceback")

    quality_lines = [line for line in text.splitlines() if line.startswith(QUALITY_PREFIX)]
    if not quality_lines:
        reasons.append("quality_metrics_missing")
        quality_line = ""
    else:
        quality_line = quality_lines[-1]
    integers = {name: int(value) for name, value in INTEGER.findall(quality_line)}
    milliseconds = {name: float(value) for name, value in MILLISECOND.findall(quality_line)}
    for name in ("submitted", *TERMINAL, "latest_replaced"):
        if name not in integers:
            reasons.append(f"missing_counter:{name}")
    submitted = integers.get("submitted", 0)
    terminal = sum(integers.get(name, 0) for name in TERMINAL)
    if submitted <= 0:
        reasons.append("no_quality_candidates")
    if integers.get("accepted", 0) <= 0:
        reasons.append("no_accepted_native_crops")
    if terminal != submitted:
        reasons.append(f"unterminated_candidates={submitted - terminal}")
    if integers.get("latest_replaced", 1) != 0:
        reasons.append(f"latest_replaced={integers.get('latest_replaced', -1)}")
    if milliseconds.get("gate_p50", 0.0) <= 0.0:
        reasons.append("no_gate_timing")
    if milliseconds.get("gate_p95", 1e9) > args.gate_p95_max_ms:
        reasons.append(
            f"gate_p95={milliseconds.get('gate_p95', -1):.3f}ms/max{args.gate_p95_max_ms:.3f}"
        )
    native = NATIVE.findall(quality_line.partition(" native=")[2])
    if len(set(native)) != 6:
        reasons.append(f"native_source_shapes={len(set(native))}/6")

    tracker_lines = [
        line for line in text.splitlines() if line.startswith("CAMERA_V11_STEP3_V2_TRACKER ")
    ]
    tracker_match = TRACKER_P95.search(tracker_lines[-1]) if tracker_lines else None
    tracker_p95 = float(tracker_match.group(1)) if tracker_match else 1e9
    if tracker_p95 > args.tracker_p95_max_ms:
        reasons.append(f"tracker_p95={tracker_p95:.3f}ms/max{args.tracker_p95_max_ms:.3f}")

    latest_lines = [
        line for line in text.splitlines() if line.startswith("CAMERA_V11_STEP2_V12_LATEST ")
    ]
    latest = LATEST.findall(latest_lines[-1]) if latest_lines else []
    if len(latest) != 6:
        reasons.append(f"step2_latest_rows={len(latest)}")
    elif max(int(row[6]) for row in latest) > 1:
        reasons.append("step2_pending_max_gt_1")
    credit_lines = [
        line for line in text.splitlines() if line.startswith("CAMERA_V11_STEP2_V13_CREDIT_STATS ")
    ]
    credits = CREDIT.findall(credit_lines[-1]) if credit_lines else []
    if len(credits) != 6:
        reasons.append(f"step2_credit_rows={len(credits)}")
    elif any(int(overflow) != 0 for _cid, _credit, overflow in credits):
        reasons.append("step2_overflow_nonzero")
    backlog_lines = [
        line for line in text.splitlines() if line.startswith("CAMERA_V11_STEP2_BACKLOG ")
    ]
    if not backlog_lines or "old_frame_retry=0" not in backlog_lines[-1]:
        reasons.append("step2_old_frame_retry_not_zero")

    if reasons:
        print("V11_STEP4_REID_QUALITY_V1 RESULT=FAIL reasons=" + ";".join(reasons))
        return 1
    print(
        "V11_STEP4_REID_QUALITY_V1 RESULT=PASS "
        f"submitted={submitted} accepted={integers['accepted']} "
        f"gate_p50={milliseconds['gate_p50']:.3f}ms gate_p95={milliseconds['gate_p95']:.3f}ms "
        f"tracker_p95={tracker_p95:.3f}ms pending_max={max(int(row[6]) for row in latest)} "
        "overflow=0 old_frame_retry=0 duplicate_errors=0 prefix_errors=0 display_smooth=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
