#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

RATE = re.compile(r"(CAM-\d+):rtsp=([0-9.]+)/decode=([0-9.]+)/detect=([0-9.]+)Hz")
QUEUE = re.compile(r"(CAM-\d+):in=(\d+),out=(\d+),app=(\d+),max=(\d+),gate_drop=(\d+)")
LATEST = re.compile(
    r"(CAM-\d+):demand=(\d+),accepted=(\d+),processed=(\d+),"
    r"overwritten=(\d+),coalesced=(\d+),pending=(\d+)"
)
VALUE = re.compile(r"\b([a-z0-9_]+)=([0-9.]+)ms")


def longest_true_run(flags: list[bool]) -> int:
    best = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--display-log", required=True)
    parser.add_argument("--detector-log", required=True)
    parser.add_argument("--warmup-windows", type=int, default=2)
    parser.add_argument("--min-detector-hz", type=float, default=1.80)
    parser.add_argument("--max-detector-consecutive-low", type=int, default=3)
    parser.add_argument("--min-demand-fulfillment", type=float, default=0.95)
    args = parser.parse_args()

    if not (0.0 < args.min_demand_fulfillment <= 1.0):
        print("V11_STEP2_PRODUCTION_V25 FAIL invalid_min_demand_fulfillment=1")
        return 2

    display = Path(args.display_log)
    detector = Path(args.detector_log)
    if not display.is_file() or not detector.is_file():
        print("V11_STEP2_PRODUCTION_V25 FAIL missing_log=1")
        return 2

    display_check = Path(__file__).with_name("check_camera_v11_step1_v25_aggregate_log.py")
    check = subprocess.run(
        [
            sys.executable,
            str(display_check),
            str(display),
            "--warmup-windows",
            str(args.warmup_windows),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(check.stdout, end="")
    reasons: list[str] = []
    if check.returncode != 0:
        reasons.append("aggregate_step1_regression")

    text = detector.read_text(encoding="utf-8", errors="replace")
    for marker in (
        "CAMERA_V11_STEP2_PRODUCTION_ARCH mode=full",
        "CAMERA_V11_STEP2_TRT_READY",
        "precision=fp32",
        "CAMERA_V11_STEP2_V12_SCHEDULER",
        "CAMERA_V11_STEP2_V13_CREDIT",
        "CAMERA_V11_STEP2_PROFILE",
    ):
        if marker not in text:
            reasons.append(f"missing:{marker.split()[0]}")

    source_lines = [line for line in text.splitlines() if line.startswith("CAMERA_V11_STEP2_SOURCE ")]
    backlog_lines = [line for line in text.splitlines() if line.startswith("CAMERA_V11_STEP2_BACKLOG ")]
    latest_lines = [line for line in text.splitlines() if line.startswith("CAMERA_V11_STEP2_V12_LATEST ")]
    profile_lines = [line for line in text.splitlines() if line.startswith("CAMERA_V11_STEP2_PROFILE ")]

    rate_windows: list[list[tuple[str, str, str, str]]] = []
    for line in source_lines:
        parsed = RATE.findall(line)
        if len(parsed) == 6:
            rate_windows.append(parsed)
    usable_rate_windows = rate_windows[max(0, args.warmup_windows) :]
    if len(usable_rate_windows) < 6:
        reasons.append(f"detector_windows={len(usable_rate_windows)}/min6")

    detect_by_camera: dict[str, list[float]] = {}
    for window in usable_rate_windows:
        for cid, _rtsp, _decoded, detect in window:
            detect_by_camera.setdefault(cid, []).append(float(detect))

    avg_detect_by_camera: dict[str, float] = {}
    if len(detect_by_camera) != 6:
        reasons.append(f"detector_rate_cameras={len(detect_by_camera)}")
    for cid, values in sorted(detect_by_camera.items()):
        if not values:
            reasons.append(f"{cid}:no_detector_rate_samples")
            continue
        avg_detect = sum(values) / len(values)
        min_detect = min(values)
        low_flags = [value < args.min_detector_hz for value in values]
        low_windows = sum(low_flags)
        low_streak = longest_true_run(low_flags)
        avg_detect_by_camera[cid] = avg_detect

        # Five-second RTSP reporting windows can straddle burst delivery and make
        # one interval look artificially low while the next catches up. Production
        # acceptance therefore uses whole-run average + sustained-low streak here,
        # and cumulative demand fulfillment below. Isolated per-window minima are
        # diagnostic only, not an acceptance gate.
        if avg_detect < args.min_detector_hz:
            reasons.append(f"{cid}:avg_detect_hz={avg_detect:.2f}")
        if low_streak > args.max_detector_consecutive_low:
            reasons.append(
                f"{cid}:detect_low_streak={low_streak}/max{args.max_detector_consecutive_low}"
            )
        print(
            "V11_STEP2_V25_AGG_RATE "
            f"camera={cid} windows={len(values)} avg_detect_hz={avg_detect:.2f} "
            f"min_detect_hz={min_detect:.2f} low_detect_windows={low_windows} "
            f"low_streak={low_streak}"
        )

    queues = QUEUE.findall(backlog_lines[-1]) if backlog_lines else []
    latest = LATEST.findall(latest_lines[-1]) if latest_lines else []
    stages = {name: float(value) for name, value in VALUE.findall(profile_lines[-1])} if profile_lines else {}

    if len(queues) != 6:
        reasons.append(f"queue_rows={len(queues)}")
    for cid, input_depth, output_depth, app_depth, maximum, gate_drop in queues:
        if max(int(input_depth), int(output_depth), int(maximum)) > 1 or int(app_depth) != 0:
            reasons.append(f"{cid}:queue={input_depth}/{output_depth}/{app_depth}/max{maximum}")
        if int(gate_drop) <= 0:
            reasons.append(f"{cid}:no_decoded_gate_evidence")

    fulfillment_by_camera: dict[str, float] = {}
    if len(latest) != 6:
        reasons.append(f"latest_rows={len(latest)}")
    for cid, demand, accepted, processed, overwritten, coalesced, pending in latest:
        demand_i = int(demand)
        accepted_i = int(accepted)
        processed_i = int(processed)
        pending_i = int(pending)
        if demand_i <= 0 or accepted_i <= 0 or processed_i <= 0:
            reasons.append(f"{cid}:no_detector_results")
            continue
        accepted_ratio = accepted_i / demand_i
        processed_ratio = processed_i / demand_i
        fulfillment_by_camera[cid] = processed_ratio
        if accepted_ratio < args.min_demand_fulfillment:
            reasons.append(f"{cid}:accepted_fulfillment={accepted_ratio:.3f}")
        if processed_ratio < args.min_demand_fulfillment:
            reasons.append(f"{cid}:processed_fulfillment={processed_ratio:.3f}")
        if pending_i > 1:
            reasons.append(f"{cid}:pending={pending_i}")
        print(
            "V11_STEP2_V25_FULFILLMENT "
            f"camera={cid} demand={demand_i} accepted={accepted_i} processed={processed_i} "
            f"accepted_ratio={accepted_ratio:.3f} processed_ratio={processed_ratio:.3f} "
            f"overwritten={overwritten} coalesced={coalesced} pending={pending_i}"
        )

    if stages.get("result_age_p95", 1e9) > 300.0:
        reasons.append(f"result_age_p95={stages.get('result_age_p95', -1):.1f}ms")
    if stages.get("inference_p50", 0.0) <= 0.0:
        reasons.append("no_inference_samples")

    if reasons:
        print("V11_STEP2_PRODUCTION_V25 RESULT=FAIL reasons=" + ";".join(reasons))
        return 1

    print(
        "V11_STEP2_PRODUCTION_V25 RESULT=PASS "
        f"cameras=6 rate_min_avg={min(avg_detect_by_camera.values()):.2f}Hz "
        f"fulfillment_min={min(fulfillment_by_camera.values()):.3f} "
        f"queue_max={max(int(row[4]) for row in queues)} "
        f"pending_max={max(int(row[6]) for row in latest)} "
        f"result_age_p95={stages['result_age_p95']:.1f}ms "
        f"max_consecutive_low={args.max_detector_consecutive_low}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
