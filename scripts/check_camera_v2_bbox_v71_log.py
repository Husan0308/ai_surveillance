#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

SOURCE_RE = re.compile(r"(?P<cid>CAM-\d+):(?P<fps>[0-9.]+)fps")
TRACKER_RE = re.compile(r"tracker_rate=(?P<hz>[0-9.]+)Hz")
ACTUAL_RE = re.compile(r"actual=\[(?P<body>[^\]]*)\]")
ACTUAL_ITEM_RE = re.compile(r"(?P<cid>CAM-\d+):(?P<hz>[0-9.]+)")
LANE_RE = re.compile(
    r"det_hold_avg=(?P<avg>[0-9.]+)ms.*det_hold_p95=(?P<p95>[0-9.]+)ms.*tracker_skips=(?P<skips>\d+)"
)
V71_RE = re.compile(
    r"empty_holds=(?P<holds>\d+).*empty_expires=(?P<expires>\d+).*capture_before_lane=(?P<capture>\d+)"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Camera V2 V7.1 no-flicker live log")
    parser.add_argument("log", nargs="?", default="/tmp/CAMERA_BBOX_V71.log")
    parser.add_argument("--camera", default="CAM-01")
    parser.add_argument("--min-source-fps", type=float, default=18.0)
    parser.add_argument("--min-tracker-hz", type=float, default=14.0)
    parser.add_argument("--min-detector-hz", type=float, default=0.90)
    parser.add_argument("--max-det-hold-avg-ms", type=float, default=50.0)
    args = parser.parse_args()

    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"V71_ACCEPT FAIL log_missing={path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    clean_stats = [line for line in lines if line.startswith("CAMERA_CLEAN_STATS ")]
    lane_stats = [line for line in lines if line.startswith("CAMERA_GPU_LANE_STATS ")]
    v71_stats = [line for line in lines if line.startswith("CAMERA_BBOX_V71_STATS ")]
    if len(clean_stats) < 1:
        raise SystemExit("V71_ACCEPT FAIL no CAMERA_CLEAN_STATS rows")
    if len(lane_stats) < 1:
        raise SystemExit("V71_ACCEPT FAIL no CAMERA_GPU_LANE_STATS rows")
    if len(v71_stats) < 1:
        raise SystemExit("V71_ACCEPT FAIL no CAMERA_BBOX_V71_STATS rows")

    windows = clean_stats[-3:]
    source_values = []
    tracker_values = []
    detector_values = []
    for line in windows:
        sources = {m.group("cid"): float(m.group("fps")) for m in SOURCE_RE.finditer(line)}
        if args.camera in sources:
            source_values.append(sources[args.camera])
        tm = TRACKER_RE.search(line)
        if tm:
            tracker_values.append(float(tm.group("hz")))
        am = ACTUAL_RE.search(line)
        if am:
            actual = {
                m.group("cid"): float(m.group("hz"))
                for m in ACTUAL_ITEM_RE.finditer(am.group("body"))
            }
            if args.camera in actual:
                detector_values.append(actual[args.camera])

    lane_values = []
    for line in lane_stats[-3:]:
        match = LANE_RE.search(line)
        if match:
            lane_values.append(float(match.group("avg")))

    v71 = V71_RE.search(v71_stats[-1])
    if not source_values or not tracker_values or not detector_values or not lane_values or v71 is None:
        raise SystemExit("V71_ACCEPT FAIL incomplete stats fields")

    source_min = min(source_values)
    tracker_min = min(tracker_values)
    detector_min = min(detector_values)
    hold_avg_max = max(lane_values)
    holds = int(v71.group("holds"))
    expires = int(v71.group("expires"))
    captures = int(v71.group("capture"))

    failures = []
    if source_min < args.min_source_fps:
        failures.append(f"source_fps={source_min:.1f}<{args.min_source_fps:.1f}")
    if tracker_min < args.min_tracker_hz:
        failures.append(f"tracker_hz={tracker_min:.1f}<{args.min_tracker_hz:.1f}")
    if detector_min < args.min_detector_hz:
        failures.append(f"detector_hz={detector_min:.2f}<{args.min_detector_hz:.2f}")
    if hold_avg_max > args.max_det_hold_avg_ms:
        failures.append(
            f"det_hold_avg={hold_avg_max:.1f}ms>{args.max_det_hold_avg_ms:.1f}ms"
        )
    if captures <= 0:
        failures.append("capture_before_lane=0")

    fatal_markers = [
        line
        for line in lines[-600:]
        if "CAMERA_CLEAN_GST ERROR" in line
        or "TRT86 fatal" in line
        or "result timeout" in line
    ]
    if fatal_markers:
        failures.append(f"runtime_errors={len(fatal_markers)}")

    status = "PASS" if not failures else "FAIL"
    print(
        f"V71_ACCEPT {status} camera={args.camera} "
        f"source_min={source_min:.1f}fps tracker_min={tracker_min:.1f}Hz "
        f"detector_min={detector_min:.2f}Hz det_hold_avg_max={hold_avg_max:.1f}ms "
        f"empty_holds={holds} empty_expires={expires} capture_before_lane={captures} "
        f"windows={len(windows)}"
    )
    if failures:
        print("V71_ACCEPT reasons=" + ",".join(failures))
        return 1

    print(
        "V71_ACCEPT visual_check=walk+bend+arms-up: bbox must not blink; "
        "when tracking truly disappears, the last real box may remain only briefly and must then vanish"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
