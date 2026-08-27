#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


SOURCE_RE = re.compile(r"(?P<cid>CAM-\d+):(?P<fps>[0-9.]+)fps")
TRACKER_RE = re.compile(r"tracker_rate=(?P<hz>[0-9.]+)Hz")
ACTUAL_RE = re.compile(r"actual=\[(?P<body>[^\]]*)\]")
ACTUAL_ITEM_RE = re.compile(r"(?P<cid>CAM-\d+):(?P<hz>[0-9.]+)")
V7_RE = re.compile(
    r"teleport_events=(?P<teleport>\d+).*empty_cache_clears=(?P<clears>\d+)"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check a Camera V2 V7 live acceptance log")
    parser.add_argument("log", nargs="?", default="/tmp/CAMERA_BBOX_V7.log")
    parser.add_argument("--camera", default="CAM-01")
    parser.add_argument("--min-source-fps", type=float, default=18.0)
    parser.add_argument("--min-tracker-hz", type=float, default=14.0)
    parser.add_argument("--min-detector-hz", type=float, default=0.90)
    args = parser.parse_args()

    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"V7_ACCEPT FAIL log_missing={path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    clean_stats = [line for line in lines if line.startswith("CAMERA_CLEAN_STATS ")]
    v7_stats = [line for line in lines if line.startswith("CAMERA_BBOX_V7_STATS ")]
    if not clean_stats:
        raise SystemExit("V7_ACCEPT FAIL no CAMERA_CLEAN_STATS rows; run for at least 10 seconds")
    if not v7_stats:
        raise SystemExit("V7_ACCEPT FAIL no CAMERA_BBOX_V7_STATS rows")

    # Use the last three windows when available: one lucky instant must not pass a bad run.
    windows = clean_stats[-3:]
    source_fps_values = []
    tracker_hz_values = []
    detector_hz_values = []
    for line in windows:
        sources = {m.group("cid"): float(m.group("fps")) for m in SOURCE_RE.finditer(line)}
        if args.camera in sources:
            source_fps_values.append(sources[args.camera])
        tm = TRACKER_RE.search(line)
        if tm:
            tracker_hz_values.append(float(tm.group("hz")))
        am = ACTUAL_RE.search(line)
        if am:
            actual = {
                m.group("cid"): float(m.group("hz"))
                for m in ACTUAL_ITEM_RE.finditer(am.group("body"))
            }
            if args.camera in actual:
                detector_hz_values.append(actual[args.camera])

    if not source_fps_values or not tracker_hz_values or not detector_hz_values:
        raise SystemExit("V7_ACCEPT FAIL incomplete stats fields")

    source_min = min(source_fps_values)
    tracker_min = min(tracker_hz_values)
    detector_min = min(detector_hz_values)
    failures = []
    if source_min < args.min_source_fps:
        failures.append(f"source_fps={source_min:.1f}<{args.min_source_fps:.1f}")
    if tracker_min < args.min_tracker_hz:
        failures.append(f"tracker_hz={tracker_min:.1f}<{args.min_tracker_hz:.1f}")
    if detector_min < args.min_detector_hz:
        failures.append(f"detector_hz={detector_min:.2f}<{args.min_detector_hz:.2f}")

    # Teleports are diagnostic rather than an automatic failure: entering/exiting the
    # frame and legitimate ID changes can trigger the geometry heuristic. A growing
    # count during a single person's normal walk is a visual-association red flag.
    vm = V7_RE.search(v7_stats[-1])
    teleport = int(vm.group("teleport")) if vm else -1
    clears = int(vm.group("clears")) if vm else -1

    fatal_markers = [
        line for line in lines[-500:]
        if "CAMERA_CLEAN_GST ERROR" in line or "TRT86 fatal" in line or "result timeout" in line
    ]
    if fatal_markers:
        failures.append(f"runtime_errors={len(fatal_markers)}")

    status = "PASS" if not failures else "FAIL"
    print(
        f"V7_ACCEPT {status} camera={args.camera} "
        f"source_min={source_min:.1f}fps tracker_min={tracker_min:.1f}Hz "
        f"detector_min={detector_min:.2f}Hz teleport_events={teleport} "
        f"empty_cache_clears={clears} windows={len(windows)}"
    )
    if failures:
        print("V7_ACCEPT reasons=" + ",".join(failures))
        return 1
    print(
        "V7_ACCEPT visual_check=walk+bend+arms-up: bbox center must follow the current person; "
        "no frozen shadow box; limbs detected by YOLO/NvDCF should remain inside the display envelope"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
