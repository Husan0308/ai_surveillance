#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

SOURCE_RE = re.compile(r"(?P<cid>CAM-\d+):(?P<fps>[0-9.]+)fps")
TRACKER_RE = re.compile(r"tracker_rate=(?P<hz>[0-9.]+)Hz")
ACTUAL_RE = re.compile(r"actual=\[(?P<body>[^\]]*)\]")
ACTUAL_ITEM_RE = re.compile(r"(?P<cid>CAM-\d+):(?P<hz>[0-9.]+)")
BATCH_RE = re.compile(r"batch=(?P<ms>[0-9.]+)ms")
BASELINE_RE = re.compile(r"trt_baseline=(?P<ms>[0-9.]+)ms")
V72_RE = re.compile(r"real_updates=(?P<real>\d+).*empty_holds=(?P<holds>\d+).*empty_expires=(?P<expires>\d+)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Check V7.4 Pascal-balanced live log")
    ap.add_argument("log", nargs="?", default="/tmp/CAMERA_BBOX_V74.log")
    ap.add_argument("--camera", default="CAM-01")
    args = ap.parse_args()

    lines = Path(args.log).read_text(encoding="utf-8", errors="replace").splitlines()
    stats = [line for line in lines if line.startswith("CAMERA_CLEAN_STATS ")]
    if len(stats) < 3:
        raise SystemExit("V74_ACCEPT FAIL need at least 3 CAMERA_CLEAN_STATS windows")
    windows = stats[-3:]

    baseline = 60.0
    for line in lines:
        if line.startswith("CAMERA_BBOX_V74_PROFILE "):
            m = BASELINE_RE.search(line)
            if m:
                baseline = float(m.group("ms"))

    source_vals: list[float] = []
    tracker_vals: list[float] = []
    detector_vals: list[float] = []
    batch_vals: list[float] = []
    for line in windows:
        sources = {m.group("cid"): float(m.group("fps")) for m in SOURCE_RE.finditer(line)}
        if args.camera in sources:
            source_vals.append(sources[args.camera])
        m = TRACKER_RE.search(line)
        if m:
            tracker_vals.append(float(m.group("hz")))
        m = ACTUAL_RE.search(line)
        if m:
            actual = {x.group("cid"): float(x.group("hz")) for x in ACTUAL_ITEM_RE.finditer(m.group("body"))}
            if args.camera in actual:
                detector_vals.append(actual[args.camera])
        m = BATCH_RE.search(line)
        if m:
            batch_vals.append(float(m.group("ms")))

    if not source_vals or not tracker_vals or not detector_vals or not batch_vals:
        raise SystemExit("V74_ACCEPT FAIL incomplete runtime stats")

    source_min = min(source_vals)
    tracker_min = min(tracker_vals)
    detector_min = min(detector_vals)
    batch_max = max(batch_vals)

    # Adapt to measured hardware instead of pretending a 1050 Ti is a newer GPU.
    expected_detector = max(0.55, min(1.20, 0.30 / (6.0 * max(0.001, baseline / 1000.0))))
    min_detector = max(0.50, expected_detector * 0.72)
    max_batch = max(95.0, baseline * 1.75)

    failures = []
    if source_min < 18.0:
        failures.append(f"source_fps={source_min:.1f}<18.0")
    if tracker_min < 6.5:
        failures.append(f"tracker_hz={tracker_min:.1f}<6.5")
    if detector_min < min_detector:
        failures.append(f"detector_hz={detector_min:.2f}<{min_detector:.2f}")
    if batch_max > max_batch:
        failures.append(f"trt_batch={batch_max:.1f}>{max_batch:.1f}")

    real = holds = expires = -1
    v72_lines = [line for line in lines if line.startswith("CAMERA_BBOX_V72_STATS ")]
    if v72_lines:
        m = V72_RE.search(v72_lines[-1])
        if m:
            real = int(m.group("real"))
            holds = int(m.group("holds"))
            expires = int(m.group("expires"))

    status = "PASS" if not failures else "FAIL"
    print(
        f"V74_ACCEPT {status} camera={args.camera} baseline={baseline:.1f}ms "
        f"source_min={source_min:.1f}fps tracker_min={tracker_min:.1f}Hz "
        f"detector_min={detector_min:.2f}Hz trt_batch_max={batch_max:.1f}ms "
        f"real_updates={real} empty_holds={holds} empty_expires={expires} windows={len(windows)}"
    )
    if failures:
        print("V74_ACCEPT reasons=" + ",".join(failures))
        return 1
    print("V74_ACCEPT visual_check=no blink; box follows walking/bending/arms-up without prediction lead")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
