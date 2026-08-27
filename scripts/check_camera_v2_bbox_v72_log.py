#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

SOURCE_RE = re.compile(r"(?P<cid>CAM-\d+):(?P<fps>[0-9.]+)fps")
TRACKER_RE = re.compile(r"tracker_rate=(?P<hz>[0-9.]+)Hz")
BATCH_RE = re.compile(r"batch=(?P<ms>[0-9.]+)ms")
ACTUAL_RE = re.compile(r"actual=\[(?P<body>[^\]]*)\]")
ACTUAL_ITEM_RE = re.compile(r"(?P<cid>CAM-\d+):(?P<hz>[0-9.]+)")
V72_RE = re.compile(
    r"real_updates=(?P<real>\d+).*empty_holds=(?P<holds>\d+).*empty_expires=(?P<expires>\d+)"
)
LANE_RE = re.compile(r"det_hold_avg=(?P<hold>[0-9.]+)ms")


def main() -> int:
    p = argparse.ArgumentParser(description="Check Camera V2 V7.2 live log")
    p.add_argument("log", nargs="?", default="/tmp/CAMERA_BBOX_V72.log")
    p.add_argument("--camera", default="CAM-01")
    p.add_argument("--min-source-fps", type=float, default=18.0)
    p.add_argument("--min-tracker-hz", type=float, default=7.5)
    p.add_argument("--min-detector-hz", type=float, default=0.90)
    p.add_argument("--max-trt-batch-ms", type=float, default=60.0)
    args = p.parse_args()

    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"V72_ACCEPT FAIL log_missing={path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    clean = [line for line in lines if line.startswith("CAMERA_CLEAN_STATS ")]
    v72 = [line for line in lines if line.startswith("CAMERA_BBOX_V72_STATS ")]
    lane = [line for line in lines if line.startswith("CAMERA_GPU_LANE_STATS ")]
    if not clean or not v72:
        raise SystemExit("V72_ACCEPT FAIL insufficient stats; run at least 30 seconds")

    windows = clean[-3:]
    source_vals, tracker_vals, detector_vals, batch_vals = [], [], [], []
    for line in windows:
        sources = {m.group("cid"): float(m.group("fps")) for m in SOURCE_RE.finditer(line)}
        if args.camera in sources:
            source_vals.append(sources[args.camera])
        tm = TRACKER_RE.search(line)
        if tm:
            tracker_vals.append(float(tm.group("hz")))
        bm = BATCH_RE.search(line)
        if bm:
            batch_vals.append(float(bm.group("ms")))
        am = ACTUAL_RE.search(line)
        if am:
            actual = {m.group("cid"): float(m.group("hz")) for m in ACTUAL_ITEM_RE.finditer(am.group("body"))}
            if args.camera in actual:
                detector_vals.append(actual[args.camera])

    if not (source_vals and tracker_vals and detector_vals and batch_vals):
        raise SystemExit("V72_ACCEPT FAIL incomplete stats fields")

    source_min = min(source_vals)
    tracker_min = min(tracker_vals)
    detector_min = min(detector_vals)
    trt_max = max(batch_vals)
    failures = []
    if source_min < args.min_source_fps:
        failures.append(f"source_fps={source_min:.1f}<{args.min_source_fps:.1f}")
    if tracker_min < args.min_tracker_hz:
        failures.append(f"tracker_hz={tracker_min:.1f}<{args.min_tracker_hz:.1f}")
    if detector_min < args.min_detector_hz:
        failures.append(f"detector_hz={detector_min:.2f}<{args.min_detector_hz:.2f}")
    if trt_max > args.max_trt_batch_ms:
        failures.append(f"trt_batch={trt_max:.1f}>{args.max_trt_batch_ms:.1f}ms")

    vm = V72_RE.search(v72[-1])
    real = int(vm.group("real")) if vm else -1
    holds = int(vm.group("holds")) if vm else -1
    expires = int(vm.group("expires")) if vm else -1
    hold_avg = -1.0
    if lane:
        lm = LANE_RE.search(lane[-1])
        if lm:
            hold_avg = float(lm.group("hold"))

    fatal = [
        line for line in lines[-600:]
        if "CAMERA_CLEAN_GST ERROR" in line or "TRT86 fatal" in line or "result timeout" in line
    ]
    if fatal:
        failures.append(f"runtime_errors={len(fatal)}")

    status = "PASS" if not failures else "FAIL"
    print(
        f"V72_ACCEPT {status} camera={args.camera} source_min={source_min:.1f}fps "
        f"tracker_min={tracker_min:.1f}Hz detector_min={detector_min:.2f}Hz "
        f"trt_batch_max={trt_max:.1f}ms det_hold_avg={hold_avg:.1f}ms "
        f"real_updates={real} empty_holds={holds} empty_expires={expires} windows={len(windows)}"
    )
    if failures:
        print("V72_ACCEPT reasons=" + ",".join(failures))
        return 1
    print(
        "V72_ACCEPT visual_check=no off/on blink during normal tracking; walk+bend+arms-up; "
        "bbox may remain frozen for at most 300ms only when normal NvDCF metadata briefly disappears"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
