#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

SOURCE_RE = re.compile(r"(?P<cid>CAM-\d+):(?P<fps>[0-9.]+)fps")
TRACKER_RE = re.compile(r"tracker_rate=(?P<hz>[0-9.]+)Hz")
ACTUAL_RE = re.compile(r"actual=\[(?P<body>[^\]]*)\]")
ACTUAL_ITEM_RE = re.compile(r"(?P<cid>CAM-\d+):(?P<hz>[0-9.]+)")
TRACKED_RE = re.compile(r"tracked_now=(?P<n>\d+)")
V84_RE = re.compile(
    r"global_actual=(?P<global>[0-9.]+)Hz.*per_camera_target=(?P<per>[0-9.]+)Hz.*"
    r"gpu_ema=(?P<gpu>[0-9.]+)ms.*roundtrip_ema=(?P<rt>[0-9.]+)ms.*"
    r"gpu_duty_est=(?P<duty>[0-9.]+).*result_age=(?P<age>[0-9.]+)ms.*"
    r"capture_miss=(?P<miss>\d+).*fairness=\[(?P<fair>[^\]]*)\]"
)


def main() -> int:
    ap = argparse.ArgumentParser(description="V8.4 batch1 low-latency acceptance")
    ap.add_argument("log", nargs="?", default="/tmp/CAMERA_BBOX_V84.log")
    ap.add_argument("--camera", default="CAM-01")
    args = ap.parse_args()

    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"V84_ACCEPT FAIL log_missing={path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    clean = [x for x in lines if x.startswith("CAMERA_CLEAN_STATS ")]
    stats = [x for x in lines if x.startswith("CAMERA_V84_STATS ")]
    det = [x for x in lines if x.startswith("CAMERA_V84_DETECT ")]
    if len(clean) < 3 or not stats or not det:
        raise SystemExit("V84_ACCEPT FAIL insufficient_runtime_data; run 40-60 seconds")

    windows = clean[-3:]
    all_source = []
    cam_source = []
    tracker = []
    camera_detector = []
    tracked = []
    for line in windows:
        src = {m.group("cid"): float(m.group("fps")) for m in SOURCE_RE.finditer(line)}
        all_source.extend(src.values())
        if args.camera in src:
            cam_source.append(src[args.camera])
        tm = TRACKER_RE.search(line)
        if tm:
            tracker.append(float(tm.group("hz")))
        am = ACTUAL_RE.search(line)
        if am:
            actual = {m.group("cid"): float(m.group("hz")) for m in ACTUAL_ITEM_RE.finditer(am.group("body"))}
            if args.camera in actual:
                camera_detector.append(actual[args.camera])
        tr = TRACKED_RE.search(line)
        if tr:
            tracked.append(int(tr.group("n")))

    vm = V84_RE.search(stats[-1])
    if not vm:
        raise SystemExit("V84_ACCEPT FAIL malformed_CAMERA_V84_STATS")

    source_min = min(all_source) if all_source else 0.0
    cam_min = min(cam_source) if cam_source else 0.0
    tracker_min = min(tracker) if tracker else 0.0
    detector_min = min(camera_detector) if camera_detector else 0.0
    tracked_max = max(tracked) if tracked else 0
    global_actual = float(vm.group("global"))
    per_target = float(vm.group("per"))
    gpu = float(vm.group("gpu"))
    rt = float(vm.group("rt"))
    duty = float(vm.group("duty"))
    age = float(vm.group("age"))
    miss = int(vm.group("miss"))

    failures = []
    if source_min < 18.0:
        failures.append(f"all_source_min={source_min:.1f}<18")
    if cam_min < 18.0:
        failures.append(f"{args.camera}_fps={cam_min:.1f}<18")
    if tracker_min < 6.0:
        failures.append(f"tracker_hz={tracker_min:.1f}<6")
    # The clean-room batch1 baseline is ~60 ms. More than ~3x in production means
    # cross-context contention is still dominating and bbox work must wait.
    if gpu > 180.0:
        failures.append(f"gpu_ema={gpu:.1f}>180ms")
    if rt > 220.0:
        failures.append(f"roundtrip_ema={rt:.1f}>220ms")
    if age > 260.0:
        failures.append(f"result_age={age:.0f}>260ms")
    if duty > 0.42:
        failures.append(f"gpu_duty={duty:.2f}>0.42")
    if global_actual < 1.8:
        failures.append(f"global_detector={global_actual:.2f}<1.8Hz")

    status = "PASS" if not failures else "FAIL"
    print(
        f"V84_ACCEPT {status} camera={args.camera} all_source_min={source_min:.1f}fps "
        f"camera_min={cam_min:.1f}fps tracker_min={tracker_min:.1f}Hz "
        f"detector_min={detector_min:.2f}Hz global_actual={global_actual:.2f}Hz "
        f"per_camera_target={per_target:.2f}Hz gpu_ema={gpu:.1f}ms "
        f"roundtrip_ema={rt:.1f}ms result_age={age:.0f}ms duty={duty:.2f} "
        f"capture_miss={miss} tracked_now_max={tracked_max} windows={len(windows)}"
    )
    if failures:
        print("V84_ACCEPT reasons=" + ",".join(failures))
        print(
            "V84_ACCEPT next=if_gpu_slow_keep_bbox_untouched_and_test DeepStream/TRT context scheduling; "
            "if_gpu_fast_then move to tracker/PTS bbox step"
        )
        return 1
    print(
        "V84_ACCEPT next=detector_latency_recovered; now bbox/PTS can be tuned separately without changing detector batching"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
