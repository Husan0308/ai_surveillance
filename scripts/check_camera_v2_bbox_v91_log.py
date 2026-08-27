#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

BASELINE_GPU_MS = 177.3


def floats(lines: list[str], pattern: str) -> list[float]:
    rx = re.compile(pattern)
    out: list[float] = []
    for line in lines:
        m = rx.search(line)
        if m:
            out.append(float(m.group(1)))
    return out


def ints(lines: list[str], pattern: str) -> list[int]:
    rx = re.compile(pattern)
    out: list[int] = []
    for line in lines:
        m = rx.search(line)
        if m:
            out.append(int(m.group(1)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--camera", default="CAM-01")
    ap.add_argument("--source-min", type=float, default=18.0)
    ap.add_argument("--tracker-min", type=float, default=6.5)
    ap.add_argument("--gpu-max", type=float, default=140.0)
    ap.add_argument("--roundtrip-max", type=float, default=155.0)
    ap.add_argument("--global-min", type=float, default=1.90)
    args = ap.parse_args()

    lines = Path(args.log).read_text(encoding="utf-8", errors="replace").splitlines()
    if any("CAMERA_V91_ERROR" in x or "CAMERA_CLEAN_GST ERROR" in x for x in lines):
        print("V91_ACCEPT FAIL reason=runtime_error")
        return 2

    context = ints(lines, r"CAMERA_V91_CONTEXT .*same=(\d+)")
    warm = floats(lines, r"CAMERA_V91_READY .*warm_med=([0-9.]+)ms")
    v91 = [x for x in lines if "CAMERA_V91_STATS" in x][-3:]
    gpu = floats(v91, r"gpu_ema=([0-9.]+)ms")
    roundtrip = floats(v91, r"roundtrip_ema=([0-9.]+)ms")
    primary = ints(v91, r"primary_same=(\d+)")

    clean = [x for x in lines if "CAMERA_CLEAN_STATS" in x][-3:]
    source_values: list[float] = []
    camera_values: list[float] = []
    tracker_values: list[float] = []
    tracked_values: list[int] = []
    for line in clean:
        source_values.extend(float(v) for v in re.findall(r"CAM-\d+:([0-9.]+)fps", line))
        m = re.search(rf"{re.escape(args.camera)}:([0-9.]+)fps", line)
        if m:
            camera_values.append(float(m.group(1)))
        m = re.search(r"tracker_rate=([0-9.]+)Hz", line)
        if m:
            tracker_values.append(float(m.group(1)))
        m = re.search(r"tracked_now=(\d+)", line)
        if m:
            tracked_values.append(int(m.group(1)))

    v84 = [x for x in lines if "CAMERA_V84_STATS" in x][-3:]
    global_actual = floats(v84, r"global_actual=([0-9.]+)Hz")

    if not context or not warm or not gpu or not roundtrip or not source_values or not tracker_values:
        print("V91_ACCEPT FAIL reason=insufficient_log")
        return 2

    source_min = min(source_values)
    camera_min = min(camera_values) if camera_values else 0.0
    tracker_min = min(tracker_values)
    tracked_max = max(tracked_values) if tracked_values else 0
    gpu_last = gpu[-1]
    roundtrip_last = roundtrip[-1]
    global_last = global_actual[-1] if global_actual else 0.0
    warm_med = warm[-1]
    primary_same = min(context + primary) if primary else min(context)
    speedup = BASELINE_GPU_MS / gpu_last if gpu_last > 0 else 0.0

    reasons: list[str] = []
    if primary_same != 1:
        reasons.append("primary_context_not_shared")
    if source_min < args.source_min:
        reasons.append(f"source_fps={source_min:.1f}<{args.source_min:.1f}")
    if tracker_min < args.tracker_min:
        reasons.append(f"tracker_hz={tracker_min:.1f}<{args.tracker_min:.1f}")
    if tracked_max <= 0:
        reasons.append("tracked_now=0")
    if gpu_last >= args.gpu_max:
        reasons.append(f"gpu_ema={gpu_last:.1f}>={args.gpu_max:.1f}ms")
    if roundtrip_last >= args.roundtrip_max:
        reasons.append(f"roundtrip={roundtrip_last:.1f}>={args.roundtrip_max:.1f}ms")
    if global_last < args.global_min:
        reasons.append(f"global_detector={global_last:.2f}<{args.global_min:.2f}Hz")

    verdict = "FAIL" if reasons else "PASS"
    print(
        f"V91_ACCEPT {verdict} camera={args.camera} primary_same={primary_same} "
        f"warm_med={warm_med:.1f}ms source_min={source_min:.1f}fps "
        f"camera_min={camera_min:.1f}fps tracker_min={tracker_min:.1f}Hz "
        f"tracked_now_max={tracked_max} global_actual={global_last:.2f}Hz "
        f"gpu_ema={gpu_last:.1f}ms roundtrip_ema={roundtrip_last:.1f}ms "
        f"vs_v85_gpu={BASELINE_GPU_MS:.1f}ms speedup={speedup:.2f}x windows={len(v91)}"
    )
    if reasons:
        print("V91_ACCEPT reasons=" + ",".join(reasons))
        print(
            "V91_ACCEPT next=if_primary_same=1_but_gpu_still_~170ms then process-boundary "
            "was_not_root; profile actual NvDCF/TRT overlap before touching bbox"
        )
        return 1
    print(
        "V91_ACCEPT next=keep in-process primary-context detector; restore bbox work only "
        "after visual walk/bend/arms-up check"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
