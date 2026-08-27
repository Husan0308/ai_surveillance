#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

WARM_RE = re.compile(
    r"CAMERA_V83_TRT_WARMUP .*first=(?P<first>[0-9.]+)ms .*last=(?P<last>[0-9.]+)ms "
    r"median_tail=(?P<med>[0-9.]+)ms p95_tail=(?P<p95>[0-9.]+)ms"
)
TRT_RE = re.compile(r"CAMERA_V8_TRT_BATCH .*gpu=(?P<gpu>[0-9.]+)ms .*roundtrip=(?P<rt>[0-9.]+)ms")
STATS_RE = re.compile(
    r"CAMERA_V8_STATS .*batch_actual=(?P<hz>[0-9.]+)Hz .*gpu_ema=(?P<gpu>[0-9.]+)ms "
    r"roundtrip_ema=(?P<rt>[0-9.]+)ms .*real_updates=(?P<updates>\d+)"
)
SOURCE_RE = re.compile(r"(?P<cid>CAM-\d+):(?P<fps>[0-9.]+)fps")
TRACKER_RE = re.compile(r"tracker_rate=(?P<hz>[0-9.]+)Hz")
TRACKED_RE = re.compile(r"tracked_now=(?P<n>\d+)")


def main() -> int:
    p = argparse.ArgumentParser(description="V8.3 Step-1 detector latency acceptance")
    p.add_argument("log", nargs="?", default="/tmp/CAMERA_V83_STEP1.log")
    p.add_argument("--camera", default="CAM-01")
    args = p.parse_args()

    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"V83_STEP1 FAIL log_missing={path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    warm = [WARM_RE.search(x) for x in lines if "CAMERA_V83_TRT_WARMUP" in x]
    warm = [m for m in warm if m]
    trt = [TRT_RE.search(x) for x in lines if x.startswith("CAMERA_V8_TRT_BATCH ")]
    trt = [m for m in trt if m]
    stats = [STATS_RE.search(x) for x in lines if x.startswith("CAMERA_V8_STATS ")]
    stats = [m for m in stats if m]
    clean = [x for x in lines if x.startswith("CAMERA_CLEAN_STATS ")]

    if not warm:
        raise SystemExit("V83_STEP1 FAIL no warmup row")
    if len(trt) < 3 or not stats or len(clean) < 3:
        raise SystemExit("V83_STEP1 FAIL insufficient runtime data; run 30-45 seconds")

    wm = warm[-1]
    warm_med = float(wm.group("med"))
    warm_p95 = float(wm.group("p95"))
    # Use last five live TRT rows when available so a cold first request cannot hide
    # the steady-state result and a lucky single request cannot pass a bad runtime.
    recent_trt = trt[-5:]
    live_gpu_max = max(float(m.group("gpu")) for m in recent_trt)
    live_gpu_med = sorted(float(m.group("gpu")) for m in recent_trt)[len(recent_trt)//2]
    live_rt_max = max(float(m.group("rt")) for m in recent_trt)

    sm = stats[-1]
    gpu_ema = float(sm.group("gpu"))
    rt_ema = float(sm.group("rt"))
    batch_hz = float(sm.group("hz"))
    updates = int(sm.group("updates"))

    windows = clean[-3:]
    source_values = []
    tracker_values = []
    tracked_values = []
    for line in windows:
        sources = {m.group("cid"): float(m.group("fps")) for m in SOURCE_RE.finditer(line)}
        if args.camera in sources:
            source_values.append(sources[args.camera])
        tm = TRACKER_RE.search(line)
        if tm:
            tracker_values.append(float(tm.group("hz")))
        nm = TRACKED_RE.search(line)
        if nm:
            tracked_values.append(int(nm.group("n")))

    failures = []
    if warm_med > 220.0:
        failures.append(f"warmup_med={warm_med:.1f}>220ms")
    if live_gpu_med > 220.0:
        failures.append(f"live_gpu_med={live_gpu_med:.1f}>220ms")
    if gpu_ema > 220.0:
        failures.append(f"gpu_ema={gpu_ema:.1f}>220ms")
    if rt_ema > 260.0:
        failures.append(f"roundtrip_ema={rt_ema:.1f}>260ms")
    if batch_hz < 1.40:
        failures.append(f"detector_hz={batch_hz:.2f}<1.40")
    if source_values and min(source_values) < 18.0:
        failures.append(f"{args.camera}_fps={min(source_values):.1f}<18")
    if tracker_values and min(tracker_values) < 6.0:
        failures.append(f"tracker_hz={min(tracker_values):.1f}<6")
    if max(tracked_values or [0]) <= 0:
        failures.append("tracked_now=0")
    if updates <= 0:
        failures.append("real_updates=0")

    status = "PASS" if not failures else "FAIL"
    print(
        f"V83_STEP1 {status} warm_first={float(wm.group('first')):.1f}ms "
        f"warm_med={warm_med:.1f}ms warm_p95={warm_p95:.1f}ms "
        f"live_gpu_med={live_gpu_med:.1f}ms live_gpu_max={live_gpu_max:.1f}ms "
        f"live_roundtrip_max={live_rt_max:.1f}ms gpu_ema={gpu_ema:.1f}ms "
        f"roundtrip_ema={rt_ema:.1f}ms detector={batch_hz:.2f}Hz "
        f"real_updates={updates} tracked_now_max={max(tracked_values or [0])}"
    )
    if failures:
        print("V83_STEP1 reasons=" + ",".join(failures))
        print(
            "V83_STEP1 next=do-not-touch-bbox; compare warmup vs live GPU latency. "
            "warm_fast/live_slow => DeepStream/TRT context contention; warm_slow => clock/engine/runtime issue"
        )
        return 1
    print(
        "V83_STEP1 next=detector latency recovered; only now proceed to bbox temporal fixes on top of this exact runtime"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
