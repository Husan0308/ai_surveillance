#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

SOURCE_RE = re.compile(r"(?P<cid>CAM-\d+):(?P<fps>[0-9.]+)fps")
TRACKER_RE = re.compile(r"tracker_rate=(?P<hz>[0-9.]+)Hz")
TRACKED_RE = re.compile(r"tracked_now=(?P<n>\d+)")
V84_RE = re.compile(
    r"global_actual=(?P<global>[0-9.]+)Hz.*gpu_ema=(?P<gpu>[0-9.]+)ms.*"
    r"roundtrip_ema=(?P<rt>[0-9.]+)ms.*gpu_duty_est=(?P<duty>[0-9.]+).*"
    r"result_age=(?P<age>[0-9.]+)ms.*capture_miss=(?P<miss>\d+)"
)
BASELINE_V85_GPU_MS = 177.3


def main() -> int:
    ap = argparse.ArgumentParser(description="V8.9 visual compositor GPU-cost A/B")
    ap.add_argument("log", nargs="?", default="/tmp/CAMERA_BBOX_V89.log")
    ap.add_argument("--camera", default="CAM-01")
    args = ap.parse_args()
    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"V89_DIAG FAIL log_missing={path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    clean = [x for x in lines if x.startswith("CAMERA_CLEAN_STATS ")]
    v84 = [x for x in lines if x.startswith("CAMERA_V84_STATS ")]
    v89 = [x for x in lines if x.startswith("CAMERA_V89_STATS ")]
    if len(clean) < 3 or not v84 or not v89:
        raise SystemExit("V89_DIAG FAIL insufficient_runtime_data; run 50-60 seconds")

    all_source: list[float] = []
    cam_source: list[float] = []
    tracker: list[float] = []
    tracked: list[int] = []
    for line in clean[-3:]:
        src = {m.group("cid"): float(m.group("fps")) for m in SOURCE_RE.finditer(line)}
        all_source.extend(src.values())
        if args.camera in src:
            cam_source.append(src[args.camera])
        tm = TRACKER_RE.search(line)
        if tm:
            tracker.append(float(tm.group("hz")))
        tr = TRACKED_RE.search(line)
        if tr:
            tracked.append(int(tr.group("n")))

    vm = V84_RE.search(v84[-1])
    if not vm:
        raise SystemExit("V89_DIAG FAIL malformed_v84_stats")
    source_min = min(all_source) if all_source else 0.0
    cam_min = min(cam_source) if cam_source else 0.0
    tracker_min = min(tracker) if tracker else 0.0
    tracked_max = max(tracked) if tracked else 0
    gpu = float(vm.group("gpu"))
    rt = float(vm.group("rt"))
    global_actual = float(vm.group("global"))
    duty = float(vm.group("duty"))
    age = float(vm.group("age"))
    miss = int(vm.group("miss"))
    speedup = BASELINE_V85_GPU_MS / max(gpu, 0.001)
    saved_pct = 100.0 * max(0.0, BASELINE_V85_GPU_MS - gpu) / BASELINE_V85_GPU_MS

    health_failures: list[str] = []
    if source_min < 18.0:
        health_failures.append(f"all_source_min={source_min:.1f}<18")
    if cam_min < 18.0:
        health_failures.append(f"{args.camera}_fps={cam_min:.1f}<18")
    if tracker_min < 6.5:
        health_failures.append(f"tracker_hz={tracker_min:.1f}<6.5")
    if tracked_max <= 0:
        health_failures.append("tracked_now=0")

    if gpu < 140.0:
        diagnosis = "DISPLAY_MAJOR"
        next_step = "optimize production display compositor before shared-context work"
    elif gpu < 160.0:
        diagnosis = "DISPLAY_PARTIAL"
        next_step = "profile display-mux scaling separately; display contributes but is not the only bottleneck"
    else:
        diagnosis = "DISPLAY_NOT_ROOT"
        next_step = "restore production display and move to same-process/shared-primary-context feasibility"

    status = "PASS" if not health_failures else "FAIL"
    print(
        f"V89_DIAG {status} diagnosis={diagnosis} camera={args.camera} "
        f"source_min={source_min:.1f}fps camera_min={cam_min:.1f}fps "
        f"tracker_min={tracker_min:.1f}Hz tracked_now_max={tracked_max} "
        f"global_actual={global_actual:.2f}Hz gpu_ema={gpu:.1f}ms "
        f"roundtrip_ema={rt:.1f}ms result_age={age:.0f}ms duty={duty:.2f} "
        f"capture_miss={miss} baseline={BASELINE_V85_GPU_MS:.1f}ms "
        f"speedup={speedup:.2f}x saved={saved_pct:.1f}% headless_compositor=1 windows=3"
    )
    if health_failures:
        print("V89_DIAG health_reasons=" + ",".join(health_failures))
    print("V89_DIAG next=" + next_step)
    return 0 if not health_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
