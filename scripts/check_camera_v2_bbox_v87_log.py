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
V87_RE = re.compile(
    r"tracker=(?P<w>\d+)x(?P<h>\d+).*gpu_ema=(?P<gpu>[0-9.]+)ms.*"
    r"roundtrip_ema=(?P<rt>[0-9.]+)ms.*tracked_now=(?P<tracked>\d+)"
)

BASELINE_V85_GPU_MS = 177.3


def main() -> int:
    ap = argparse.ArgumentParser(description="V8.7 NvDCF 480x288 A/B acceptance")
    ap.add_argument("log", nargs="?", default="/tmp/CAMERA_BBOX_V87.log")
    ap.add_argument("--camera", default="CAM-01")
    args = ap.parse_args()
    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"V87_ACCEPT FAIL log_missing={path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    clean = [x for x in lines if x.startswith("CAMERA_CLEAN_STATS ")]
    v84 = [x for x in lines if x.startswith("CAMERA_V84_STATS ")]
    v87 = [x for x in lines if x.startswith("CAMERA_V87_STATS ")]
    arch = [x for x in lines if x.startswith("CAMERA_V87_ARCH ")]
    if len(clean) < 3 or not v84 or not v87 or not arch:
        raise SystemExit("V87_ACCEPT FAIL insufficient_runtime_data; run 50-60 seconds")

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
    rm = V87_RE.search(v87[-1])
    if not vm or not rm:
        raise SystemExit("V87_ACCEPT FAIL malformed_stats")

    source_min = min(all_source) if all_source else 0.0
    cam_min = min(cam_source) if cam_source else 0.0
    tracker_min = min(tracker) if tracker else 0.0
    tracked_max = max(tracked) if tracked else 0
    global_actual = float(vm.group("global"))
    gpu = float(vm.group("gpu"))
    rt = float(vm.group("rt"))
    duty = float(vm.group("duty"))
    age = float(vm.group("age"))
    miss = int(vm.group("miss"))
    width = int(rm.group("w"))
    height = int(rm.group("h"))
    speedup = BASELINE_V85_GPU_MS / max(gpu, 0.001)

    failures: list[str] = []
    if (width, height) != (480, 288):
        failures.append(f"tracker={width}x{height}!=480x288")
    if "blocking_sync_override=0" not in arch[-1]:
        failures.append("blocking_sync_override_not_reverted")
    if source_min < 18.0:
        failures.append(f"all_source_min={source_min:.1f}<18")
    if cam_min < 18.0:
        failures.append(f"{args.camera}_fps={cam_min:.1f}<18")
    if tracker_min < 6.5:
        failures.append(f"tracker_hz={tracker_min:.1f}<6.5")
    if tracked_max <= 0:
        failures.append("tracked_now=0")
    if gpu >= 165.0:
        failures.append(f"gpu_ema={gpu:.1f}>=165ms")
    if rt >= 180.0:
        failures.append(f"roundtrip_ema={rt:.1f}>=180ms")
    if global_actual < 1.8:
        failures.append(f"global_detector={global_actual:.2f}<1.8Hz")

    status = "PASS" if not failures else "FAIL"
    print(
        f"V87_ACCEPT {status} camera={args.camera} tracker={width}x{height} "
        f"source_min={source_min:.1f}fps camera_min={cam_min:.1f}fps "
        f"tracker_min={tracker_min:.1f}Hz tracked_now_max={tracked_max} "
        f"global_actual={global_actual:.2f}Hz gpu_ema={gpu:.1f}ms "
        f"roundtrip_ema={rt:.1f}ms result_age={age:.0f}ms duty={duty:.2f} "
        f"capture_miss={miss} vs_v85_gpu={BASELINE_V85_GPU_MS:.1f}ms speedup={speedup:.2f}x windows=3"
    )
    if failures:
        print("V87_ACCEPT reasons=" + ",".join(failures))
        print(
            "V87_ACCEPT next=if_no_meaningful_gain stop reducing tracker quality; profile/serialize CUDA ownership before bbox changes"
        )
        return 1
    print(
        "V87_ACCEPT next=keep tracker480; detector contention is improved enough to move to bbox/PTS correctness separately"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
