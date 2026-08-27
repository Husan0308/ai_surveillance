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
V88_RE = re.compile(
    r"reservations=(?P<res>\d+).*reservation_ema=(?P<resms>[0-9.]+)ms.*"
    r"quiet_before=(?P<quiet>[0-9.]+)ms.*tracker_gate_drops=(?P<tdrop>\d+).*"
    r"mux_gate_drops=(?P<mdrop>\d+).*gpu_ema=(?P<gpu>[0-9.]+)ms"
)
BASELINE_V85_GPU_MS = 177.3


def main() -> int:
    ap = argparse.ArgumentParser(description="V8.8 serialized GPU ownership acceptance")
    ap.add_argument("log", nargs="?", default="/tmp/CAMERA_BBOX_V88.log")
    ap.add_argument("--camera", default="CAM-01")
    args = ap.parse_args()
    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"V88_ACCEPT FAIL log_missing={path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    clean = [x for x in lines if x.startswith("CAMERA_CLEAN_STATS ")]
    v84 = [x for x in lines if x.startswith("CAMERA_V84_STATS ")]
    v88 = [x for x in lines if x.startswith("CAMERA_V88_STATS ")]
    if len(clean) < 3 or not v84 or not v88:
        raise SystemExit("V88_ACCEPT FAIL insufficient_runtime_data; run 50-60 seconds")

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
    sm = V88_RE.search(v88[-1])
    if not vm or not sm:
        raise SystemExit("V88_ACCEPT FAIL malformed_stats")

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
    reservations = int(sm.group("res"))
    reservation_ms = float(sm.group("resms"))
    quiet = float(sm.group("quiet"))
    tracker_drops = int(sm.group("tdrop"))
    mux_drops = int(sm.group("mdrop"))
    speedup = BASELINE_V85_GPU_MS / max(gpu, 0.001)

    failures: list[str] = []
    if source_min < 18.0:
        failures.append(f"all_source_min={source_min:.1f}<18")
    if cam_min < 18.0:
        failures.append(f"{args.camera}_fps={cam_min:.1f}<18")
    if tracker_min < 5.5:
        failures.append(f"tracker_hz={tracker_min:.1f}<5.5")
    if tracked_max <= 0:
        failures.append("tracked_now=0")
    if reservations <= 0:
        failures.append("reservations=0")
    if tracker_drops + mux_drops <= 0:
        failures.append("reservation_did_not_gate_tracker")
    if gpu >= 150.0:
        failures.append(f"gpu_ema={gpu:.1f}>=150ms")
    if rt >= 165.0:
        failures.append(f"roundtrip_ema={rt:.1f}>=165ms")
    if global_actual < 1.9:
        failures.append(f"global_detector={global_actual:.2f}<1.9Hz")

    status = "PASS" if not failures else "FAIL"
    print(
        f"V88_ACCEPT {status} camera={args.camera} source_min={source_min:.1f}fps "
        f"camera_min={cam_min:.1f}fps tracker_min={tracker_min:.1f}Hz "
        f"tracked_now_max={tracked_max} global_actual={global_actual:.2f}Hz "
        f"gpu_ema={gpu:.1f}ms roundtrip_ema={rt:.1f}ms result_age={age:.0f}ms "
        f"duty={duty:.2f} capture_miss={miss} reservations={reservations} "
        f"reservation_ema={reservation_ms:.1f}ms quiet={quiet:.0f}ms "
        f"tracker_drops={tracker_drops} mux_drops={mux_drops} "
        f"vs_v85_gpu={BASELINE_V85_GPU_MS:.1f}ms speedup={speedup:.2f}x windows=3"
    )
    if failures:
        print("V88_ACCEPT reasons=" + ",".join(failures))
        print(
            "V88_ACCEPT next=if_gpu_latency_drops_but_tracker_stays>=5.5Hz keep serialization; "
            "if_no_gain stop GPU knob tuning and move to single-process/shared-context design"
        )
        return 1
    print(
        "V88_ACCEPT next=keep short serialization; then return to bbox PTS/current-frame alignment"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
