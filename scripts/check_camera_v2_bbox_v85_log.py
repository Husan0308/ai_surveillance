#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

SOURCE_RE = re.compile(r"(?P<cid>CAM-\d+):(?P<fps>[0-9.]+)fps")
TRACKER_RE = re.compile(r"tracker_rate=(?P<hz>[0-9.]+)Hz")
TRACKED_RE = re.compile(r"tracked_now=(?P<n>\d+)")
V84_RE = re.compile(
    r"global_actual=(?P<global>[0-9.]+)Hz.*per_camera_target=(?P<per>[0-9.]+)Hz.*"
    r"gpu_ema=(?P<gpu>[0-9.]+)ms.*roundtrip_ema=(?P<rt>[0-9.]+)ms.*"
    r"gpu_duty_est=(?P<duty>[0-9.]+).*result_age=(?P<age>[0-9.]+)ms.*"
    r"capture_miss=(?P<miss>\d+)"
)
V85_RE = re.compile(r"feature_level=(?P<level>\d+).*gpu_ema=(?P<gpu>[0-9.]+)ms")

BASELINE_GPU_MS = 172.5


def main() -> int:
    ap = argparse.ArgumentParser(description="V8.5 NvDCF feature-size relief A/B acceptance")
    ap.add_argument("log", nargs="?", default="/tmp/CAMERA_BBOX_V85.log")
    ap.add_argument("--camera", default="CAM-01")
    args = ap.parse_args()

    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"V85_ACCEPT FAIL log_missing={path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    clean = [x for x in lines if x.startswith("CAMERA_CLEAN_STATS ")]
    stats84 = [x for x in lines if x.startswith("CAMERA_V84_STATS ")]
    stats85 = [x for x in lines if x.startswith("CAMERA_V85_STATS ")]
    arch85 = [x for x in lines if x.startswith("CAMERA_V85_ARCH ")]
    if len(clean) < 3 or not stats84 or not stats85 or not arch85:
        raise SystemExit("V85_ACCEPT FAIL insufficient_runtime_data; run 45-60 seconds")

    sources = []
    cam_sources = []
    tracker = []
    tracked = []
    for line in clean[-3:]:
        src = {m.group("cid"): float(m.group("fps")) for m in SOURCE_RE.finditer(line)}
        sources.extend(src.values())
        if args.camera in src:
            cam_sources.append(src[args.camera])
        tm = TRACKER_RE.search(line)
        if tm:
            tracker.append(float(tm.group("hz")))
        tr = TRACKED_RE.search(line)
        if tr:
            tracked.append(int(tr.group("n")))

    m84 = V84_RE.search(stats84[-1])
    m85 = V85_RE.search(stats85[-1])
    if not m84 or not m85:
        raise SystemExit("V85_ACCEPT FAIL malformed_stats")

    source_min = min(sources) if sources else 0.0
    cam_min = min(cam_sources) if cam_sources else 0.0
    tracker_min = min(tracker) if tracker else 0.0
    tracked_max = max(tracked) if tracked else 0
    global_hz = float(m84.group("global"))
    gpu = float(m84.group("gpu"))
    rt = float(m84.group("rt"))
    age = float(m84.group("age"))
    duty = float(m84.group("duty"))
    feature_level = int(m85.group("level"))
    improvement = BASELINE_GPU_MS / gpu if gpu > 0 else 0.0

    failures = []
    if feature_level != 1:
        failures.append(f"feature_level={feature_level}!=1")
    if source_min < 17.5:
        failures.append(f"all_source_min={source_min:.1f}<17.5")
    if cam_min < 18.0:
        failures.append(f"{args.camera}_fps={cam_min:.1f}<18")
    if tracker_min < 6.5:
        failures.append(f"tracker_hz={tracker_min:.1f}<6.5")
    if tracked_max <= 0:
        failures.append("tracked_now=0")
    # V8.4 measured ~172.5 ms integrated GPU EMA. Require at least ~15% recovery
    # from this single NvDCF feature-size change before touching more tracker knobs.
    if gpu > 150.0:
        failures.append(f"gpu_ema={gpu:.1f}>150ms")
    if rt > 185.0:
        failures.append(f"roundtrip_ema={rt:.1f}>185ms")
    if age > 240.0:
        failures.append(f"result_age={age:.0f}>240ms")
    if global_hz < 1.9:
        failures.append(f"global_detector={global_hz:.2f}<1.9Hz")

    status = "PASS" if not failures else "FAIL"
    print(
        f"V85_ACCEPT {status} camera={args.camera} feature_level={feature_level} "
        f"source_min={source_min:.1f}fps camera_min={cam_min:.1f}fps "
        f"tracker_min={tracker_min:.1f}Hz tracked_now_max={tracked_max} "
        f"global_actual={global_hz:.2f}Hz gpu_ema={gpu:.1f}ms "
        f"roundtrip_ema={rt:.1f}ms result_age={age:.0f}ms duty={duty:.2f} "
        f"vs_v84_gpu={BASELINE_GPU_MS:.1f}ms speedup={improvement:.2f}x windows=3"
    )
    if failures:
        print("V85_ACCEPT reasons=" + ",".join(failures))
        print(
            "V85_ACCEPT next=do-not-touch-bbox; if feature-level1 is insufficient, "
            "next A/B is tracker resolution 512x288 -> 480x288 while keeping feature-level1"
        )
        return 1
    print(
        "V85_ACCEPT next=NvDCF contention materially reduced; keep this baseline and move to PTS/current-box flicker fix"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
