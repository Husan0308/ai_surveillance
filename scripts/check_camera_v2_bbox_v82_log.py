#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

SOURCE_RE = re.compile(r"(?P<cid>CAM-\d+):(?P<fps>[0-9.]+)fps")
TRACKER_RE = re.compile(r"tracker_rate=(?P<hz>[0-9.]+)Hz")
ACTUAL_RE = re.compile(r"actual=\[(?P<body>[^\]]*)\]")
ACTUAL_ITEM_RE = re.compile(r"(?P<cid>CAM-\d+):(?P<hz>[0-9.]+)")
V81_RE = re.compile(
    r"real_updates=(?P<updates>\d+).*cache_prunes=(?P<prunes>\d+).*"
    r"currentized=(?P<currentized>\d+).*raw_new=(?P<raw_new>\d+).*"
    r"stale_new_dropped=(?P<dropped>\d+).*empty_detector_skips=(?P<empty>\d+).*"
    r"track_conf_p10=(?P<conf10>[0-9.]+).*track_conf_p50=(?P<conf50>[0-9.]+).*"
    r"overlay_age_p50=(?P<age50>[0-9.]+)ms.*overlay_age_p95=(?P<age95>[0-9.]+)ms.*"
    r"held_draw_ratio=(?P<held>[0-9.]+)"
)
V8_RE = re.compile(
    r"gpu_ema=(?P<gpu>[0-9.]+)ms.*roundtrip_ema=(?P<roundtrip>[0-9.]+)ms.*"
    r"gpu_duty_est=(?P<duty>[0-9.]+).*result_age=(?P<age>[0-9.]+)ms"
)
V82_RE = re.compile(
    r"comp_draws=(?P<draws>\d+).*comp_shift_avg=(?P<avg>[0-9.]+)px.*"
    r"comp_shift_max=(?P<max>[0-9.]+)px.*tracker_target=(?P<target>[0-9.]+)Hz"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check V8.2 low-latency bbox acceptance")
    parser.add_argument("log", nargs="?", default="/tmp/CAMERA_BBOX_V82.log")
    parser.add_argument("--camera", default="CAM-01")
    parser.add_argument("--min-source-fps", type=float, default=18.0)
    parser.add_argument("--min-tracker-hz", type=float, default=6.5)
    parser.add_argument("--min-detector-hz", type=float, default=1.50)
    parser.add_argument("--max-gpu-ema-ms", type=float, default=180.0)
    parser.add_argument("--max-roundtrip-ema-ms", type=float, default=230.0)
    parser.add_argument("--max-overlay-p95-ms", type=float, default=200.0)
    parser.add_argument("--max-held-ratio", type=float, default=0.35)
    args = parser.parse_args()

    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"V82_ACCEPT FAIL log_missing={path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    clean = [line for line in lines if line.startswith("CAMERA_CLEAN_STATS ")]
    v81 = [line for line in lines if line.startswith("CAMERA_V81_STATS ")]
    v8 = [line for line in lines if line.startswith("CAMERA_V8_STATS ")]
    v82 = [line for line in lines if line.startswith("CAMERA_V82_STATS ")]
    if len(clean) < 3:
        raise SystemExit("V82_ACCEPT FAIL need at least 3 CAMERA_CLEAN_STATS windows")
    if not v81 or not v8 or not v82:
        raise SystemExit("V82_ACCEPT FAIL missing V81/V8/V82 stats")

    windows = clean[-3:]
    source_values = []
    tracker_values = []
    detector_values = []
    for line in windows:
        sources = {m.group("cid"): float(m.group("fps")) for m in SOURCE_RE.finditer(line)}
        source_values.extend(sources.values())
        tm = TRACKER_RE.search(line)
        if tm:
            tracker_values.append(float(tm.group("hz")))
        am = ACTUAL_RE.search(line)
        if am:
            actual = {
                m.group("cid"): float(m.group("hz"))
                for m in ACTUAL_ITEM_RE.finditer(am.group("body"))
            }
            if args.camera in actual:
                detector_values.append(actual[args.camera])

    vm81 = V81_RE.search(v81[-1])
    vm8 = V8_RE.search(v8[-1])
    vm82 = V82_RE.search(v82[-1])
    if not source_values or not tracker_values or not detector_values or not vm81 or not vm8 or not vm82:
        raise SystemExit("V82_ACCEPT FAIL malformed/incomplete stats")

    source_min = min(source_values)
    tracker_min = min(tracker_values)
    detector_min = min(detector_values)
    gpu = float(vm8.group("gpu"))
    roundtrip = float(vm8.group("roundtrip"))
    age95 = float(vm81.group("age95"))
    held = float(vm81.group("held"))
    updates = int(vm81.group("updates"))
    conf50 = float(vm81.group("conf50"))

    failures = []
    if source_min < args.min_source_fps:
        failures.append(f"source_fps={source_min:.1f}<{args.min_source_fps:.1f}")
    if tracker_min < args.min_tracker_hz:
        failures.append(f"tracker_hz={tracker_min:.1f}<{args.min_tracker_hz:.1f}")
    if detector_min < args.min_detector_hz:
        failures.append(f"detector_hz={detector_min:.2f}<{args.min_detector_hz:.2f}")
    if gpu > args.max_gpu_ema_ms:
        failures.append(f"gpu_ema={gpu:.1f}>{args.max_gpu_ema_ms:.1f}ms")
    if roundtrip > args.max_roundtrip_ema_ms:
        failures.append(f"roundtrip_ema={roundtrip:.1f}>{args.max_roundtrip_ema_ms:.1f}ms")
    if age95 > args.max_overlay_p95_ms:
        failures.append(f"overlay_age_p95={age95:.0f}>{args.max_overlay_p95_ms:.0f}ms")
    if held > args.max_held_ratio:
        failures.append(f"held_draw_ratio={held:.3f}>{args.max_held_ratio:.3f}")
    if updates <= 0:
        failures.append("real_updates=0")
    if conf50 <= 0.0:
        failures.append("track_conf_p50=0")

    fatal = [
        line for line in lines[-1000:]
        if "CAMERA_CLEAN_GST ERROR" in line
        or "CAMERA_V81_TRACK warning=" in line
        or "TRT86 fatal" in line
    ]
    if fatal:
        failures.append(f"runtime_errors={len(fatal)}")

    status = "PASS" if not failures else "FAIL"
    print(
        f"V82_ACCEPT {status} camera={args.camera} source_min={source_min:.1f}fps "
        f"tracker_min={tracker_min:.1f}Hz detector_min={detector_min:.2f}Hz "
        f"gpu_ema={gpu:.1f}ms roundtrip_ema={roundtrip:.1f}ms "
        f"overlay_p95={age95:.0f}ms held_ratio={held:.3f} "
        f"real_updates={updates} conf_p50={conf50:.3f} "
        f"comp_draws={vm82.group('draws')} comp_avg={vm82.group('avg')}px "
        f"comp_max={vm82.group('max')}px windows={len(windows)}"
    )
    if failures:
        print("V82_ACCEPT reasons=" + ",".join(failures))
        return 1
    print(
        "V82_ACCEPT visual_check=walk quickly left/right: wall must stay low-latency and bbox "
        "must follow without the V8 frozen-behind effect; bend/arms-up must not blink"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
