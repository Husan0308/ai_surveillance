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


def main() -> int:
    parser = argparse.ArgumentParser(description="Check V8.1 temporal bbox acceptance")
    parser.add_argument("log", nargs="?", default="/tmp/CAMERA_BBOX_V81.log")
    parser.add_argument("--camera", default="CAM-01")
    parser.add_argument("--min-source-fps", type=float, default=18.0)
    parser.add_argument("--min-tracker-hz", type=float, default=9.5)
    parser.add_argument("--min-detector-hz", type=float, default=1.50)
    parser.add_argument("--max-overlay-p95-ms", type=float, default=180.0)
    parser.add_argument("--max-held-ratio", type=float, default=0.25)
    args = parser.parse_args()

    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"V81_ACCEPT FAIL log_missing={path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    clean = [line for line in lines if line.startswith("CAMERA_CLEAN_STATS ")]
    sync = [line for line in lines if line.startswith("CAMERA_V81_STATS ")]
    if len(clean) < 1:
        raise SystemExit("V81_ACCEPT FAIL no CAMERA_CLEAN_STATS; run at least 20 seconds")
    if not sync:
        raise SystemExit("V81_ACCEPT FAIL no CAMERA_V81_STATS")

    windows = clean[-3:]
    source_values = []
    tracker_values = []
    detector_values = []
    for line in windows:
        sources = {m.group("cid"): float(m.group("fps")) for m in SOURCE_RE.finditer(line)}
        if sources:
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

    if not source_values or not tracker_values or not detector_values:
        raise SystemExit("V81_ACCEPT FAIL incomplete CAMERA_CLEAN_STATS fields")

    vm = V81_RE.search(sync[-1])
    if not vm:
        raise SystemExit("V81_ACCEPT FAIL malformed CAMERA_V81_STATS")

    source_min = min(source_values)
    tracker_min = min(tracker_values)
    detector_min = min(detector_values)
    age95 = float(vm.group("age95"))
    held = float(vm.group("held"))
    failures = []
    if source_min < args.min_source_fps:
        failures.append(f"source_fps={source_min:.1f}<{args.min_source_fps:.1f}")
    if tracker_min < args.min_tracker_hz:
        failures.append(f"tracker_hz={tracker_min:.1f}<{args.min_tracker_hz:.1f}")
    if detector_min < args.min_detector_hz:
        failures.append(f"detector_hz={detector_min:.2f}<{args.min_detector_hz:.2f}")
    if age95 > args.max_overlay_p95_ms:
        failures.append(f"overlay_age_p95={age95:.0f}>{args.max_overlay_p95_ms:.0f}ms")
    if held > args.max_held_ratio:
        failures.append(f"held_draw_ratio={held:.3f}>{args.max_held_ratio:.3f}")

    fatal = [
        line for line in lines[-800:]
        if "CAMERA_CLEAN_GST ERROR" in line
        or "CAMERA_V81_TRACK warning=" in line
        or "TRT86 fatal" in line
    ]
    if fatal:
        failures.append(f"runtime_errors={len(fatal)}")

    status = "PASS" if not failures else "FAIL"
    print(
        f"V81_ACCEPT {status} camera={args.camera} source_min={source_min:.1f}fps "
        f"tracker_min={tracker_min:.1f}Hz detector_min={detector_min:.2f}Hz "
        f"overlay_p95={age95:.0f}ms held_ratio={held:.3f} "
        f"conf_p10={float(vm.group('conf10')):.3f} "
        f"currentized={vm.group('currentized')} empty_detector_skips={vm.group('empty')} "
        f"cache_prunes={vm.group('prunes')} windows={len(windows)}"
    )
    if failures:
        print("V81_ACCEPT reasons=" + ",".join(failures))
        return 1
    print(
        "V81_ACCEPT visual_check=walk quickly left/right + bend + arms-up: box must move with the body; "
        "no backwards snap at detector correction, no long frozen rectangle, no blink on one detector miss"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
