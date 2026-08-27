#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


SOURCE_RE = re.compile(r"(?P<cid>CAM-\d+):(?P<fps>[0-9.]+)fps pts=(?P<step>\d+)/(?P<age>\d+)ms")
TRACKER_RE = re.compile(r"tracker_rate=(?P<hz>[0-9.]+)Hz")
ACTUAL_RE = re.compile(r"actual=\[(?P<body>[^\]]*)\]")
ACTUAL_ITEM_RE = re.compile(r"(?P<cid>CAM-\d+):(?P<hz>[0-9.]+)")
V8_FIELDS = {
    "batch_actual": re.compile(r"batch_actual=(?P<v>[0-9.]+)Hz"),
    "gpu_ema": re.compile(r"gpu_ema=(?P<v>[0-9.]+)ms"),
    "roundtrip_ema": re.compile(r"roundtrip_ema=(?P<v>[0-9.]+)ms"),
    "gpu_duty": re.compile(r"gpu_duty_est=(?P<v>[0-9.]+)"),
    "result_age": re.compile(r"result_age=(?P<v>[0-9.]+)ms"),
    "capture_partial": re.compile(r"capture_partial=(?P<v>\d+)"),
    "empty_holds": re.compile(r"empty_holds=(?P<v>\d+)"),
    "empty_expires": re.compile(r"empty_expires=(?P<v>\d+)"),
    "teleport": re.compile(r"teleport_events=(?P<v>\d+)"),
}


def field(line: str, name: str, default: float = -1.0) -> float:
    match = V8_FIELDS[name].search(line)
    return float(match.group("v")) if match else default


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate V8 live six-camera performance")
    parser.add_argument("log", nargs="?", default="/tmp/CAMERA_BBOX_V8.log")
    parser.add_argument("--camera", default="CAM-01")
    parser.add_argument("--min-source-fps", type=float, default=18.0)
    parser.add_argument("--min-tracker-hz", type=float, default=6.5)
    parser.add_argument("--min-detector-hz", type=float, default=0.65)
    parser.add_argument("--max-camera-lag-ms", type=float, default=180.0)
    parser.add_argument("--max-result-age-ms", type=float, default=800.0)
    args = parser.parse_args()

    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"V8_ACCEPT FAIL log_missing={path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    clean = [line for line in lines if line.startswith("CAMERA_CLEAN_STATS ")]
    v8 = [line for line in lines if line.startswith("CAMERA_V8_STATS ")]
    if len(clean) < 3 or not v8:
        raise SystemExit("V8_ACCEPT FAIL run for at least three stats windows")

    windows = clean[-3:]
    all_source_fps: list[float] = []
    selected_fps: list[float] = []
    selected_lag: list[float] = []
    tracker_hz: list[float] = []
    detector_hz: list[float] = []

    for line in windows:
        sources = {
            m.group("cid"): (float(m.group("fps")), float(m.group("age")))
            for m in SOURCE_RE.finditer(line)
        }
        if len(sources) != 6:
            raise SystemExit(f"V8_ACCEPT FAIL expected six source stats, got {sorted(sources)}")
        all_source_fps.extend(v[0] for v in sources.values())
        if args.camera not in sources:
            raise SystemExit(f"V8_ACCEPT FAIL camera_missing={args.camera}")
        selected_fps.append(sources[args.camera][0])
        selected_lag.append(sources[args.camera][1])

        tm = TRACKER_RE.search(line)
        if tm:
            tracker_hz.append(float(tm.group("hz")))
        am = ACTUAL_RE.search(line)
        if am:
            actual = {
                m.group("cid"): float(m.group("hz"))
                for m in ACTUAL_ITEM_RE.finditer(am.group("body"))
            }
            if args.camera in actual:
                detector_hz.append(actual[args.camera])

    if not tracker_hz or not detector_hz:
        raise SystemExit("V8_ACCEPT FAIL incomplete tracker/detector stats")

    last_v8 = v8[-1]
    result_age = field(last_v8, "result_age")
    batch_actual = field(last_v8, "batch_actual")
    gpu_ema = field(last_v8, "gpu_ema")
    roundtrip_ema = field(last_v8, "roundtrip_ema")
    gpu_duty = field(last_v8, "gpu_duty")
    capture_partial = int(field(last_v8, "capture_partial", 0))
    empty_holds = int(field(last_v8, "empty_holds", 0))
    empty_expires = int(field(last_v8, "empty_expires", 0))
    teleport = int(field(last_v8, "teleport", -1))

    source_all_min = min(all_source_fps)
    source_selected_min = min(selected_fps)
    lag_max = max(selected_lag)
    tracker_min = min(tracker_hz)
    detector_min = min(detector_hz)

    failures: list[str] = []
    if source_all_min < args.min_source_fps:
        failures.append(f"all_source_fps={source_all_min:.1f}<{args.min_source_fps:.1f}")
    if tracker_min < args.min_tracker_hz:
        failures.append(f"tracker_hz={tracker_min:.1f}<{args.min_tracker_hz:.1f}")
    if detector_min < args.min_detector_hz:
        failures.append(f"detector_hz={detector_min:.2f}<{args.min_detector_hz:.2f}")
    if lag_max > args.max_camera_lag_ms:
        failures.append(f"{args.camera}_lag={lag_max:.0f}ms>{args.max_camera_lag_ms:.0f}ms")
    if result_age >= 0.0 and result_age > args.max_result_age_ms:
        failures.append(f"result_age={result_age:.0f}ms>{args.max_result_age_ms:.0f}ms")
    if teleport > 0:
        failures.append(f"teleport_events={teleport}")
    if any("CAMERA_GPU_LANE_STATS" in line for line in lines):
        failures.append("legacy_gpu_lane_present")
    if not any("tracker_drop_for_detector=0" in line for line in lines):
        failures.append("missing_no-drop-contract")
    fatal = [
        line for line in lines[-800:]
        if "CAMERA_CLEAN_GST ERROR" in line or "V8 TRT86 fatal" in line or "result timeout" in line
    ]
    if fatal:
        failures.append(f"runtime_errors={len(fatal)}")

    status = "PASS" if not failures else "FAIL"
    print(
        f"V8_ACCEPT {status} camera={args.camera} all_source_min={source_all_min:.1f}fps "
        f"camera_source_min={source_selected_min:.1f}fps camera_lag_max={lag_max:.0f}ms "
        f"tracker_min={tracker_min:.1f}Hz detector_min={detector_min:.2f}Hz "
        f"batch_actual={batch_actual:.2f}Hz gpu_ema={gpu_ema:.1f}ms "
        f"roundtrip_ema={roundtrip_ema:.1f}ms gpu_duty={gpu_duty:.2f} "
        f"result_age={result_age:.0f}ms capture_partial={capture_partial} "
        f"empty_holds={empty_holds} empty_expires={empty_expires} teleport={teleport} windows=3"
    )
    if failures:
        print("V8_ACCEPT reasons=" + ",".join(failures))
        return 1
    print(
        "V8_ACCEPT visual_check=walk+bend+arms-up: no blink, no ID jump, "
        "current NvDCF box follows body; display must remain visibly low-latency"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
