#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

BASELINE_SOURCE_DISPLAY_P95 = {
    "CAM-01": 500.0,
    "CAM-02": 400.0,
    "CAM-03": 450.0,
    "CAM-04": 650.0,
    "CAM-05": 600.0,
    "CAM-06": 600.0,
}


def value(line: str, key: str, default: float = 0.0) -> float:
    match = re.search(rf"\b{re.escape(key)}=(-?[0-9.]+)", line)
    return float(match.group(1)) if match else default


def latest_camera_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "CAMERA_V95_PTS" not in line:
            continue
        match = re.search(r"\bcamera=(CAM-[0-9]+)\b", line)
        if match:
            out[match.group(1)] = line
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    args = ap.parse_args()

    text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    if "CAMERA_V96_ARCH" not in text or "CAMERA_V96_AUDIT status=OK" not in text:
        print("V96_DISPLAY FAIL missing=V96_ARCH_or_AUDIT")
        return 2

    lines = latest_camera_lines(text)
    missing = [cid for cid in BASELINE_SOURCE_DISPLAY_P95 if cid not in lines]
    if missing:
        print("V96_DISPLAY FAIL missing_pts=" + ",".join(missing))
        return 2

    current: list[float] = []
    print("V96_DISPLAY per_camera:")
    for cid, baseline in BASELINE_SOURCE_DISPLAY_P95.items():
        now = value(lines[cid], "source_minus_display_p95")
        tracker = value(lines[cid], "source_minus_tracker_p95")
        current.append(now)
        reduction = 100.0 * (1.0 - now / baseline) if baseline > 0 else 0.0
        print(
            f"  {cid} source_display_p95={now:.0f}ms baseline={baseline:.0f}ms "
            f"reduction={reduction:.0f}% source_tracker_p95={tracker:.0f}ms"
        )

    median_ms = statistics.median(current)
    max_ms = max(current)
    avg_reduction = statistics.mean(
        1.0 - current[index] / list(BASELINE_SOURCE_DISPLAY_P95.values())[index]
        for index in range(len(current))
    )

    # Closing the display-backlog issue requires the wall to stay reasonably close
    # to the decoded source on every camera, not merely improve one stream.
    passed = max_ms <= 250.0 and median_ms <= 200.0
    status = "PASS" if passed else "FAIL"
    print(
        f"V96_DISPLAY {status} median_source_display_p95={median_ms:.0f}ms "
        f"max_source_display_p95={max_ms:.0f}ms avg_reduction={avg_reduction * 100.0:.0f}%"
    )
    if passed:
        print("V96_DISPLAY next=display backlog closed; proceed to tracker branch freshness only")
        return 0
    print(
        "V96_DISPLAY next=display backlog remains; keep tracker/detector unchanged and "
        "continue display-only isolation"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
