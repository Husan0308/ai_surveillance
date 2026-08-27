#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def last_camera_line(text: str, camera: str) -> str:
    rows = [
        line
        for line in text.splitlines()
        if "CAMERA_V95_PTS" in line and f"camera={camera}" in line
    ]
    return rows[-1] if rows else ""


def value(line: str, key: str, default: float = 0.0) -> float:
    match = re.search(rf"\b{re.escape(key)}=(-?[0-9.]+)", line)
    return float(match.group(1)) if match else default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--camera", default="CAM-01")
    args = ap.parse_args()

    text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    if "CAMERA_V95_ARCH" not in text or "CAMERA_V94_XMAP" not in text:
        print("V95_PTS FAIL missing=V95_ARCH_or_V94_XMAP")
        return 2

    line = last_camera_line(text, args.camera)
    if not line:
        print(f"V95_PTS FAIL camera={args.camera} missing=CAMERA_V95_PTS")
        return 2

    tracker_hz = value(line, "tracker_pts_hz")
    tracker_dt_p95 = value(line, "tracker_dt_p95")
    d_t_p50 = value(line, "display_minus_tracker_p50")
    d_t_p95 = value(line, "display_minus_tracker_p95")
    s_d_p95 = value(line, "source_minus_display_p95")
    s_t_p95 = value(line, "source_minus_tracker_p95")
    det_p50 = value(line, "detector_to_inject_p50")
    det_p95 = value(line, "detector_to_inject_p95")

    if s_d_p95 > 110.0:
        diagnosis = "DISPLAY_BRANCH_BACKLOG"
    elif d_t_p95 > 140.0 or s_t_p95 > 180.0 or tracker_dt_p95 > 155.0:
        diagnosis = "TRACKER_FRAME_STALENESS"
    elif det_p95 > 260.0:
        diagnosis = "DETECTOR_CORRECTION_STALE"
    else:
        diagnosis = "PTS_PATH_HEALTHY"

    print(
        f"V95_PTS RESULT camera={args.camera} diagnosis={diagnosis} "
        f"tracker_pts_hz={tracker_hz:.2f} tracker_dt_p95={tracker_dt_p95:.0f}ms "
        f"display_tracker_p50={d_t_p50:.0f}ms display_tracker_p95={d_t_p95:.0f}ms "
        f"source_display_p95={s_d_p95:.0f}ms source_tracker_p95={s_t_p95:.0f}ms "
        f"detector_inject_p50={det_p50:.0f}ms detector_inject_p95={det_p95:.0f}ms"
    )

    if diagnosis == "DISPLAY_BRANCH_BACKLOG":
        print("V95_PTS next=fix display branch queue/mux timing only; do not tune NvDCF yet")
    elif diagnosis == "TRACKER_FRAME_STALENESS":
        print("V95_PTS next=fix per-camera tracker mux/admission timing only; keep detector/X-map unchanged")
    elif diagnosis == "DETECTOR_CORRECTION_STALE":
        print("V95_PTS next=fix detector capture/injection timing only; keep tracker quality unchanged")
    else:
        print("V95_PTS next=PTS is not the main visual lag; move to NvDCF target lifecycle/feature quality")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
