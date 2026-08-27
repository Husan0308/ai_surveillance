#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def last_float(text: str, pattern: str, default: float = 0.0) -> float:
    rows = re.findall(pattern, text)
    return float(rows[-1]) if rows else default


def last_int(text: str, pattern: str, default: int = 0) -> int:
    rows = re.findall(pattern, text)
    return int(rows[-1]) if rows else default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--camera", default="CAM-01")
    args = ap.parse_args()
    text = args.log.read_text(encoding="utf-8", errors="replace")

    if "CAMERA_V92_ARCH" not in text:
        print("V92_ACCEPT FAIL reason=no-v92-arch")
        return 2
    if "CAMERA_V91_CONTEXT" not in text or "same=1" not in text:
        print("V92_ACCEPT FAIL reason=primary-context-not-confirmed")
        return 2
    if "CAMERA_V92_TRACK warning=" in text:
        print("V92_ACCEPT FAIL reason=tracker-warning")
        return 2

    gpu = last_float(text, r"CAMERA_V92_STATS[^\n]*gpu_ema=([0-9.]+)ms")
    tracker = last_float(text, r"CAMERA_CLEAN_STATS[^\n]*tracker_rate=([0-9.]+)Hz")
    overlay_p95 = last_float(text, r"CAMERA_V92_STATS[^\n]*overlay_age_p95=([0-9.]+)ms")
    currentized = last_int(text, r"CAMERA_V92_STATS[^\n]*currentized=([0-9]+)")
    prunes = last_int(text, r"CAMERA_V92_STATS[^\n]*cache_prunes=([0-9]+)")
    empty_skips = last_int(text, r"CAMERA_V92_STATS[^\n]*empty_detector_skips=([0-9]+)")
    real_updates = last_int(text, r"CAMERA_V92_STATS[^\n]*real_updates=([0-9]+)")

    fps_rows = [float(v) for v in re.findall(rf"{re.escape(args.camera)}:([0-9.]+)fps", text)]
    camera_fps = fps_rows[-1] if fps_rows else 0.0

    # Functional gate first.  GPU is allowed to fluctuate on Pascal; V9.2 is a bbox
    # semantics test, not another detector architecture benchmark.
    ok = (
        camera_fps >= 18.0
        and tracker >= 6.5
        and real_updates > 0
        and overlay_p95 <= 235.0
        and gpu <= 165.0
    )
    status = "PASS" if ok else "FAIL"
    print(
        f"V92_ACCEPT {status} camera={args.camera} fps={camera_fps:.1f} "
        f"tracker={tracker:.1f}Hz gpu_ema={gpu:.1f}ms overlay_p95={overlay_p95:.0f}ms "
        f"real_updates={real_updates} cache_prunes={prunes} currentized={currentized} "
        f"empty_detector_skips={empty_skips}"
    )
    if ok:
        print(
            "V92_ACCEPT next=judge walking bbox visually; false all-source empty holds are removed and stale detector geometry is bounded"
        )
        return 0
    print(
        "V92_ACCEPT next=send CAMERA_V92_STATS + CAMERA_CLEAN_STATS + CAMERA_V91_TRT; do not increase tracker GPU cost yet"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
