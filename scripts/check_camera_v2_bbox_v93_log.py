#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def last_float(text: str, key: str, default: float = 0.0) -> float:
    matches = re.findall(rf"\b{re.escape(key)}=([0-9.]+)", text)
    return float(matches[-1]) if matches else default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--camera", default="CAM-01")
    args = ap.parse_args()
    text = Path(args.log).read_text(encoding="utf-8", errors="replace")

    if "CAMERA_V93_ARCH" not in text:
        print("V93_ACCEPT FAIL missing=CAMERA_V93_ARCH")
        return 2
    if "CAMERA_V91_CONTEXT" not in text or "same=1" not in text:
        print("V93_ACCEPT FAIL primary_same=0")
        return 2

    camera_fps = 0.0
    tracker_rate = 0.0
    for line in text.splitlines():
        if "CAMERA_CLEAN_STATS" not in line:
            continue
        m = re.search(rf"\b{re.escape(args.camera)}:([0-9.]+)fps", line)
        if m:
            camera_fps = float(m.group(1))
        m = re.search(r"tracker_rate=([0-9.]+)Hz", line)
        if m:
            tracker_rate = float(m.group(1))

    age_p50 = last_float(text, "age_p50")
    age_p95 = last_float(text, "age_p95")
    shift_p50 = last_float(text, "shift_p50")
    shift_p95 = last_float(text, "shift_p95")
    gpu_ema = last_float(text, "gpu_ema")
    projected = int(last_float(text, "projected"))
    empty_holds = int(last_float(text, "empty_holds"))

    ok = (
        camera_fps >= 18.0
        and tracker_rate >= 8.5
        and projected > 0
        and empty_holds == 0
        and gpu_ema <= 165.0
        and age_p95 <= 210.0
        and shift_p95 <= 140.0
    )
    status = "PASS" if ok else "FAIL"
    print(
        f"V93_ACCEPT {status} camera={args.camera} fps={camera_fps:.1f} "
        f"tracker={tracker_rate:.1f}Hz gpu_ema={gpu_ema:.1f}ms "
        f"age_p50={age_p50:.0f}ms age_p95={age_p95:.0f}ms "
        f"shift_p50={shift_p50:.1f}px shift_p95={shift_p95:.1f}px "
        f"projected={projected} empty_holds={empty_holds}"
    )
    print(
        "V93_ACCEPT next=judge walking/arms/bending visually; if projected shift is stable "
        "and boxes no longer trail, keep V9.3. If overshoot appears, reduce DISPLAY_COMP_GAIN."
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
