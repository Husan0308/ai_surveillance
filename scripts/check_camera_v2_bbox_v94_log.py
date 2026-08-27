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

    m = re.search(
        r"CAMERA_V94_XMAP .*?x_scale=([0-9.]+) y_scale=([0-9.]+)", text
    )
    if not m:
        print("V94_ACCEPT FAIL missing=CAMERA_V94_XMAP")
        return 2
    x_scale = float(m.group(1))
    y_scale = float(m.group(2))

    camera_fps = 0.0
    tracker_rate = 0.0
    for line in text.splitlines():
        if "CAMERA_CLEAN_STATS" not in line:
            continue
        fm = re.search(rf"\b{re.escape(args.camera)}:([0-9.]+)fps", line)
        if fm:
            camera_fps = float(fm.group(1))
        tm = re.search(r"tracker_rate=([0-9.]+)Hz", line)
        if tm:
            tracker_rate = float(tm.group(1))

    gpu_ema = last_float(text, "gpu_ema")
    empty_holds = int(last_float(text, "empty_holds"))
    track_conf_p10 = last_float(text, "track_conf_p10")

    map_ok = abs(x_scale - (512.0 / 672.0)) <= 0.002 and abs(x_scale - y_scale) <= 0.002
    runtime_ok = camera_fps >= 18.0 and tracker_rate >= 8.5 and gpu_ema <= 165.0 and empty_holds == 0
    ok = map_ok and runtime_ok

    print(
        f"V94_ACCEPT {'NUMERIC_PASS' if ok else 'FAIL'} camera={args.camera} "
        f"fps={camera_fps:.1f} tracker={tracker_rate:.1f}Hz gpu_ema={gpu_ema:.1f}ms "
        f"x_scale={x_scale:.6f} y_scale={y_scale:.6f} empty_holds={empty_holds} "
        f"track_conf_p10={track_conf_p10:.3f}"
    )
    print(
        "V94_ACCEPT next=VISUAL_REQUIRED compare right-side bbox alignment and local-ID churn. "
        "Do not tune PTS, detector fairness, NvDCF features, or smoothing until this X-map step is judged."
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
