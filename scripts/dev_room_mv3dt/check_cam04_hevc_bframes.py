#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sample", type=Path)
    ap.add_argument("--min-frames", type=int, default=20)
    args = ap.parse_args()

    if not args.sample.is_file() or args.sample.stat().st_size == 0:
        print(json.dumps({"status": "FAIL", "reason": "missing_or_empty_sample"}))
        return 2
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        print(json.dumps({"status": "FAIL", "reason": "ffprobe_not_found"}))
        return 2

    command = [
        ffprobe,
        "-v", "error",
        "-f", "hevc",
        "-select_streams", "v:0",
        "-show_frames",
        "-show_entries", "frame=pict_type",
        "-of", "json",
        str(args.sample),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(json.dumps({
            "status": "FAIL",
            "reason": "ffprobe_failed",
            "stderr": result.stderr[-1000:],
        }, indent=2))
        return 2

    payload = json.loads(result.stdout or "{}")
    types = [
        str(frame.get("pict_type", "")).strip()
        for frame in payload.get("frames", [])
        if str(frame.get("pict_type", "")).strip()
    ]
    counts = collections.Counter(types)
    report = {
        "status": "PASS",
        "sample": str(args.sample),
        "frames": len(types),
        "frame_types": dict(sorted(counts.items())),
        "b_frames": counts.get("B", 0),
        "low_latency_eligible": counts.get("B", 0) == 0 and len(types) >= args.min_frames,
    }
    if len(types) < args.min_frames:
        report["status"] = "FAIL"
        report["reason"] = "insufficient_frames"
    elif counts.get("B", 0):
        report["status"] = "FAIL"
        report["reason"] = "b_frames_present"

    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
