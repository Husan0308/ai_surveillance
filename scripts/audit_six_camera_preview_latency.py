#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

CAMERAS = tuple(f"CAM-{i:02d}" for i in range(1, 7))
P95_LIMIT_MS = 40.0
MIN_SAMPLES = 200


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def stats(values: list[float]) -> dict:
    return {
        "samples": len(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values) if values else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    args = ap.parse_args()

    rows: dict[str, list[dict]] = defaultdict(list)
    malformed = 0
    for line in args.log.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        camera = str(row.get("camera_id", ""))
        if camera in CAMERAS:
            rows[camera].append(row)

    report = {
        "status": "PASS",
        "limit_p95_ms": P95_LIMIT_MS,
        "min_samples_per_camera": MIN_SAMPLES,
        "malformed_lines": malformed,
        "cameras": {},
        "failures": [],
    }

    for camera in CAMERAS:
        e2e = []
        decoder = []
        publish = []
        ui_receive = []
        paint = []
        skipped = 0
        invalid_timing = 0
        for row in rows.get(camera, []):
            t0 = int(row.get("t0_decoder_reference_monotonic_ns") or 0)
            t1 = int(row.get("t1_decoder_out_monotonic_ns") or 0)
            t6 = int(row.get("t6_ipc_publish_monotonic_ns") or 0)
            t7 = int(row.get("t7_ui_receive_monotonic_ns") or 0)
            t8 = int(row.get("t8_ui_paint_monotonic_ns") or 0)
            skipped += int(row.get("ui_skipped_preview_frames") or 0)
            if not (0 < t0 <= t1 <= t6 <= t7 <= t8):
                invalid_timing += 1
                continue
            e2e.append((t8 - t0) / 1e6)
            decoder.append((t1 - t0) / 1e6)
            publish.append((t6 - t1) / 1e6)
            ui_receive.append((t7 - t6) / 1e6)
            paint.append((t8 - t7) / 1e6)

        camera_report = {
            "decoder_reference_to_ui_paint": stats(e2e),
            "decoder_stage": stats(decoder),
            "decoder_out_to_ipc_publish": stats(publish),
            "ipc_publish_to_ui_receive": stats(ui_receive),
            "ui_receive_to_paint": stats(paint),
            "ui_skipped_preview_frames": skipped,
            "invalid_timing_rows": invalid_timing,
        }
        report["cameras"][camera] = camera_report

        if len(e2e) < MIN_SAMPLES:
            report["failures"].append(f"{camera}: insufficient timing samples {len(e2e)}<{MIN_SAMPLES}")
        elif percentile(e2e, 0.95) >= P95_LIMIT_MS:
            report["failures"].append(
                f"{camera}: decoder-reference->UI-paint P95 {percentile(e2e, 0.95):.3f} ms >= {P95_LIMIT_MS:.1f} ms"
            )
        if invalid_timing:
            report["failures"].append(f"{camera}: invalid timing rows={invalid_timing}")

    if malformed:
        report["failures"].append(f"malformed JSONL rows={malformed}")
    if report["failures"]:
        report["status"] = "FAIL"

    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
