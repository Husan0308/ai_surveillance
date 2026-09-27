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
    ap.add_argument(
        "--cameras", nargs="+", choices=CAMERAS, default=list(CAMERAS),
        help="Cameras under test; defaults to the complete six-camera gate",
    )
    ap.add_argument("--min-duration-seconds", type=float, default=0.0)
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
        if camera in args.cameras:
            rows[camera].append(row)

    report = {
        "status": "PASS",
        "limit_p95_ms": P95_LIMIT_MS,
        "min_samples_per_camera": MIN_SAMPLES,
        "malformed_lines": malformed,
        "cameras": {},
        "failures": [],
    }

    for camera in args.cameras:
        e2e = []
        decoder = []
        publish = []
        ui_receive = []
        paint = []
        pts_dts_delta = []
        decoder_dts = []
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
            pts = int(row.get("pts_ns") or 0)
            dts = int(row.get("decoder_dts_ns") or 0)
            if pts > 0 and dts > 0:
                pts_dts_delta.append((pts - dts) / 1e6)
                decoder_dts.append(dts)

        dts_regressions = sum(
            current < previous for previous, current in zip(decoder_dts, decoder_dts[1:])
        )
        reorder_frame_period_ms = 50.0
        max_positive_pts_dts_ms = max((value for value in pts_dts_delta if value > 0), default=0.0)

        camera_report = {
            "decoder_reference_to_ui_paint": stats(e2e),
            "decoder_stage": stats(decoder),
            "decoder_out_to_ipc_publish": stats(publish),
            "ipc_publish_to_ui_receive": stats(ui_receive),
            "ui_receive_to_paint": stats(paint),
            "ui_skipped_preview_frames": skipped,
            "invalid_timing_rows": invalid_timing,
            "measurement_duration_seconds": (
                (max(int(row["t8_ui_paint_monotonic_ns"]) for row in rows.get(camera, []))
                 - min(int(row["t8_ui_paint_monotonic_ns"]) for row in rows.get(camera, []))) / 1e9
                if len(e2e) > 1 else 0.0
            ),
            "displayed_fps": (
                (len(e2e) - 1) / ((max(int(row["t8_ui_paint_monotonic_ns"]) for row in rows.get(camera, []))
                                  - min(int(row["t8_ui_paint_monotonic_ns"]) for row in rows.get(camera, []))) / 1e9)
                if len(e2e) > 1 and rows.get(camera) else 0.0
            ),
            "pts_minus_dts_ms": {
                "samples": len(pts_dts_delta),
                "nonzero_samples": sum(value != 0 for value in pts_dts_delta),
                "min": min(pts_dts_delta) if pts_dts_delta else None,
                "p50": percentile(pts_dts_delta, 0.50) if pts_dts_delta else None,
                "p95": percentile(pts_dts_delta, 0.95) if pts_dts_delta else None,
                "max": max(pts_dts_delta) if pts_dts_delta else None,
                "dts_regressions_in_present_order": dts_regressions,
                "estimated_max_reorder_frames_at_20fps": int(math.ceil(
                    max_positive_pts_dts_ms / reorder_frame_period_ms
                )),
            },
        }
        report["cameras"][camera] = camera_report

        if camera_report["measurement_duration_seconds"] < args.min_duration_seconds:
            report["failures"].append(
                f"{camera}: measurement duration {camera_report['measurement_duration_seconds']:.3f}s"
                f" < {args.min_duration_seconds:.3f}s"
            )

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
