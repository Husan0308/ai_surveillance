#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

CAM = re.compile(
    r"CAMERA_V11_STEP1V4_CAMERA camera=(\S+) source_fps=([0-9.]+) render_fps=([0-9.]+) "
    r"wall_p95=([0-9.]+)ms pts_p95=([0-9.]+)ms display_age_p95=([0-9.]+)ms "
    r"render_gap_p95=([0-9.]+)ms display_samples=(\d+) pts_match_miss=(\d+) "
    r"input_q=(\d+) input_qmax=(\d+) errors=(\d+) warnings=(\d+)"
)
WINDOW = re.compile(
    r"CAMERA_V11_STEP1V7_WINDOW camera=(\S+) transport=(\S+) low_latency=(\d+) "
    r"xid=(\d+) overlay=(\d+) tile=(\d+)x(\d+)"
)
DECODER = re.compile(
    r"CAMERA_V11_STEP1V7_DECODER camera=(\S+) low_latency=(-?\d+) property=(\S+) "
    r"element=(\S+) expected=(\d+)"
)
QUALITY = re.compile(
    r"CAMERA_V11_STEP1V4_QUALITY interpolation=(\d+) gpu_scaling=(\d+) single_resize=(\d+) "
    r"jpeg=(\d+) main_streams=(\d+) tile=(\d+)x(\d+) mux=(\d+) tiler=(\d+) "
    r"independent=(\d+) videooverlay=(\d+)"
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Long-run aggregate gate for frozen V11 Step1 logs")
    ap.add_argument("log")
    ap.add_argument("--warmup-windows", type=int, default=2)
    ap.add_argument("--min-windows", type=int, default=6)
    ap.add_argument("--min-avg-fps", type=float, default=18.0)
    ap.add_argument("--min-ratio", type=float, default=90.0)
    ap.add_argument("--hard-floor-fps", type=float, default=15.0)
    ap.add_argument("--max-transient-windows", type=int, default=1)
    args = ap.parse_args()

    text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    reasons: list[str] = []
    for marker in (
        "CAMERA_V11_STEP1V7_PREFLIGHT",
        "CAMERA_V11_STEP1V7_INVARIANT",
        "CAMERA_V11_STEP1V7_AB",
        "CAMERA_V11_STEP1V7_ARCH",
        "CAMERA_V11_STEP1V7_POLICY",
        "CAMERA_V11_STEP1V4_QUALITY",
    ):
        if marker not in text:
            reasons.append(f"missing:{marker}")

    rows_by_camera: dict[str, list[re.Match[str]]] = {}
    for match in CAM.finditer(text):
        rows_by_camera.setdefault(match.group(1), []).append(match)
    windows = {m.group(1): m for m in WINDOW.finditer(text)}
    decoders = {m.group(1): m for m in DECODER.finditer(text)}
    qualities = list(QUALITY.finditer(text))

    if len(rows_by_camera) != 6:
        reasons.append(f"cameras={len(rows_by_camera)}")
    if len(windows) != 6:
        reasons.append(f"windows={len(windows)}")
    if len(decoders) != 6:
        reasons.append(f"decoders={len(decoders)}")
    if not qualities:
        reasons.append("quality_samples=0")

    usable: dict[str, list[re.Match[str]]] = {}
    for cid, rows in sorted(rows_by_camera.items()):
        trimmed = rows[max(0, args.warmup_windows) :]
        usable[cid] = trimmed
        if len(trimmed) < args.min_windows:
            reasons.append(f"{cid}:aggregate_windows={len(trimmed)}/min{args.min_windows}")

    avg_source_by_camera: dict[str, float] = {}
    for cid, rows in usable.items():
        if rows:
            avg_source_by_camera[cid] = sum(float(row.group(2)) for row in rows) / len(rows)
    max_avg_source = max(avg_source_by_camera.values(), default=0.0)

    for cid, rows in sorted(usable.items()):
        if not rows:
            continue
        source_values = [float(row.group(2)) for row in rows]
        render_values = [float(row.group(3)) for row in rows]
        avg_source = sum(source_values) / len(source_values)
        avg_render = sum(render_values) / len(render_values)
        source_ratio = 100.0 * avg_source / max(0.001, max_avg_source)
        render_ratio = 100.0 * sum(render_values) / max(0.001, sum(source_values))
        min_source = min(source_values)
        min_render = min(render_values)

        transient_source = sum(1 for value in source_values if value < args.min_avg_fps)
        transient_render = sum(
            1
            for source, render in zip(source_values, render_values)
            if render < args.min_avg_fps or (100.0 * render / max(0.001, source)) < args.min_ratio
        )

        if avg_source < args.min_avg_fps or source_ratio < args.min_ratio:
            reasons.append(f"{cid}:avg_source={avg_source:.2f}/ratio={source_ratio:.1f}%")
        if avg_render < args.min_avg_fps or render_ratio < args.min_ratio:
            reasons.append(f"{cid}:avg_render={avg_render:.2f}/ratio={render_ratio:.1f}%")
        if min_source < args.hard_floor_fps:
            reasons.append(f"{cid}:source_hard_floor={min_source:.2f}")
        if min_render < args.hard_floor_fps:
            reasons.append(f"{cid}:render_hard_floor={min_render:.2f}")
        if transient_source > args.max_transient_windows:
            reasons.append(f"{cid}:low_source_windows={transient_source}/max{args.max_transient_windows}")
        if transient_render > args.max_transient_windows:
            reasons.append(f"{cid}:low_render_windows={transient_render}/max{args.max_transient_windows}")

        latest = rows[-1]
        wall95 = float(latest.group(4))
        pts95 = float(latest.group(5))
        display95 = float(latest.group(6))
        render_gap95 = float(latest.group(7))
        samples = int(latest.group(8))
        misses = int(latest.group(9))
        qmax = int(latest.group(11))
        errors = int(latest.group(12))
        total = samples + misses
        miss_pct = 100.0 * misses / max(1, total)

        if wall95 > 170.0:
            reasons.append(f"{cid}:source_wall_p95={wall95:.0f}ms")
        if pts95 > 70.0:
            reasons.append(f"{cid}:source_pts_p95={pts95:.0f}ms")
        if display95 > 200.0:
            reasons.append(f"{cid}:display_age_p95={display95:.0f}ms")
        if render_gap95 > 170.0:
            reasons.append(f"{cid}:render_gap_p95={render_gap95:.0f}ms")
        if samples < 200:
            reasons.append(f"{cid}:insufficient_display_samples={samples}")
        if total >= 100 and miss_pct > 5.0:
            reasons.append(f"{cid}:pts_match_miss={miss_pct:.1f}%")
        if qmax > 1:
            reasons.append(f"{cid}:input_qmax={qmax}")
        if errors > 0:
            reasons.append(f"{cid}:errors={errors}")

        window = windows.get(cid)
        decoder = decoders.get(cid)
        expected_lowlat = 1 if cid == "CAM-02" else 0
        if window is not None:
            if window.group(2) != "tcp":
                reasons.append(f"{cid}:transport={window.group(2)}")
            if int(window.group(3)) != expected_lowlat:
                reasons.append(f"{cid}:requested_low_latency={window.group(3)}/expected={expected_lowlat}")
            if int(window.group(5)) != 1:
                reasons.append(f"{cid}:overlay={window.group(5)}")
        if decoder is not None:
            if decoder.group(3) != "low-latency-mode":
                reasons.append(f"{cid}:decoder_property={decoder.group(3)}")
            if int(decoder.group(2)) != expected_lowlat or int(decoder.group(5)) != expected_lowlat:
                reasons.append(f"{cid}:decoder_low_latency={decoder.group(2)}/expected={expected_lowlat}")

        print(
            "V11_STEP1V25_AGG_CAMERA "
            f"camera={cid} windows={len(rows)} avg_source_fps={avg_source:.2f} "
            f"avg_render_fps={avg_render:.2f} source_ratio={source_ratio:.1f}% "
            f"render_ratio={render_ratio:.1f}% min_source_fps={min_source:.2f} "
            f"min_render_fps={min_render:.2f} low_source_windows={transient_source} "
            f"low_render_windows={transient_render} wall_p95={wall95:.0f}ms "
            f"display_age_p95={display95:.0f}ms render_gap_p95={render_gap95:.0f}ms "
            f"qmax={qmax} match_miss={miss_pct:.1f}%"
        )

    if qualities:
        q = qualities[-1]
        interpolation = int(q.group(1))
        gpu_scaling = int(q.group(2))
        single_resize = int(q.group(3))
        jpeg = int(q.group(4))
        main_streams = int(q.group(5))
        tile_w = int(q.group(6))
        tile_h = int(q.group(7))
        mux = int(q.group(8))
        tiler = int(q.group(9))
        independent = int(q.group(10))
        videooverlay = int(q.group(11))
        if interpolation != 4:
            reasons.append(f"quality:interpolation={interpolation}")
        if gpu_scaling != 1 or single_resize != 1 or jpeg != 0:
            reasons.append(f"quality:path=gpu{gpu_scaling}/single{single_resize}/jpeg{jpeg}")
        if main_streams != 1:
            reasons.append("quality:not_all_main_streams")
        if tile_w != 640 or tile_h != 360:
            reasons.append(f"quality:tile={tile_w}x{tile_h}")
        if mux != 0 or tiler != 0 or independent != 1 or videooverlay != 1:
            reasons.append(
                f"arch:mux{mux}/tiler{tiler}/independent{independent}/videooverlay{videooverlay}"
            )

    if reasons:
        print("V11_STEP1V25_AGG RESULT=FAIL reasons=" + ";".join(reasons))
        return 1
    print(
        "V11_STEP1V25_AGG RESULT=PASS "
        f"cameras=6 min_avg_fps={args.min_avg_fps:.1f} min_ratio={args.min_ratio:.1f}% "
        f"max_transient_windows={args.max_transient_windows} hard_floor_fps={args.hard_floor_fps:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
