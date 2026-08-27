#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

CAM = re.compile(
    r"CAMERA_V11_STEP1V2_CAMERA camera=(\S+) source_fps=([0-9.]+) wall_p95=([0-9.]+)ms "
    r"pts_p95=([0-9.]+)ms mux_age_p95=([0-9.]+)ms display_age_p95=([0-9.]+)ms "
    r"mux_gap_p95=([0-9.]+)ms mux_samples=(\d+) display_samples=(\d+) match_miss=(\d+) "
    r"input_q=(\d+) input_qmax=(\d+) errors=(\d+) warnings=(\d+)"
)
DISPLAY = re.compile(
    r"CAMERA_V11_STEP1V2_DISPLAY batch_fps=([0-9.]+) render_fps=([0-9.]+) "
    r"batch_size_p50=([0-9.]+) batch_size_p95=([0-9.]+) full_pct=([0-9.]+) "
    r"batch_q=(\d+) batch_qmax=(\d+) render_q=(\d+) render_qmax=(\d+) batches=(\d+)"
)
QUALITY = re.compile(
    r"CAMERA_V11_STEP1V2_QUALITY mux_interpolation=(\d+) gpu_scaling=(\d+) single_resize=(\d+) "
    r"jpeg=(\d+) main_streams=(\d+) visible_tile=(\d+)x(\d+)"
)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_camera_v11_step1_v2_log.py /tmp/CAMERA_V11_STEP1V2.log")
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"V11_STEP1V2 FAIL log_missing={path}")
        return 2
    text = path.read_text(encoding="utf-8", errors="replace")

    required = (
        "CAMERA_V11_STEP1V2_ARCH",
        "CAMERA_V11_STEP1V2_POLICY",
        "CAMERA_V11_STEP1V2_QUALITY",
        "CAMERA_V11_STEP1V2_MEASURE",
        "CAMERA_V11_STEP1V2_INVARIANT",
    )
    for marker in required:
        if marker not in text:
            print(f"V11_STEP1V2 FAIL missing={marker}")
            return 2

    latest_cam: dict[str, re.Match[str]] = {}
    for match in CAM.finditer(text):
        latest_cam[match.group(1)] = match
    displays = list(DISPLAY.finditer(text))
    qualities = list(QUALITY.finditer(text))
    if len(latest_cam) != 6 or not displays or not qualities:
        print(
            f"V11_STEP1V2 FAIL cameras={len(latest_cam)} "
            f"display_samples={len(displays)} quality_samples={len(qualities)}"
        )
        return 2

    reasons: list[str] = []
    max_fps = max(float(m.group(2)) for m in latest_cam.values())

    for cid, m in sorted(latest_cam.items()):
        fps = float(m.group(2))
        wall95 = float(m.group(3))
        pts95 = float(m.group(4))
        mux_age95 = float(m.group(5))
        display_age95 = float(m.group(6))
        mux_gap95 = float(m.group(7))
        mux_samples = int(m.group(8))
        display_samples = int(m.group(9))
        misses = int(m.group(10))
        qmax = int(m.group(12))
        errors = int(m.group(13))
        ratio = 100.0 * fps / max(0.001, max_fps)
        provenance_total = mux_samples + misses
        miss_pct = 100.0 * misses / max(1, provenance_total)

        if fps < 18.0 or ratio < 90.0:
            reasons.append(f"{cid}:source_fps={fps:.1f}/ratio={ratio:.1f}%")
        if wall95 > 170.0:
            reasons.append(f"{cid}:source_wall_p95={wall95:.0f}ms")
        if pts95 > 70.0:
            reasons.append(f"{cid}:source_pts_p95={pts95:.0f}ms")
        if mux_age95 > 160.0:
            reasons.append(f"{cid}:mux_age_p95={mux_age95:.0f}ms")
        if display_age95 > 220.0:
            reasons.append(f"{cid}:display_age_p95={display_age95:.0f}ms")
        if mux_gap95 > 180.0:
            reasons.append(f"{cid}:mux_gap_p95={mux_gap95:.0f}ms")
        if mux_samples < 200 or display_samples < 200:
            reasons.append(f"{cid}:insufficient_samples={mux_samples}/{display_samples}")
        if provenance_total >= 100 and miss_pct > 5.0:
            reasons.append(f"{cid}:pts_match_miss={miss_pct:.1f}%")
        if qmax > 1:
            reasons.append(f"{cid}:input_qmax={qmax}")
        if errors > 0:
            reasons.append(f"{cid}:errors={errors}")

        print(
            "V11_STEP1V2_CAMERA "
            f"camera={cid} fps={fps:.2f} ratio={ratio:.1f}% wall_p95={wall95:.0f}ms "
            f"mux_age_p95={mux_age95:.0f}ms display_age_p95={display_age95:.0f}ms "
            f"mux_gap_p95={mux_gap95:.0f}ms qmax={qmax} samples={mux_samples}/{display_samples} "
            f"match_miss={miss_pct:.1f}%"
        )

    d = displays[-1]
    batch_fps = float(d.group(1))
    render_fps = float(d.group(2))
    size50 = float(d.group(3))
    size95 = float(d.group(4))
    full_pct = float(d.group(5))
    batch_qmax = int(d.group(7))
    render_qmax = int(d.group(9))
    batches = int(d.group(10))
    if batch_fps < 18.0:
        reasons.append(f"display:batch_fps={batch_fps:.1f}")
    if render_fps < 18.0:
        reasons.append(f"display:render_fps={render_fps:.1f}")
    if batch_qmax > 1 or render_qmax > 1:
        reasons.append(f"display:qmax={batch_qmax}/{render_qmax}")
    if batches < 200:
        reasons.append(f"display:batches={batches}")

    print(
        "V11_STEP1V2_DISPLAY "
        f"batch_fps={batch_fps:.2f} render_fps={render_fps:.2f} "
        f"size_p50={size50:.0f} size_p95={size95:.0f} full_pct={full_pct:.1f}% "
        f"qmax={batch_qmax}/{render_qmax} batches={batches}"
    )

    q = qualities[-1]
    interpolation = int(q.group(1))
    gpu_scaling = int(q.group(2))
    single_resize = int(q.group(3))
    jpeg = int(q.group(4))
    main_streams = int(q.group(5))
    tile_w = int(q.group(6))
    tile_h = int(q.group(7))
    if interpolation != 4:
        reasons.append(f"quality:interpolation={interpolation}")
    if gpu_scaling != 1 or single_resize != 1 or jpeg != 0:
        reasons.append(
            f"quality:path=gpu{gpu_scaling}/single{single_resize}/jpeg{jpeg}"
        )
    if main_streams != 1:
        reasons.append("quality:not_all_main_streams")

    print(
        "V11_STEP1V2_QUALITY "
        f"interpolation={interpolation} gpu_scaling={gpu_scaling} single_resize={single_resize} "
        f"jpeg={jpeg} main_streams={main_streams} tile={tile_w}x{tile_h}"
    )

    if reasons:
        print("V11_STEP1V2 RESULT diagnosis=FAIL_DISPLAY_V2 reasons=" + ";".join(reasons))
        print("V11_STEP1V2 next=do not add detector/tracker; fix only V2 display runtime")
        return 1

    print("V11_STEP1V2 RESULT diagnosis=PASS_DISPLAY_V2_FRESH_CLEAR")
    print("V11_STEP1V2 next=freeze Step1 V2; then create detector-only Step2 branch without changing ingest/display")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
