#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

CAM = re.compile(
    r"CAMERA_V11_STEP1V4_CAMERA camera=(\S+) source_fps=([0-9.]+) render_fps=([0-9.]+) "
    r"wall_p95=([0-9.]+)ms pts_p95=([0-9.]+)ms display_age_p95=([0-9.]+)ms "
    r"render_gap_p95=([0-9.]+)ms display_samples=(\d+) pts_match_miss=(\d+) "
    r"input_q=(\d+) input_qmax=(\d+) errors=(\d+) warnings=(\d+)"
)
SCALER = re.compile(
    r"CAMERA_V11_STEP1V8_SCALER camera=CAM-02 interpolation=(\d+) expected=(\d+) "
    r"gpu_scaling=(\d+) single_resize=(\d+)"
)
QUALITY = re.compile(
    r"CAMERA_V11_STEP1V4_QUALITY interpolation=(\d+) gpu_scaling=(\d+) single_resize=(\d+) "
    r"jpeg=(\d+) main_streams=(\d+) tile=(\d+)x(\d+) mux=(\d+) tiler=(\d+) "
    r"independent=(\d+) videooverlay=(\d+)"
)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_camera_v11_step1_v8_log.py /tmp/CAMERA_V11_STEP1V8.log")
        return 2

    text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    for marker in (
        "CAMERA_V11_STEP1V8_PREFLIGHT",
        "CAMERA_V11_STEP1V8_INVARIANT",
        "CAMERA_V11_STEP1V8_AB",
        "CAMERA_V11_STEP1V8_ARCH",
        "CAMERA_V11_STEP1V8_POLICY",
        "CAMERA_V11_STEP1V8_SCALER",
        "CAMERA_V11_STEP1V4_QUALITY",
    ):
        if marker not in text:
            print(f"V11_STEP1V8 FAIL missing={marker}")
            return 2

    latest: dict[str, re.Match[str]] = {}
    for m in CAM.finditer(text):
        latest[m.group(1)] = m
    scalers = list(SCALER.finditer(text))
    qualities = list(QUALITY.finditer(text))

    if len(latest) != 6 or not scalers or not qualities:
        print(
            f"V11_STEP1V8 FAIL cameras={len(latest)} scaler_samples={len(scalers)} "
            f"quality_samples={len(qualities)}"
        )
        return 2

    reasons: list[str] = []
    max_source_fps = max(float(m.group(2)) for m in latest.values())

    for cid, m in sorted(latest.items()):
        source_fps = float(m.group(2))
        render_fps = float(m.group(3))
        wall95 = float(m.group(4))
        pts95 = float(m.group(5))
        display95 = float(m.group(6))
        render_gap95 = float(m.group(7))
        samples = int(m.group(8))
        misses = int(m.group(9))
        qmax = int(m.group(11))
        errors = int(m.group(12))

        source_ratio = 100.0 * source_fps / max(0.001, max_source_fps)
        render_ratio = 100.0 * render_fps / max(0.001, source_fps)
        total = samples + misses
        miss_pct = 100.0 * misses / max(1, total)

        if source_fps < 18.0 or source_ratio < 90.0:
            reasons.append(f"{cid}:source_fps={source_fps:.1f}/ratio={source_ratio:.1f}%")
        if render_fps < 18.0 or render_ratio < 90.0:
            reasons.append(f"{cid}:render_fps={render_fps:.1f}/ratio={render_ratio:.1f}%")
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

        print(
            "V11_STEP1V8_CAMERA "
            f"camera={cid} source_fps={source_fps:.2f} render_fps={render_fps:.2f} "
            f"source_ratio={source_ratio:.1f}% render_ratio={render_ratio:.1f}% "
            f"wall_p95={wall95:.0f}ms display_age_p95={display95:.0f}ms "
            f"render_gap_p95={render_gap95:.0f}ms qmax={qmax} "
            f"samples={samples} match_miss={miss_pct:.1f}%"
        )

    s = scalers[-1]
    interpolation = int(s.group(1))
    expected = int(s.group(2))
    gpu_scaling = int(s.group(3))
    single_resize = int(s.group(4))
    if interpolation != 2 or expected != 2:
        reasons.append(f"cam02_scaler:interpolation={interpolation}/expected={expected}")
    if gpu_scaling != 1 or single_resize != 1:
        reasons.append(f"cam02_scaler:path=gpu{gpu_scaling}/single{single_resize}")

    q = qualities[-1]
    base_interpolation = int(q.group(1))
    base_gpu = int(q.group(2))
    base_single = int(q.group(3))
    jpeg = int(q.group(4))
    main_streams = int(q.group(5))
    tile_w = int(q.group(6))
    tile_h = int(q.group(7))
    mux = int(q.group(8))
    tiler = int(q.group(9))
    independent = int(q.group(10))
    videooverlay = int(q.group(11))

    if base_interpolation != 4:
        reasons.append(f"quality:base_interpolation={base_interpolation}")
    if base_gpu != 1 or base_single != 1 or jpeg != 0:
        reasons.append(f"quality:path=gpu{base_gpu}/single{base_single}/jpeg{jpeg}")
    if main_streams != 1:
        reasons.append("quality:not_all_main_streams")
    if tile_w != 640 or tile_h != 360:
        reasons.append(f"quality:tile={tile_w}x{tile_h}")
    if mux != 0 or tiler != 0 or independent != 1 or videooverlay != 1:
        reasons.append(
            f"arch:mux{mux}/tiler{tiler}/independent{independent}/videooverlay{videooverlay}"
        )

    print(
        "V11_STEP1V8_SCALER "
        f"cam02_interpolation={interpolation} others_interpolation={base_interpolation} "
        f"gpu_scaling={gpu_scaling} single_resize={single_resize}"
    )

    if reasons:
        print("V11_STEP1V8 RESULT diagnosis=FAIL_CAM02_CUBIC_AB reasons=" + ";".join(reasons))
        print("V11_STEP1V8 next=do not add detector/tracker; evaluate CAM-02 scaler cost only")
        return 1

    print("V11_STEP1V8 RESULT diagnosis=PASS_CAM02_CUBIC_DISPLAY")
    print("V11_STEP1V8 next=freeze Step1 scaler policy; then create detector-only Step2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
