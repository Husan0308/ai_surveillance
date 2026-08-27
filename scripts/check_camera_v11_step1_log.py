#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

CAM = re.compile(
    r"CAMERA_V11_STEP1_CAMERA camera=(\S+) source_fps=([0-9.]+) wall_p95=([0-9.]+)ms "
    r"pts_p95=([0-9.]+)ms mux_stale_p95=([0-9.]+)ms tile_stale_p95=([0-9.]+)ms "
    r"tile_dt_p95=([0-9.]+)ms mux_samples=(\d+) tile_samples=(\d+) input_q=(\d+) input_qmax=(\d+) "
    r"errors=(\d+) warnings=(\d+)"
)
DISPLAY = re.compile(
    r"CAMERA_V11_STEP1_DISPLAY batch_fps=([0-9.]+) render_fps=([0-9.]+) "
    r"batch_size_p50=([0-9.]+) batch_size_p95=([0-9.]+) full_pct=([0-9.]+) "
    r"batch_q=(\d+) batch_qmax=(\d+) render_q=(\d+) render_qmax=(\d+) batches=(\d+)"
)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_camera_v11_step1_log.py /tmp/CAMERA_V11_STEP1.log")
        return 2
    text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    for marker in ("CAMERA_V11_STEP1_ARCH", "CAMERA_V11_STEP1_POLICY", "CAMERA_V11_STEP1_INVARIANT"):
        if marker not in text:
            print(f"V11_STEP1 FAIL missing={marker}")
            return 2

    latest_cam: dict[str, re.Match[str]] = {}
    for match in CAM.finditer(text):
        latest_cam[match.group(1)] = match
    displays = list(DISPLAY.finditer(text))
    if len(latest_cam) != 6 or not displays:
        print(f"V11_STEP1 FAIL cameras={len(latest_cam)} display_samples={len(displays)}")
        return 2

    reasons: list[str] = []
    max_fps = max(float(m.group(2)) for m in latest_cam.values())
    for cid, m in sorted(latest_cam.items()):
        fps = float(m.group(2))
        wall95 = float(m.group(3))
        pts95 = float(m.group(4))
        mux95 = float(m.group(5))
        tile95 = float(m.group(6))
        tiledt95 = float(m.group(7))
        mux_samples = int(m.group(8))
        tile_samples = int(m.group(9))
        qmax = int(m.group(11))
        errors = int(m.group(12))
        ratio = 100.0 * fps / max(0.001, max_fps)
        if fps < 18.0 or ratio < 90.0:
            reasons.append(f"{cid}:source_fps={fps:.1f}/ratio={ratio:.1f}%")
        if wall95 > 170.0:
            reasons.append(f"{cid}:source_wall_p95={wall95:.0f}ms")
        if pts95 > 70.0:
            reasons.append(f"{cid}:source_pts_p95={pts95:.0f}ms")
        if mux95 > 160.0:
            reasons.append(f"{cid}:mux_stale_p95={mux95:.0f}ms")
        if tile95 > 210.0:
            reasons.append(f"{cid}:tile_stale_p95={tile95:.0f}ms")
        if tiledt95 > 210.0:
            reasons.append(f"{cid}:tile_dt_p95={tiledt95:.0f}ms")
        if mux_samples < 100 or tile_samples < 100:
            reasons.append(f"{cid}:insufficient_meta={mux_samples}/{tile_samples}")
        if qmax > 1:
            reasons.append(f"{cid}:input_qmax={qmax}")
        if errors > 0:
            reasons.append(f"{cid}:errors={errors}")
        print(
            "V11_STEP1_CAMERA "
            f"camera={cid} fps={fps:.2f} ratio={ratio:.1f}% wall_p95={wall95:.0f}ms "
            f"mux_stale_p95={mux95:.0f}ms tile_stale_p95={tile95:.0f}ms "
            f"tile_dt_p95={tiledt95:.0f}ms qmax={qmax} meta={mux_samples}/{tile_samples}"
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
        "V11_STEP1_DISPLAY "
        f"batch_fps={batch_fps:.2f} render_fps={render_fps:.2f} size_p50={size50:.0f} "
        f"size_p95={size95:.0f} full_pct={full_pct:.1f}% qmax={batch_qmax}/{render_qmax} batches={batches}"
    )

    if reasons:
        print("V11_STEP1 RESULT diagnosis=FAIL_DISPLAY_FRESHNESS reasons=" + ";".join(reasons))
        print("V11_STEP1 next=do not add detector/tracker; fix only the failing display stage")
        return 1

    print("V11_STEP1 RESULT diagnosis=PASS_DISPLAY_FRESH")
    print("V11_STEP1 next=freeze Step1; then add detector-only Step2 without changing ingest/display")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
