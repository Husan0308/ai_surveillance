#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_CHECKER = ROOT / "scripts/check_camera_v11_deepstream_yolo_cam01_cam02_v1_log.py"
ARCH_PREFIX = "CAMERA_V11_UI_PREVIEW_ARCH "
STAT_PREFIX = "CAMERA_V11_UI_PREVIEW_STATS "
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")
ALL_BASE_CAMERAS = ",".join(f"CAM-{index:02d}" for index in range(1, 7))


def parse_kv(line: str) -> dict[str, str]:
    return {key: value for key, value in KV_RE.findall(line)}


def number(row: dict[str, str], name: str, fallback: float = -1.0) -> float:
    try:
        return float(row.get(name, fallback))
    except (TypeError, ValueError):
        return fallback


def main() -> int:
    parser = argparse.ArgumentParser(description="Check staged Sentinel UI preview and frozen six-camera base")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--required-ui-cameras", required=True)
    parser.add_argument("--min-runtime-stats", type=int, default=5)
    parser.add_argument("--max-preview-age-ms", type=float, default=1200.0)
    parser.add_argument("--max-queue", type=int, default=1)
    args = parser.parse_args()

    required = tuple(value.strip() for value in args.required_ui_cameras.split(",") if value.strip())
    reasons: list[str] = []
    if not required:
        reasons.append("no_required_ui_cameras")
    if len(set(required)) != len(required):
        reasons.append("duplicate_required_ui_cameras")
    if not args.log.is_file():
        print(f"V11_UI_PREVIEW_CHECK RESULT=FAIL reason=log_missing path={args.log}")
        return 1

    base = subprocess.run(
        [
            sys.executable,
            str(BASE_CHECKER),
            "--log", str(args.log),
            "--required-cameras", ALL_BASE_CAMERAS,
            "--min-runtime-stats", str(args.min_runtime_stats),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if base.returncode != 0:
        reasons.append("frozen_base_checker_failed")

    text = args.log.read_text(encoding="utf-8", errors="replace")
    if "CAMERA_V11_UI_PREVIEW camera=" in text and " warning=" in text:
        reasons.append("preview_warning_marker")
    if "Traceback (most recent call last)" in text or "KeyboardInterrupt" in text:
        reasons.append("runtime_traceback")
    if "Cuda failure:" in text or "nvbufsurface: Error" in text:
        reasons.append("cuda_teardown_error")

    arch_rows = [parse_kv(line) for line in text.splitlines() if line.startswith(ARCH_PREFIX)]
    arch_by_camera = {row.get("camera", ""): row for row in arch_rows}
    if set(arch_by_camera) != set(required):
        reasons.append(
            f"preview_arch_cameras={','.join(sorted(arch_by_camera))}!={','.join(required)}"
        )
    paths: list[str] = []
    preview_hz: list[float] = []
    for cid in required:
        row = arch_by_camera.get(cid, {})
        if row.get("source") != "post-osd-same-pipeline":
            reasons.append(f"{cid}:source={row.get('source', 'missing')}")
        if row.get("rtsp_extra") != "0":
            reasons.append(f"{cid}:rtsp_extra={row.get('rtsp_extra', 'missing')}")
        if row.get("queue") != "latest1":
            reasons.append(f"{cid}:queue_policy={row.get('queue', 'missing')}")
        if row.get("transport") != "raw-bgrx-shm":
            reasons.append(f"{cid}:transport={row.get('transport', 'missing')}")
        path = row.get("path", "")
        if not path:
            reasons.append(f"{cid}:path_missing")
        paths.append(path)
        preview_hz.append(number(row, "fps"))
    if len(paths) != len(set(paths)):
        reasons.append("preview_paths_not_unique")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for line in text.splitlines():
        if line.startswith(STAT_PREFIX):
            row = parse_kv(line)
            grouped[row.get("camera", "")].append(row)

    details: list[str] = []
    for cid in required:
        rows = grouped.get(cid, [])
        if len(rows) < args.min_runtime_stats:
            reasons.append(f"{cid}:preview_stats={len(rows)}<{args.min_runtime_stats}")
            continue
        tail = rows[-min(5, len(rows)):]
        latest = rows[-1]
        exported = int(number(latest, "exported", -1))
        sequence = int(number(latest, "sequence", -1))
        errors = int(number(latest, "errors", 999))
        queue = max(int(number(row, "queue", 999)) for row in tail)
        age_ms = max(number(row, "age_ms", 999999.0) for row in tail)
        thread_alive = int(number(latest, "thread_alive", 0))
        file_exists = int(number(latest, "file_exists", 0))
        width = int(number(latest, "width", 0))
        height = int(number(latest, "height", 0))
        stride = int(number(latest, "stride", 0))
        sequences = [int(number(row, "sequence", -1)) for row in tail]
        if exported <= 0 or sequence <= 0:
            reasons.append(f"{cid}:no_preview_export exported={exported} sequence={sequence}")
        if len(sequences) >= 2 and sequences[-1] <= sequences[0]:
            reasons.append(f"{cid}:sequence_not_increasing={sequences[0]}->{sequences[-1]}")
        if errors != 0:
            reasons.append(f"{cid}:preview_errors={errors}")
        if queue > args.max_queue:
            reasons.append(f"{cid}:preview_queue={queue}>{args.max_queue}")
        if age_ms < 0 or age_ms > args.max_preview_age_ms:
            reasons.append(f"{cid}:preview_age_ms={age_ms:.1f}>{args.max_preview_age_ms:.1f}")
        if thread_alive != 1:
            reasons.append(f"{cid}:preview_thread_alive={thread_alive}")
        if file_exists != 1:
            reasons.append(f"{cid}:preview_file_exists={file_exists}")
        if width <= 0 or height <= 0 or stride < width * 4:
            reasons.append(f"{cid}:invalid_geometry={width}x{height}x{stride}")
        details.append(
            f"{cid}:exported={exported},sequence={sequence},errors={errors},queue={queue},"
            f"age_ms={age_ms:.1f},geometry={width}x{height}x{stride}"
        )

    summary = " | ".join(details)
    hz_text = ",".join(f"{value:.1f}" for value in preview_hz)
    if reasons:
        base_text = (base.stdout or base.stderr).strip().replace("\n", " | ")
        print(
            f"V11_UI_PREVIEW_CHECK RESULT=FAIL reasons={';'.join(reasons)} "
            f"ui_cameras={','.join(required)} preview_hz={hz_text} details={summary} base={base_text}"
        )
        return 1
    print(
        f"V11_UI_PREVIEW_CHECK RESULT=PASS ui_cameras={','.join(required)} "
        f"preview_hz={hz_text} rtsp_sources=6 rtsp_extra=0 detector_workers=1 details={summary}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
