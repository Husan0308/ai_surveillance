#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

PREFIX = "CAMERA_V11_BBOX_SINGLE_TARGET "
PUBLISHER_PREFIX = "CAMERA_V11_BBOX_PUBLISHER "


def parse_kv(line: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key] = value
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Check V11 single-target bbox live log")
    parser.add_argument("--log", type=Path, default=Path("/tmp/CAMERA_V11_BBOX_TRACKER.log"))
    parser.add_argument(
        "--required-cameras",
        default="CAM-01,CAM-04",
        help="Comma-separated cameras that must acquire a target; empty means any acquired camera",
    )
    args = parser.parse_args()

    if not args.log.is_file():
        print(f"V11_BBOX_SINGLE_TARGET_CHECK RESULT=FAIL reason=log_missing path={args.log}")
        return 1

    latest: dict[str, dict[str, str]] = {}
    publisher_errors: int | None = None
    for raw in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.startswith(PREFIX):
            row = parse_kv(raw)
            camera = row.get("camera")
            if camera:
                latest[camera] = row
        elif raw.startswith(PUBLISHER_PREFIX):
            row = parse_kv(raw)
            try:
                publisher_errors = int(row.get("errors", "0"))
            except ValueError:
                publisher_errors = 999999

    if not latest:
        print("V11_BBOX_SINGLE_TARGET_CHECK RESULT=FAIL reason=single_target_markers_missing")
        return 1

    required = [item.strip() for item in args.required_cameras.split(",") if item.strip()]
    reasons: list[str] = []
    acquired_total = 0
    suppressed_total = 0
    handoff_total = 0
    max_output = 0
    violations = 0

    for camera, row in sorted(latest.items()):
        try:
            enabled = int(row.get("enabled", "0"))
            acquired = int(row.get("acquired", "0"))
            handoff = int(row.get("handoff", "0"))
            suppressed = int(row.get("suppressed", "0"))
            output_max = int(row.get("output_max", "999"))
            camera_violations = int(row.get("violations", "999"))
        except ValueError:
            reasons.append(f"{camera}:malformed_marker")
            continue
        if enabled != 1:
            reasons.append(f"{camera}:single_target_disabled")
        if output_max > 1:
            reasons.append(f"{camera}:output_max={output_max}")
        if camera_violations != 0:
            reasons.append(f"{camera}:violations={camera_violations}")
        acquired_total += acquired
        handoff_total += handoff
        suppressed_total += suppressed
        max_output = max(max_output, output_max)
        violations += camera_violations

    for camera in required:
        row = latest.get(camera)
        if row is None:
            reasons.append(f"{camera}:marker_missing")
            continue
        try:
            acquired = int(row.get("acquired", "0"))
            handoff = int(row.get("handoff", "0"))
        except ValueError:
            reasons.append(f"{camera}:malformed_acquisition")
            continue
        if acquired + handoff < 1:
            reasons.append(f"{camera}:target_not_acquired")

    if not required and acquired_total < 1:
        reasons.append("no_camera_acquired_target")

    if publisher_errors is None:
        reasons.append("publisher_marker_missing")
    elif publisher_errors != 0:
        reasons.append(f"publisher_errors={publisher_errors}")

    if reasons:
        print(
            "V11_BBOX_SINGLE_TARGET_CHECK RESULT=FAIL "
            f"reasons={';'.join(reasons)} cameras={len(latest)} acquired={acquired_total} "
            f"handoff={handoff_total} suppressed={suppressed_total} max_output={max_output} "
            f"violations={violations}"
        )
        return 1

    required_text = ",".join(required) if required else "any"
    print(
        "V11_BBOX_SINGLE_TARGET_CHECK RESULT=PASS "
        f"required={required_text} cameras={len(latest)} acquired={acquired_total} "
        f"handoff={handoff_total} suppressed={suppressed_total} max_output={max_output} "
        f"violations={violations} publisher_errors={publisher_errors}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
