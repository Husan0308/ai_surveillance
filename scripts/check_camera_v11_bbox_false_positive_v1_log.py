#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

TRACKER_PREFIX = "CAMERA_V11_STEP3_V2_TRACKER "
LOCK_PREFIX = "CAMERA_V11_BBOX_SINGLE_TARGET "
POLICY_PREFIX = "CAMERA_V11_BBOX_FP_GUARD_POLICY "
PUBLISHER_PREFIX = "CAMERA_V11_BBOX_PUBLISHER "
CAMERA_RE = re.compile(
    r"(?P<camera>CAM-\d+):updates=(?P<updates>\d+),created=(?P<created>\d+),"
    r"recovered=(?P<recovered>\d+),removed=(?P<removed>\d+),"
    r"visible=(?P<visible>\d+),ids=(?P<ids>[^| ]+)"
)


def parse_kv(line: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key] = value
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check one-person bbox run for source-track churn and display false positives"
    )
    parser.add_argument("--log", type=Path, default=Path("/tmp/CAMERA_V11_BBOX_TRACKER.log"))
    parser.add_argument("--required-cameras", default="CAM-01,CAM-04")
    parser.add_argument("--max-created-per-1000-updates", type=float, default=4.0)
    parser.add_argument("--max-acquire-per-1000-updates", type=float, default=3.0)
    parser.add_argument("--max-visible-per-camera", type=int, default=2)
    parser.add_argument("--max-suppressed-per-update", type=float, default=0.20)
    args = parser.parse_args()

    if not args.log.is_file():
        print(f"V11_BBOX_FALSE_POSITIVE_CHECK RESULT=FAIL reason=log_missing path={args.log}")
        return 1

    latest_tracker: dict[str, dict[str, int | str]] = {}
    max_visible: dict[str, int] = {}
    latest_lock: dict[str, dict[str, str]] = {}
    policy: dict[str, str] | None = None
    publisher_errors: int | None = None

    for raw in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.startswith(TRACKER_PREFIX):
            for match in CAMERA_RE.finditer(raw):
                camera = match.group("camera")
                row: dict[str, int | str] = {
                    "updates": int(match.group("updates")),
                    "created": int(match.group("created")),
                    "recovered": int(match.group("recovered")),
                    "removed": int(match.group("removed")),
                    "visible": int(match.group("visible")),
                    "ids": match.group("ids"),
                }
                latest_tracker[camera] = row
                max_visible[camera] = max(max_visible.get(camera, 0), int(row["visible"]))
        elif raw.startswith(LOCK_PREFIX):
            row = parse_kv(raw)
            camera = row.get("camera")
            if camera:
                latest_lock[camera] = row
        elif raw.startswith(POLICY_PREFIX):
            policy = parse_kv(raw)
        elif raw.startswith(PUBLISHER_PREFIX):
            row = parse_kv(raw)
            try:
                publisher_errors = int(row.get("errors", "0"))
            except ValueError:
                publisher_errors = 999999

    required = [item.strip() for item in args.required_cameras.split(",") if item.strip()]
    reasons: list[str] = []
    summaries: list[str] = []

    if policy is None:
        reasons.append("fp_guard_policy_missing")
    else:
        try:
            new_track_conf = float(policy.get("new_track_conf", "0"))
            confirm_hits = int(policy.get("confirm_hits", "0"))
            max_lost = float(policy.get("max_lost", "0").rstrip("s"))
        except ValueError:
            new_track_conf = 0.0
            confirm_hits = 0
            max_lost = 0.0
        if new_track_conf < 0.45:
            reasons.append(f"new_track_conf_too_low={new_track_conf:.2f}")
        if confirm_hits < 3:
            reasons.append(f"confirm_hits_too_low={confirm_hits}")
        if max_lost < 4.0:
            reasons.append(f"max_lost_too_short={max_lost:.2f}")

    for camera in required:
        tracker = latest_tracker.get(camera)
        lock = latest_lock.get(camera)
        if tracker is None:
            reasons.append(f"{camera}:tracker_marker_missing")
            continue
        if lock is None:
            reasons.append(f"{camera}:lock_marker_missing")
            continue

        updates = int(tracker["updates"])
        created = int(tracker["created"])
        peak_visible = max_visible.get(camera, int(tracker["visible"]))
        try:
            acquired = int(lock.get("acquired", "0"))
            handoff = int(lock.get("handoff", "0"))
            released = int(lock.get("released", "0"))
            suppressed = int(lock.get("suppressed", "0"))
            output_max = int(lock.get("output_max", "999"))
            violations = int(lock.get("violations", "999"))
        except ValueError:
            reasons.append(f"{camera}:malformed_lock_marker")
            continue

        denom = max(1, updates)
        created_per_1000 = 1000.0 * created / denom
        acquire_per_1000 = 1000.0 * acquired / denom
        suppression_ratio = suppressed / denom
        summaries.append(
            f"{camera}:updates={updates},created={created},created_per_1000={created_per_1000:.2f},"
            f"peak_visible={peak_visible},suppressed={suppressed},suppressed_per_update={suppression_ratio:.3f},"
            f"acquired={acquired},acquire_per_1000={acquire_per_1000:.2f},handoff={handoff},released={released}"
        )

        if acquired + handoff < 1:
            reasons.append(f"{camera}:target_not_acquired")
        if created_per_1000 > args.max_created_per_1000_updates:
            reasons.append(
                f"{camera}:created_per_1000={created_per_1000:.2f}>{args.max_created_per_1000_updates:.2f}"
            )
        if acquire_per_1000 > args.max_acquire_per_1000_updates:
            reasons.append(
                f"{camera}:acquire_per_1000={acquire_per_1000:.2f}>{args.max_acquire_per_1000_updates:.2f}"
            )
        if peak_visible > args.max_visible_per_camera:
            reasons.append(f"{camera}:peak_visible={peak_visible}>{args.max_visible_per_camera}")
        if suppression_ratio > args.max_suppressed_per_update:
            reasons.append(
                f"{camera}:suppressed_per_update={suppression_ratio:.3f}>{args.max_suppressed_per_update:.3f}"
            )
        if output_max > 1:
            reasons.append(f"{camera}:output_max={output_max}")
        if violations != 0:
            reasons.append(f"{camera}:violations={violations}")

    if publisher_errors is None:
        reasons.append("publisher_marker_missing")
    elif publisher_errors != 0:
        reasons.append(f"publisher_errors={publisher_errors}")

    summary_text = " | ".join(summaries) if summaries else "none"
    if reasons:
        print(
            "V11_BBOX_FALSE_POSITIVE_CHECK RESULT=FAIL "
            f"reasons={';'.join(reasons)} required={','.join(required) or 'any'} "
            f"details={summary_text}"
        )
        return 1

    print(
        "V11_BBOX_FALSE_POSITIVE_CHECK RESULT=PASS "
        f"required={','.join(required) or 'any'} "
        f"max_created_per_1000={args.max_created_per_1000_updates:.2f} "
        f"max_acquire_per_1000={args.max_acquire_per_1000_updates:.2f} "
        f"max_visible={args.max_visible_per_camera} "
        f"max_suppressed_per_update={args.max_suppressed_per_update:.3f} "
        f"details={summary_text} publisher_errors={publisher_errors}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
