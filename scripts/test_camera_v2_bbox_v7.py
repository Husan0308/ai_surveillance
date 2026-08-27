#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.runtime_bbox_v7 import _expand_box, _stable_size


def close(a: float, b: float, eps: float = 1e-6) -> None:
    if abs(float(a) - float(b)) > eps:
        raise AssertionError(f"{a} != {b}")


def test_fast_open() -> None:
    size, expanded_at = _stable_size(
        80.0,
        150.0,
        1.0,
        2.0,
        hold_sec=0.22,
        shrink_alpha=0.42,
    )
    close(size, 150.0)
    close(expanded_at, 2.0)


def test_short_hold_prevents_limb_snap() -> None:
    size, expanded_at = _stable_size(
        150.0,
        90.0,
        2.0,
        2.15,
        hold_sec=0.22,
        shrink_alpha=0.42,
    )
    close(size, 150.0)
    close(expanded_at, 2.0)


def test_slow_close_after_hold() -> None:
    size, _ = _stable_size(
        150.0,
        90.0,
        2.0,
        2.30,
        hold_sec=0.22,
        shrink_alpha=0.42,
    )
    expected = 150.0 + 0.42 * (90.0 - 150.0)
    close(size, expected)
    if size <= 90.0 or size >= 150.0:
        raise AssertionError(f"slow close invalid: {size}")


def test_full_body_margin_contains_tracker_box() -> None:
    raw = (200.0, 100.0, 400.0, 500.0)
    expanded = _expand_box(
        raw,
        1280.0,
        720.0,
        side_margin=0.06,
        top_margin=0.04,
        bottom_margin=0.07,
    )
    if not (
        expanded[0] <= raw[0]
        and expanded[1] <= raw[1]
        and expanded[2] >= raw[2]
        and expanded[3] >= raw[3]
    ):
        raise AssertionError(f"expanded box does not contain raw box: {expanded}")
    close(expanded[0], 188.0)
    close(expanded[1], 84.0)
    close(expanded[2], 412.0)
    close(expanded[3], 528.0)


def test_margin_clamps_at_frame_edges() -> None:
    expanded = _expand_box(
        (2.0, 3.0, 1278.0, 719.0),
        1280.0,
        720.0,
        side_margin=0.06,
        top_margin=0.04,
        bottom_margin=0.07,
    )
    close(expanded[0], 0.0)
    close(expanded[1], 0.0)
    close(expanded[2], 1279.0)
    close(expanded[3], 719.0)


def test_center_is_not_predicted() -> None:
    # The policy functions only alter size/margin; there is intentionally no velocity
    # input. Verify symmetric side expansion keeps the current horizontal center.
    raw = (300.0, 160.0, 500.0, 560.0)
    expanded = _expand_box(
        raw,
        1280.0,
        720.0,
        side_margin=0.06,
        top_margin=0.04,
        bottom_margin=0.07,
    )
    raw_cx = 0.5 * (raw[0] + raw[2])
    exp_cx = 0.5 * (expanded[0] + expanded[2])
    close(raw_cx, exp_cx)
    if not math.isfinite(exp_cx):
        raise AssertionError("invalid center")


def main() -> int:
    tests = [
        test_fast_open,
        test_short_hold_prevents_limb_snap,
        test_slow_close_after_hold,
        test_full_body_margin_contains_tracker_box,
        test_margin_clamps_at_frame_edges,
        test_center_is_not_predicted,
    ]
    for test in tests:
        test()
        print(f"V7_BBOX_TEST PASS {test.__name__}")
    print(f"V7_BBOX_TESTS status=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
