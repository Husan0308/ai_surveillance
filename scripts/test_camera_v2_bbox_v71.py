#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.visibility_policy_v71 import should_hold_last_good


def test_hold_inside_window() -> None:
    if not should_hold_last_good(10.0, 10.319, 320.0):
        raise AssertionError("last-good box should survive one short empty gap")


def test_expire_outside_window() -> None:
    if should_hold_last_good(10.0, 10.321, 320.0):
        raise AssertionError("last-good box must expire after the bounded hold")


def test_zero_hold_is_strict() -> None:
    if should_hold_last_good(10.0, 10.001, 0.0):
        raise AssertionError("zero hold must not preserve a stale box")


def test_exact_timestamp_is_visible() -> None:
    if not should_hold_last_good(10.0, 10.0, 320.0):
        raise AssertionError("current last-good box must be visible")


def main() -> int:
    tests = [
        test_hold_inside_window,
        test_expire_outside_window,
        test_zero_hold_is_strict,
        test_exact_timestamp_is_visible,
    ]
    for test in tests:
        test()
        print(f"V71_VISIBILITY_TEST PASS {test.__name__}")
    print(f"V71_VISIBILITY_TESTS status=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
