#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.visibility_policy_v72 import should_hold_last_good


def main() -> int:
    cases = [
        (10.000, 10.100, 300.0, True),
        (10.000, 10.299, 300.0, True),
        (10.000, 10.301, 300.0, False),
        (10.000, 11.000, 300.0, False),
    ]
    for updated, now, hold_ms, expected in cases:
        actual = should_hold_last_good(updated, now, hold_ms)
        if actual is not expected:
            raise AssertionError((updated, now, hold_ms, actual, expected))
    print(f"V72_VISIBILITY_TESTS status=PASS count={len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
