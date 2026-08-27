#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

# When Python executes `scripts/test_v94_xmap.py` directly, sys.path[0] is the
# scripts/ directory, not the repository root.  Add the repo root explicitly so
# the top-level `services` package resolves regardless of the caller's PYTHONPATH.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.runtime_v94_xmap import PascalXMapRuntime


def close(a: float, b: float, tol: float = 0.02) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> int:
    runtime = PascalXMapRuntime.__new__(PascalXMapRuntime)
    runtime.track_width = 512
    runtime.track_height = 288

    # A representative right-side detector box.  Before V9.4 this became
    # x=430..511 in tracker space; the correct mapping is ~327.62..472.38.
    rows = [((430.0, 100.0, 620.0, 200.0), 0.90)]
    mapped = runtime._map_detector_rows(rows)
    assert len(mapped) == 1, mapped
    x1, y1, x2, y2, conf = mapped[0]

    expected_x1 = 430.0 * 512.0 / 672.0
    expected_x2 = 620.0 * 512.0 / 672.0
    expected_y1 = (100.0 - 3.0) * 288.0 / 378.0
    expected_y2 = (200.0 - 3.0) * 288.0 / 378.0

    assert close(x1, expected_x1), (x1, expected_x1)
    assert close(x2, expected_x2), (x2, expected_x2)
    assert close(y1, expected_y1), (y1, expected_y1)
    assert close(y2, expected_y2), (y2, expected_y2)
    assert close(conf, 0.90), conf

    x_scale = 512.0 / 672.0
    y_scale = 288.0 / 378.0
    assert close(x_scale, y_scale, 1e-6), (x_scale, y_scale)

    print(
        "V94_XMAP_TEST PASS "
        f"x={x1:.2f}..{x2:.2f} y={y1:.2f}..{y2:.2f} "
        f"x_scale={x_scale:.6f} y_scale={y_scale:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
