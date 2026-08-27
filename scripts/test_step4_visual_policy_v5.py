#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.ml_service.app.visual_box_policy import (
    bounded_center_prediction,
    visual_track_is_fresh,
)


def main() -> int:
    box = (0.40, 0.20, 0.60, 0.80)
    velocity = (3.0, -3.0, 8.0, -8.0)
    out = bounded_center_prediction(box, velocity, 1.0)

    w0 = box[2] - box[0]
    h0 = box[3] - box[1]
    w1 = out[2] - out[0]
    h1 = out[3] - out[1]
    assert math.isclose(w0, w1, abs_tol=1e-9), (box, out)
    assert math.isclose(h0, h1, abs_tol=1e-9), (box, out)

    cx0 = 0.5 * (box[0] + box[2])
    cy0 = 0.5 * (box[1] + box[3])
    cx1 = 0.5 * (out[0] + out[2])
    cy1 = 0.5 * (out[1] + out[3])
    assert abs(cx1 - cx0) <= 0.20 * w0 + 1e-9
    assert abs(cy1 - cy0) <= 0.12 * h0 + 1e-9

    assert visual_track_is_fresh(0.0)
    assert visual_track_is_fresh(1.20)
    assert not visual_track_is_fresh(1.21)
    assert not visual_track_is_fresh(-0.01)

    edge = bounded_center_prediction((0.90, 0.90, 1.00, 1.00), (2.0, 2.0, 0.0, 0.0), 0.2)
    assert all(0.0 <= value <= 1.0 for value in edge), edge
    assert math.isclose(edge[2] - edge[0], 0.10, abs_tol=1e-9), edge
    assert math.isclose(edge[3] - edge[1], 0.10, abs_tol=1e-9), edge

    print("STEP4_V5_VISUAL_POLICY_TEST status=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
