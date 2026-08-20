from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.flow_assisted_tracker import FlowAssistedPersonTracker
from services.camera_v2.motion_flow_branch import _robust_displacement


def _center(row):
    x1, y1, x2, y2, _conf = row
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _frame(offset_x: int) -> np.ndarray:
    image = np.zeros((144, 256), dtype=np.uint8)
    # Textured body-like patch: edges/corners give LK stable features.
    x1, y1, x2, y2 = 80 + offset_x, 30, 120 + offset_x, 120
    cv2.rectangle(image, (x1, y1), (x2, y2), 120, -1)
    for y in range(y1 + 8, y2 - 4, 12):
        for x in range(x1 + 6, x2 - 4, 10):
            cv2.circle(image, (x, y), 2, 235, -1)
    cv2.line(image, (x1 + 4, y1 + 8), (x2 - 5, y2 - 10), 200, 2)
    return image


def main() -> int:
    tracker = FlowAssistedPersonTracker(2560, 1440)
    cid = "CAM-TEST"
    t0 = 100.0

    # 256x144 motion tile maps 10x to the 2560x1440 source state.
    source_box = (800.0, 300.0, 1200.0, 1200.0)
    tracker.update(cid, t0, [(source_box, 0.88)])
    before = tracker.render(cid, t0)
    assert len(before) == 1, before
    before_cx, before_cy = _center(before[0])

    prev = _frame(0)
    curr = _frame(5)
    flow = _robust_displacement(prev, curr, (80.0, 30.0, 120.0, 120.0))
    assert flow is not None, flow
    dx, dy, quality, good = flow
    assert 3.5 <= dx <= 6.5, (dx, dy, quality, good)
    assert abs(dy) <= 1.5, (dx, dy, quality, good)
    assert quality >= tracker.flow_min_quality, (quality, tracker.flow_min_quality)

    anchors = tracker.anchors(cid, t0)
    assert len(anchors) == 1
    tid = int(anchors[0]["track_id"])
    applied = tracker.apply_flow(
        cid,
        tid,
        dx * 10.0,
        dy * 10.0,
        t0 + 0.05,
        quality,
    )
    assert applied

    after = tracker.render(cid, t0 + 0.05)
    assert len(after) == 1, after
    after_cx, after_cy = _center(after[0])
    assert after_cx > before_cx + 30.0, (before_cx, after_cx)
    assert abs(after_cy - before_cy) < 20.0, (before_cy, after_cy)

    # Fresh optical flow must suppress old open-loop detector velocity. Rendering
    # a few milliseconds later should not add a second large forward jump.
    later = tracker.render(cid, t0 + 0.10)
    later_cx, _ = _center(later[0])
    assert abs(later_cx - after_cx) < 12.0, (after_cx, later_cx)

    print(
        "MOTION_FLOW_TRACKER_TEST=PASS "
        f"dx_tile={dx:.2f} quality={quality:.2f} features={good} "
        f"anchor_move={after_cx-before_cx:.1f}px overshoot_guard=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
