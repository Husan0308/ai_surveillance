from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.stable_visual_adapter import StableVisualFlowBoxManager


def _center(row):
    x1, y1, x2, y2, _conf = row
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def main() -> int:
    tracker = StableVisualFlowBoxManager(2560, 1440)
    cid = "CAM-TEST"
    t0 = 100.0

    first = ((900.0, 260.0, 1200.0, 1040.0), 0.72)
    second = ((930.0, 270.0, 1230.0, 1050.0), 0.70)

    # Exact old stable behavior: one high-confidence observation is only a birth
    # candidate; the second spatially-consistent observation confirms the track.
    tracker.update(cid, t0, [first])
    assert tracker.render(cid, t0 + 0.02) == []

    tracker.update(cid, t0 + 0.20, [second])
    visible = tracker.render(cid, t0 + 0.22)
    assert len(visible) == 1, visible
    before_x, before_y = _center(visible[0])

    regions = tracker.flow_regions(cid, t0 + 0.22)
    assert len(regions) == 1, regions
    tid = int(regions[0]["track_id"])

    # Measured motion moves the already-confirmed track; it cannot create one.
    assert tracker.apply_flow(cid, tid, 45.0, 3.0, t0 + 0.25, 0.80)
    moved = tracker.render(cid, t0 + 0.26)
    assert len(moved) == 1, moved
    after_x, after_y = _center(moved[0])
    assert after_x > before_x + 20.0, (before_x, after_x)
    assert abs(after_y - before_y) < 20.0, (before_y, after_y)

    # A short detector miss does not immediately extinguish the confirmed person.
    tracker.update(cid, t0 + 0.40, [])
    held = tracker.render(cid, t0 + 1.00)
    assert len(held) == 1, held

    # A track must still die after the hard detector-age bound. Optical flow may
    # bridge sparse corrections but can never turn into indefinite ghost memory.
    gone = tracker.render(cid, t0 + 4.60)
    assert gone == [], gone

    print(
        "STABLE_VISUAL_ADAPTER_TEST=PASS "
        f"track_id={tid} birth_hits=2 flow_move={after_x-before_x:.1f}px "
        "short_miss_hold=1 hard_expiry=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
