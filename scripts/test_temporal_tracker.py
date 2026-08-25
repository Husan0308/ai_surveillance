from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.temporal_tracker import AnchoredPersonTracker


def _center(row):
    x1, y1, x2, y2, _conf = row
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def main() -> int:
    tracker = AnchoredPersonTracker(2560, 1440)
    cid = "CAM-TEST"
    t0 = 100.0

    # A single plausible detection must NOT immediately become a visible person.
    tracker.update(cid, t0, [((900.0, 260.0, 1200.0, 1040.0), 0.82)])
    assert tracker.render(cid, t0) == []
    assert tracker.anchors(cid, t0) == []
    internal = tracker.tracks[cid]
    assert len(internal) == 1, internal
    tid = next(iter(internal))
    assert internal[tid].confirmed is False

    # A second spatially-consistent observation confirms the same candidate.
    tracker.update(cid, t0 + 0.50, [((1010.0, 275.0, 1310.0, 1050.0), 0.73)])
    anchors1 = tracker.anchors(cid, t0 + 0.50)
    assert len(anchors1) == 1, anchors1
    assert anchors1[0]["track_id"] == tid, (tid, anchors1)
    assert anchors1[0]["confirmed"] is True

    rows = tracker.render(cid, t0 + 0.75)
    assert len(rows) == 1, rows
    cx, _cy = _center(rows[0])
    # Bounded lead: prediction follows movement but must not run far ahead.
    assert 1050.0 < cx < 1450.0, cx

    # Standing -> seated/bent posture: height shrinks and center moves down.
    # The posture-tolerant association must keep the original track id.
    tracker.update(cid, t0 + 1.00, [((1040.0, 560.0, 1450.0, 1110.0), 0.61)])
    anchors2 = tracker.anchors(cid, t0 + 1.00)
    assert len(anchors2) == 1, anchors2
    assert anchors2[0]["track_id"] == tid, (tid, anchors2)

    # A detector miss must not make a confirmed person disappear immediately.
    tracker.update(cid, t0 + 1.50, [])
    held = tracker.render(cid, t0 + 2.80)
    assert len(held) == 1, held
    anchors_held = tracker.anchors(cid, t0 + 2.80)
    assert len(anchors_held) == 1
    assert anchors_held[0]["track_id"] == tid

    # Once the bounded display hold is genuinely exceeded, no stale ghost remains.
    stale = tracker.render(cid, t0 + 4.00)
    assert stale == [], stale

    # A one-off medium-confidence furniture-like candidate never renders.
    false_cid = "CAM-FALSE"
    tracker.update(false_cid, t0, [((300.0, 300.0, 520.0, 520.0), 0.67)])
    assert tracker.render(false_cid, t0 + 0.30) == []

    print(
        "TEMPORAL_TRACKER_TEST=PASS "
        f"track_id={tid} probation=1 posture_same_id=1 miss_hold=1 "
        "bounded_prediction=1 single_false_hidden=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
