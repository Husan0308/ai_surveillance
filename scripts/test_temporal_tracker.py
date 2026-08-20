from __future__ import annotations

from services.camera_v2.temporal_tracker import AnchoredPersonTracker


def _center(row):
    x1, y1, x2, y2, _conf = row
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def main() -> int:
    tracker = AnchoredPersonTracker(2560, 1440)
    cid = "CAM-TEST"
    t0 = 100.0

    # Strong first observation: track is immediately confirmed.
    tracker.update(cid, t0, [((900.0, 260.0, 1200.0, 1040.0), 0.82)])
    anchors0 = tracker.anchors(cid, t0)
    assert len(anchors0) == 1, anchors0
    tid = anchors0[0]["track_id"]
    assert anchors0[0]["confirmed"] is True

    # Walking correction. The same anchor must survive and move right.
    tracker.update(cid, t0 + 0.50, [((1010.0, 275.0, 1310.0, 1050.0), 0.73)])
    anchors1 = tracker.anchors(cid, t0 + 0.50)
    assert len(anchors1) == 1, anchors1
    assert anchors1[0]["track_id"] == tid, (tid, anchors1)
    assert anchors1[0]["cx"] > anchors0[0]["cx"]

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

    # A detector miss must not make the person box disappear immediately.
    tracker.update(cid, t0 + 1.50, [])
    held = tracker.render(cid, t0 + 2.80)
    assert len(held) == 1, held
    anchors_held = tracker.anchors(cid, t0 + 2.80)
    assert len(anchors_held) == 1
    assert anchors_held[0]["track_id"] == tid

    # Once the configured hold window is genuinely exceeded the stale ghost is
    # removed. Long-term survival will be owned by the optical-flow stage.
    stale = tracker.render(cid, t0 + 4.30)
    assert stale == [], stale

    print(
        "TEMPORAL_TRACKER_TEST=PASS "
        f"track_id={tid} posture_same_id=1 miss_hold=1 bounded_prediction=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
