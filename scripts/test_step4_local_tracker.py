#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from services.ml_service.app.local_tracker import MultiCameraLocalTracker

W, H = 672, 378
CAM = "CAM-TEST"


def box(cx: float, cy: float = 190.0, w: float = 70.0, h: float = 180.0, score: float = 0.55):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score]


def frame_for(rows):
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    for label, row in rows:
        x1, y1, x2, y2, _score = row
        xa, ya = max(0, int(x1)), max(0, int(y1))
        xb, yb = min(W, int(x2)), min(H, int(y2))
        if label == "A":
            frame[ya:yb, xa:xb] = (20, 40, 220)  # BGR: red-ish clothing
        else:
            frame[ya:yb, xa:xb] = (220, 50, 20)  # BGR: blue-ish clothing
    return frame


def main() -> int:
    tracker = MultiCameraLocalTracker(
        [CAM],
        W,
        H,
        low_thresh=0.18,
        high_thresh=0.30,
        new_track_thresh=0.30,
        confirm_hits=2,
        shadow_sec=0.9,
        max_lost_sec=2.5,
        appearance_weight=0.18,
    )

    sequence = [
        (0.0, [("A", box(90)), ("B", box(560))]),
        (0.5, [("A", box(150)), ("B", box(500))]),
        (1.0, [("A", box(210, score=0.24)), ("B", box(440))]),  # ByteTrack low stage
        (1.5, [("A", box(280)), ("B", box(360))]),
        (2.0, [("B", box(295))]),  # A missed once; shadow/prediction should survive
        (2.5, [("A", box(410)), ("B", box(225))]),  # recovered after crossing
        (3.0, [("A", box(470)), ("B", box(165))]),
    ]

    first_ids = None
    recovered_seen = False
    last_update = None
    for ts, labeled in sequence:
        rows = [row for _label, row in labeled]
        update = tracker.update(CAM, rows, frame_for(labeled), int(ts * 1_000_000_000))
        last_update = update
        ids = [snap.track_id for snap in update.snapshots if snap.confirmed]
        print(
            f"STEP4_TRACK_TEST t={ts:.1f}s det={len(rows)} active={update.active} "
            f"render={update.renderable} new={update.created} recovered={update.recovered} ids={ids}"
        )
        if ts == 0.5:
            if len(ids) != 2:
                raise AssertionError(f"expected 2 confirmed IDs at t=0.5, got {ids}")
            first_ids = tuple(ids)
        if update.recovered:
            recovered_seen = True

    if first_ids is None or last_update is None:
        raise AssertionError("tracker test did not initialize")
    final = {snap.track_id: snap for snap in last_update.snapshots if snap.confirmed}
    if set(final) != set(first_ids):
        raise AssertionError(f"IDs fragmented: initial={first_ids} final={tuple(final)}")
    if not recovered_seen:
        raise AssertionError("missed person was not recovered")

    # T00001 started on the left (A) and must end on the right after crossing.
    a = final[f"{CAM}-T00001"].bbox_xyxy
    b = final[f"{CAM}-T00002"].bbox_xyxy
    acx = 0.5 * (a[0] + a[2])
    bcx = 0.5 * (b[0] + b[2])
    if acx <= bcx:
        raise AssertionError(f"identity switched after crossing: A_cx={acx:.1f} B_cx={bcx:.1f}")

    print(
        "STEP4_TRACK_TEST_RESULT status=PASS "
        f"A={first_ids[0]} B={first_ids[1]} recovered=1 final_A_cx={acx:.1f} final_B_cx={bcx:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
