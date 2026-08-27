#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from services.ml_service.app.local_tracker_sparse_v2 import MultiCameraSparseRecoveryTracker

W, H = 672, 378
CAM = "CAM-SPARSE"


def box(cx: float, score: float = 0.60):
    w, h = 76.0, 184.0
    cy = 190.0
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score]


def frame_for(rows):
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    for row in rows:
        x1, y1, x2, y2, _score = row
        xa, ya = max(0, int(x1)), max(0, int(y1))
        xb, yb = min(W, int(x2)), min(H, int(y2))
        frame[ya:yb, xa:xb] = (25, 50, 220)
    return frame


def main() -> int:
    tracker = MultiCameraSparseRecoveryTracker(
        [CAM],
        W,
        H,
        low_thresh=0.18,
        high_thresh=0.30,
        new_track_thresh=0.30,
        confirm_hits=2,
        shadow_sec=1.1,
        max_lost_sec=2.5,
        appearance_weight=0.22,
        reacquire_thresh=0.12,
        low_recovery_thresh=0.10,
        low_recovery_sec=1.6,
        duplicate_iou=0.60,
    )

    # Sparse 2 Hz pattern: confirm -> miss -> low-score comeback while lost -> miss ->
    # high-score reacquisition. V1 fragmented this pattern because low-score stage only
    # admitted tracks that were still in state=tracked.
    sequence = [
        (0.0, [box(100, 0.66)]),
        (0.5, [box(145, 0.62)]),
        (1.0, []),
        (1.5, [box(235, 0.24)]),
        (2.0, []),
        (2.5, [box(325, 0.58)]),
        (3.0, [box(370, 0.55)]),
    ]

    confirmed_id = None
    recovered = 0
    created_after_confirm = 0
    for ts, rows in sequence:
        update = tracker.update(CAM, rows, frame_for(rows), int(ts * 1_000_000_000))
        confirmed = [snap.track_id for snap in update.snapshots if snap.confirmed]
        print(
            f"STEP4_V2_TEST t={ts:.1f}s det={len(rows)} active={update.active} "
            f"render={update.renderable} new={update.created} recovered={update.recovered} ids={confirmed}"
        )
        if ts == 0.5:
            if len(confirmed) != 1:
                raise AssertionError(f"expected one confirmed track, got {confirmed}")
            confirmed_id = confirmed[0]
        elif confirmed_id is not None:
            created_after_confirm += update.created
        recovered += update.recovered

    if confirmed_id is None:
        raise AssertionError("track never confirmed")
    final_update = tracker.trackers[CAM]
    live_ids = [
        track.track_id
        for track in final_update._tracks
        if track.status != "removed" and track.hits >= final_update.confirm_hits
    ]
    if live_ids != [confirmed_id]:
        raise AssertionError(f"fragmented IDs: expected only {confirmed_id}, got {live_ids}")
    if created_after_confirm != 0:
        raise AssertionError(f"created {created_after_confirm} new IDs after confirmation")
    if recovered < 2:
        raise AssertionError(f"expected at least two recoveries, got {recovered}")

    print(
        "STEP4_V2_TEST_RESULT status=PASS "
        f"id={confirmed_id} recovered={recovered} created_after_confirm=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
