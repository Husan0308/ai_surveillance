#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from services.ml_service.app.local_tracker_sparse_v3 import (
    MultiCameraObservationRecoveryTracker,
)

W, H = 672, 378
RED = (25, 50, 220)
BLUE = (220, 170, 25)


def box(cx: float, score: float = 0.60, *, w: float = 76.0, h: float = 184.0):
    cy = 190.0
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score]


def frame_for(items):
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    for row, color in items:
        x1, y1, x2, y2, _score = row
        xa, ya = max(0, int(x1)), max(0, int(y1))
        xb, yb = min(W, int(x2)), min(H, int(y2))
        frame[ya:yb, xa:xb] = color
    return frame


def update(tracker, cam: str, ts: float, items):
    rows = [row for row, _color in items]
    return tracker.update(cam, rows, frame_for(items), int(ts * 1_000_000_000))


def make_tracker(cam: str):
    return MultiCameraObservationRecoveryTracker(
        [cam],
        W,
        H,
        low_thresh=0.18,
        high_thresh=0.30,
        new_track_thresh=0.30,
        confirm_hits=2,
        tentative_ttl_sec=0.9,
        shadow_sec=1.1,
        max_lost_sec=5.0,
        appearance_weight=0.22,
        reacquire_thresh=0.12,
        low_recovery_thresh=0.10,
        low_recovery_sec=3.0,
        duplicate_iou=0.60,
        low_appearance_weight=0.16,
        low_appearance_floor=0.45,
        live_duplicate_iou=0.72,
        lost_velocity_half_life_sec=0.9,
    )


def confirmed_ids(update_result):
    return [snap.track_id for snap in update_result.snapshots if snap.confirmed]


def test_long_gap_recovery() -> None:
    cam = "CAM-LONG"
    tracker = make_tracker(cam)

    first = update(tracker, cam, 0.0, [(box(120, 0.68), RED)])
    second = update(tracker, cam, 0.5, [(box(130, 0.64), RED)])
    ids = confirmed_ids(second)
    if len(ids) != 1:
        raise AssertionError(f"long-gap track did not confirm: {ids}")
    original_id = ids[0]

    # Seven missed 2 Hz measurements: 3.5 s without a detector box. V2's 2.5 s
    # lifetime would expire this identity. V3 must keep it dormant but not render a
    # ghost after the short 1.1 s shadow window.
    last_miss = None
    for ts in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
        last_miss = update(tracker, cam, ts, [])
    assert last_miss is not None
    if last_miss.renderable != 0:
        raise AssertionError(
            f"dormant long-gap identity should not render, render={last_miss.renderable}"
        )

    comeback = update(tracker, cam, 4.0, [(box(150, 0.61), RED)])
    ids_after = confirmed_ids(comeback)
    if ids_after != [original_id]:
        raise AssertionError(
            f"long-gap fragmentation: expected {original_id}, got {ids_after}"
        )
    if comeback.created != 0 or comeback.recovered < 1:
        raise AssertionError(
            "long-gap reappearance must recover existing ID: "
            f"new={comeback.created} recovered={comeback.recovered}"
        )

    print(
        "STEP4_V3_LONG_GAP status=PASS "
        f"id={original_id} recovered={comeback.recovered} ghost_render=0"
    )


def test_low_score_hijack_guard() -> None:
    cam = "CAM-HIJACK"
    tracker = make_tracker(cam)

    a0 = box(260, 0.72)
    b0 = box(330, 0.70)
    update(tracker, cam, 0.0, [(a0, RED), (b0, BLUE)])
    confirmed = update(
        tracker,
        cam,
        0.5,
        [(box(260, 0.71), RED), (box(330, 0.69), BLUE)],
    )
    ids = confirmed_ids(confirmed)
    if len(ids) != 2:
        raise AssertionError(f"expected two confirmed IDs, got {ids}")
    original_ids = set(ids)

    # Person B is missed. Detector emits a low-score duplicate of person A. Geometry
    # alone is close enough that a sparse low-stage matcher can incorrectly drag B onto
    # A. V3's weak appearance veto must leave B lost instead of hijacking its identity.
    ambiguous = update(
        tracker,
        cam,
        1.0,
        [(box(260, 0.74), RED), (box(260, 0.22), RED)],
    )
    if ambiguous.matched_low != 0:
        raise AssertionError(
            f"low-score duplicate hijacked another identity: matched_low={ambiguous.matched_low}"
        )
    if ambiguous.created != 0:
        raise AssertionError(f"low-score duplicate created ID: new={ambiguous.created}")

    restored = update(
        tracker,
        cam,
        1.5,
        [(box(262, 0.73), RED), (box(330, 0.68), BLUE)],
    )
    ids_after = set(confirmed_ids(restored))
    if ids_after != original_ids:
        raise AssertionError(
            f"two-person identity fragmentation: expected {original_ids}, got {ids_after}"
        )
    if restored.created != 0 or restored.recovered < 1:
        raise AssertionError(
            "missed person should recover old ID: "
            f"new={restored.created} recovered={restored.recovered}"
        )

    print(
        "STEP4_V3_HIJACK_GUARD status=PASS "
        f"ids={sorted(original_ids)} recovered={restored.recovered} low_hijack=0"
    )


def test_low_never_creates() -> None:
    cam = "CAM-LOW"
    tracker = make_tracker(cam)
    result = update(tracker, cam, 0.0, [(box(200, 0.22), RED)])
    if result.created != 0 or result.active != 0:
        raise AssertionError(
            f"low-confidence detection minted identity: new={result.created} active={result.active}"
        )
    print("STEP4_V3_LOW_CREATE status=PASS new=0")


def main() -> int:
    test_long_gap_recovery()
    test_low_score_hijack_guard()
    test_low_never_creates()
    print("STEP4_V3_TEST_RESULT status=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
