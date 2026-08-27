#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from services.ml_service.app.local_tracker import Detection
from services.ml_service.app.local_tracker_sparse_v6 import (
    BodyEnvelopeObservationRecoveryTracker,
)


APP = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def det(box, score=0.85, app=APP):
    return Detection(np.array(box, dtype=np.float64), float(score), app.copy())


def one_snapshot(update):
    assert len(update.snapshots) == 1, update
    return update.snapshots[0]


def main() -> int:
    tracker = BodyEnvelopeObservationRecoveryTracker(
        "CAM-TEST",
        672,
        378,
        low_thresh=0.08,
        high_thresh=0.25,
        new_track_thresh=0.30,
        confirm_hits=2,
        shadow_sec=1.1,
        max_lost_sec=5.0,
        low_recovery_sec=1.8,
        lost_low_jump_diag=1.05,
        lost_high_jump_diag=1.35,
        render_anchor_alpha=0.72,
        render_size_alpha=0.20,
        render_recovery_size_alpha=0.34,
        render_max_size_step=0.28,
        render_velocity_gain=0.30,
        render_expand_alpha=0.82,
        render_contract_alpha=0.14,
        render_expand_max_step=0.72,
        render_contract_max_step=0.22,
        envelope_pad_x=0.07,
        envelope_pad_top=0.04,
        envelope_pad_bottom=0.03,
        envelope_compact_extra_x=0.06,
    )

    t0 = 1000.0
    base = [250, 80, 330, 320]
    out = tracker.update([det(base)], t0)
    assert out.created == 1
    out = tracker.update([det(base)], t0 + 0.5)
    snap = one_snapshot(out)
    stable_id = snap.track_id
    base_width = snap.bbox_xyxy[2] - snap.bbox_xyxy[0]

    arm = [185, 45, 395, 320]
    out = tracker.update([det(arm, 0.78)], t0 + 1.0)
    snap = one_snapshot(out)
    assert snap.track_id == stable_id
    x1, y1, x2, y2 = snap.bbox_xyxy
    assert x1 <= arm[0] and y1 <= arm[1] and x2 >= arm[2] and y2 >= arm[3], (
        snap.bbox_xyxy,
        arm,
    )
    assert (x2 - x1) > base_width * 1.6
    print("STEP4_V6_ARM_ENVELOPE status=PASS", flush=True)

    compact = [262, 105, 322, 245]
    out = tracker.update([det(compact, 0.42)], t0 + 1.5)
    snap = one_snapshot(out)
    width_after_compact = snap.bbox_xyxy[2] - snap.bbox_xyxy[0]
    assert width_after_compact > base_width * 1.35, width_after_compact
    print("STEP4_V6_SLOW_CONTRACT status=PASS", flush=True)

    guard = BodyEnvelopeObservationRecoveryTracker(
        "CAM-GUARD",
        672,
        378,
        low_thresh=0.08,
        high_thresh=0.25,
        new_track_thresh=0.30,
        confirm_hits=2,
        shadow_sec=1.1,
        max_lost_sec=5.0,
        low_recovery_sec=1.8,
        lost_low_jump_diag=1.05,
        lost_high_jump_diag=1.35,
    )
    origin = [350, 100, 400, 250]
    out = guard.update([det(origin, 0.8)], t0)
    out = guard.update([det(origin, 0.8)], t0 + 0.5)
    stable = one_snapshot(out).track_id

    out = guard.update([], t0 + 1.0)
    far = [500, 215, 550, 365]
    out = guard.update([det(far, 0.09)], t0 + 1.5)
    ids = [s.track_id for s in out.snapshots]
    assert stable in ids, ids
    snap = next(s for s in out.snapshots if s.track_id == stable)
    cx = 0.5 * (snap.bbox_xyxy[0] + snap.bbox_xyxy[2])
    assert cx < 450.0, snap.bbox_xyxy
    assert snap.predicted, snap
    print("STEP4_V6_NO_TELEPORT status=PASS", flush=True)

    print("STEP4_V6_TEST_RESULT status=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
