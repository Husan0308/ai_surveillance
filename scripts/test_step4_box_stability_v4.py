#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from services.ml_service.app.local_tracker import Detection
from services.ml_service.app.local_tracker_sparse_v4 import BoxStableObservationRecoveryTracker


def det(box, score=0.85, app=None):
    if app is None:
        app = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return Detection(np.array(box, dtype=np.float64), float(score), app.copy())


def width(snapshot):
    x1, _y1, x2, _y2 = snapshot.bbox_xyxy
    return x2 - x1


def height(snapshot):
    _x1, y1, _x2, y2 = snapshot.bbox_xyxy
    return y2 - y1


def main() -> int:
    tracker = BoxStableObservationRecoveryTracker(
        "CAM-TEST",
        672,
        378,
        confirm_hits=2,
        max_lost_sec=5.0,
        shadow_sec=1.1,
        nested_duplicate_ios=0.82,
        nested_duplicate_app_floor=0.58,
        render_anchor_alpha=0.72,
        render_size_alpha=0.20,
        render_max_size_step=0.28,
    )

    t0 = 1000.0
    base = [240, 90, 330, 320]
    out = tracker.update([det(base)], t0)
    assert out.created == 1
    out = tracker.update([det(base)], t0 + 0.5)
    assert len(out.snapshots) == 1 and out.snapshots[0].confirmed
    stable_id = out.snapshots[0].track_id
    w0 = width(out.snapshots[0])
    h0 = height(out.snapshots[0])

    # Simulate the detector producing the normal full-person box plus a nested upper-body
    # box after a raised arm / partial-body change. V3 IoU-only veto can miss this case.
    nested = [252, 92, 322, 210]
    out = tracker.update([det(base, 0.82), det(nested, 0.66)], t0 + 1.0)
    assert out.created == 0, f"nested duplicate minted new ID: {out}"
    ids = [row.track_id for row in out.snapshots]
    assert ids == [stable_id], ids
    print("STEP4_V4_NESTED_DUPLICATE status=PASS", flush=True)

    # Arm-up geometry: raw detector box suddenly becomes much wider and taller while the
    # person is still in the same place. Published box should move smoothly, not breathe.
    arm_up = [195, 42, 370, 320]
    out = tracker.update([det(arm_up, 0.78)], t0 + 1.5)
    assert len(out.snapshots) == 1
    row = out.snapshots[0]
    assert row.track_id == stable_id
    wr = width(row) / max(1e-6, w0)
    hr = height(row) / max(1e-6, h0)
    assert wr < 1.12, f"render width jumped too much: ratio={wr:.3f}"
    assert hr < 1.12, f"render height jumped too much: ratio={hr:.3f}"
    print(f"STEP4_V4_ARM_BOX status=PASS width_ratio={wr:.3f} height_ratio={hr:.3f}", flush=True)

    # Several consistent larger observations must still let the render box catch up; the
    # smoother must not freeze a real person approaching the camera.
    for i in range(1, 6):
        out = tracker.update([det(arm_up, 0.80)], t0 + 1.5 + 0.5 * i)
    row = out.snapshots[0]
    assert width(row) > w0 * 1.20
    print("STEP4_V4_CATCHUP status=PASS", flush=True)
    print("STEP4_V4_TEST_RESULT status=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
