from __future__ import annotations

import math
import os

os.environ.setdefault("CAMERA_V2_REID_ROOM_MAP", "0:0,3:0,1:1,4:1,2:2,5:2")
os.environ.setdefault("CAMERA_V2_REID_CONFIRM_VOTES", "4")
os.environ.setdefault("CAMERA_V2_ADAPT_BASE_SAME", "0.50")
os.environ.setdefault("CAMERA_V2_ADAPT_MARGIN", "0.045")
os.environ.setdefault("CAMERA_V2_ADAPT_VOTES", "4")
os.environ.setdefault("CAMERA_V2_ADAPT_VOTE_SPAN", "1.0")
os.environ.setdefault("CAMERA_V2_POS_MIN_SAMPLES", "4")
os.environ.setdefault("CAMERA_V2_POS_LINK_VOTES", "3")
os.environ.setdefault("CAMERA_V2_POS_LINK_MIN_REID", "0.52")
os.environ.setdefault("CAMERA_V2_POS_MATCH_BOOST", "0.11")

from services.camera_v2.position_aware_reid import PositionAwareAdaptiveTrackletReID
from services.camera_v2.stable_global_reid import StableGlobalReIDManager


def norm(values):
    n = math.sqrt(sum(v * v for v in values))
    return tuple(v / n for v in values)


def cross_view(base_axis: int, sim: float, dim: int = 8):
    row = [0.0] * dim
    row[base_axis] = sim
    row[base_axis + 1] = math.sqrt(max(0.0, 1.0 - sim * sim))
    return norm(row)


def row(source, oid, feature, cx, cy):
    w, h = 72.0, 170.0
    x1 = cx - w * 0.5
    y2 = cy
    return {
        "source_id": source,
        "object_id": oid,
        "feature": feature,
        "color_feature": (),
        "confidence": 0.90,
        "tracker_confidence": 0.93,
        "bbox": (x1, y2 - h, x1 + w, y2),
    }


def feed(manager, adaptive, rows, t):
    manager.update_active_tracks(rows, t)
    adaptive.observe_rows(rows, t)
    manager.observe(rows, t)
    adaptive.reconcile(t)


def gid(manager, key):
    return manager._resolve(manager.bindings[key].global_id)


def main() -> int:
    manager = StableGlobalReIDManager()
    adaptive = PositionAwareAdaptiveTrackletReID(manager, frame_width=640, frame_height=360)

    # Two fixed seats in CAM-01 and their peer views in CAM-04. Initial occupants
    # have strong enough appearance to teach the geometry safely.
    a0 = norm((1, 0, 0, 0, 0, 0, 0, 0))
    a3 = cross_view(0, 0.62)
    b0 = norm((0, 0, 1, 0, 0, 0, 0, 0))
    b3 = cross_view(2, 0.64)

    for step in range(16):
        t = 100.0 + step * 0.45
        jitter = (step % 3 - 1) * 1.0
        rows = [
            row(0, 101, a0, 160 + jitter, 300 + jitter * 0.2),
            row(3, 201, a3, 470 - jitter, 294 - jitter * 0.2),
            row(0, 102, b0, 360 - jitter, 305),
            row(3, 202, b3, 260 + jitter, 302),
        ]
        feed(manager, adaptive, rows, t)

    snap1 = adaptive.snapshot()
    taught = snap1["seat_links"] >= 2 and snap1["peer_locks_active"] >= 2

    # Much later the NvDCF local IDs restart at the same physical seats. Cross-view
    # appearance is deliberately only 0.43 (below the normal 0.50 merge threshold).
    # Learned geometry should add +0.11 only to the mapped seat and allow a stable
    # merge without lowering the global ReID threshold for everybody else.
    weak0 = norm((0, 0, 0, 0, 1, 0, 0, 0))
    weak3 = cross_view(4, 0.43)
    t0 = 150.0
    for step in range(14):
        t = t0 + step * 0.45
        jitter = (step % 2) * 0.8
        rows = [
            row(0, 111, weak0, 160 + jitter, 300),
            row(3, 211, weak3, 470 - jitter, 294),
        ]
        feed(manager, adaptive, rows, t)

    relation = adaptive._position_relation((0, 111), (3, 211), t)
    raw = adaptive._raw_tracklet_similarity((0, 111), (3, 211), t)
    boosted = adaptive.tracklet_similarity((0, 111), (3, 211), t)
    merged = gid(manager, (0, 111)) == gid(manager, (3, 211))

    snap2 = adaptive.snapshot()
    ok = taught and relation == "match" and boosted >= raw + 0.09 and merged
    print(
        "position_aware "
        f"seat_links={snap2['seat_links']} anchors={snap2['seat_anchors']} "
        f"relation={relation} raw={raw:.3f} boosted={boosted:.3f} "
        f"new_gid={gid(manager, (0, 111))}/{gid(manager, (3, 211))} "
        f"pos_match={snap2['position_matches']} pos_veto={snap2['position_vetoes']}"
    )
    print("POSITION_AWARE_REID_PREFLIGHT=" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
