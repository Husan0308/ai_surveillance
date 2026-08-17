from __future__ import annotations

import math
import os

os.environ.setdefault("CAMERA_V2_REID_ROOM_MAP", "0:0,3:0,1:1,4:1,2:2,5:2")
os.environ.setdefault("CAMERA_V2_REID_CONFIRM_VOTES", "4")
os.environ.setdefault("CAMERA_V2_ADAPT_BASE_SAME", "0.50")
os.environ.setdefault("CAMERA_V2_ADAPT_MARGIN", "0.045")
os.environ.setdefault("CAMERA_V2_ADAPT_VOTES", "4")
os.environ.setdefault("CAMERA_V2_ADAPT_VOTE_SPAN", "1.20")
os.environ.setdefault("CAMERA_V2_ADAPT_RELEASE_FLOOR", "0.32")
os.environ.setdefault("CAMERA_V2_ADAPT_RELEASE_VOTES", "6")

from services.camera_v2.stable_adaptive_reid import StableAdaptiveTrackletReID
from services.camera_v2.stable_global_reid import StableGlobalReIDManager


def norm(values):
    n = math.sqrt(sum(v * v for v in values))
    return tuple(v / n for v in values)


def blend(a, b, t):
    return norm(tuple((1.0 - t) * x + t * y for x, y in zip(a, b)))


def row(source, oid, feature, x):
    return {
        "source_id": source,
        "object_id": oid,
        "feature": feature,
        "color_feature": (),
        "confidence": 0.90,
        "tracker_confidence": 0.93,
        "bbox": (x, 50.0, x + 70.0, 250.0),
    }


def gid(manager, key):
    return manager._resolve(manager.bindings[key].global_id)


def feed(manager, adaptive, rows, t):
    manager.update_active_tracks(rows, t)
    adaptive.observe_rows(rows, t)
    manager.observe(rows, t)
    adaptive.reconcile(t)


def main() -> int:
    manager = StableGlobalReIDManager()
    adaptive = StableAdaptiveTrackletReID(manager)

    a0 = norm((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    a3 = norm((0.58, math.sqrt(1.0 - 0.58**2), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    b0 = norm((0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    b3 = norm((0.0, 0.0, 0.58, math.sqrt(1.0 - 0.58**2), 0.0, 0.0, 0.0, 0.0))
    jitter = norm((0.0, 0.0, 0.0, 0.0, 1.0, -0.7, 0.2, 0.1))

    # Bootstrap two true peer-camera identities with independent observations.
    for step in range(12):
        t = 100.0 + step * 0.55
        eps = 0.018 + 0.004 * (step % 3)
        rows = [
            row(0, 101, blend(a0, jitter, eps), 30.0),
            row(3, 201, blend(a3, jitter, eps * 0.8), 35.0),
            row(0, 102, blend(b0, jitter, eps * 0.7), 180.0),
            row(3, 202, blend(b3, jitter, eps * 0.9), 185.0),
        ]
        feed(manager, adaptive, rows, t)

    a_gid = gid(manager, (0, 101))
    b_gid = gid(manager, (0, 102))
    merged_ok = a_gid == gid(manager, (3, 201)) and b_gid == gid(manager, (3, 202)) and a_gid != b_gid

    # Add a distractor in CAM-04 that temporarily looks MORE like CAM-01/A than
    # A's already-confirmed peer. A score-only matcher would steal the identity;
    # the peer lease must keep A<->A while both original tracks remain active.
    c3 = norm((0.90, math.sqrt(1.0 - 0.90**2), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    for step in range(10):
        t = 108.0 + step * 0.55
        eps = 0.020 + 0.003 * (step % 2)
        rows = [
            row(0, 101, blend(a0, jitter, eps), 30.0),
            row(3, 201, blend(a3, jitter, eps), 35.0),
            row(0, 102, blend(b0, jitter, eps), 180.0),
            row(3, 202, blend(b3, jitter, eps), 185.0),
            row(3, 203, blend(c3, jitter, eps * 0.6), 300.0),
        ]
        feed(manager, adaptive, rows, t)

    stable_ok = (
        gid(manager, (0, 101)) == a_gid
        and gid(manager, (3, 201)) == a_gid
        and gid(manager, (3, 203)) != a_gid
    )

    snap = adaptive.snapshot()
    ok = merged_ok and stable_ok and snap.get("peer_locks_active", 0) >= 2
    print(
        "stable_adaptive "
        f"A={gid(manager, (0, 101))}/{gid(manager, (3, 201))} "
        f"B={gid(manager, (0, 102))}/{gid(manager, (3, 202))} "
        f"C={gid(manager, (3, 203))} "
        f"merges={snap['adaptive_merges']} locks={snap.get('peer_locks_active', 0)} "
        f"lock_blocks={snap.get('peer_lock_blocks', 0)} fresh_skip={snap.get('fresh_vote_skip', 0)}"
    )
    print("STABLE_ADAPTIVE_REID_PREFLIGHT=" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
