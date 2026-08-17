from __future__ import annotations

import math
import os

os.environ.setdefault("CAMERA_V2_REID_ROOM_MAP", "0:0,3:0,1:1,4:1,2:2,5:2")
os.environ.setdefault("CAMERA_V2_REID_PEER_MIN_REID", "0.36")
os.environ.setdefault("CAMERA_V2_REID_PEER_CONFIRM_REID", "0.42")
os.environ.setdefault("CAMERA_V2_REID_SAME_ROOM", "0.54")
os.environ.setdefault("CAMERA_V2_REID_COVISIBLE", "0.52")
os.environ.setdefault("CAMERA_V2_REID_CONFIRM_VOTES", "4")

from services.camera_v2.adaptive_reid import AdaptiveTrackletReID
from services.camera_v2.global_reid import GlobalReIDManager


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
        "confidence": 0.88,
        "tracker_confidence": 0.92,
        "bbox": (x, 50.0, x + 70.0, 250.0),
    }


def main() -> int:
    manager = GlobalReIDManager()
    adaptive = AdaptiveTrackletReID(manager)

    # Two people, each observed by both cameras in room 0. Cross-view cosine is
    # intentionally only ~0.52: weak for a one-frame match but stable across a
    # tracklet. Wrong-person similarities remain near zero.
    a0 = norm((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    a3 = norm((0.52, math.sqrt(1.0 - 0.52**2), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    b0 = norm((0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    b3 = norm((0.0, 0.0, 0.52, math.sqrt(1.0 - 0.52**2), 0.0, 0.0, 0.0, 0.0))
    jitter = norm((0.0, 0.0, 0.0, 0.0, 1.0, -0.7, 0.2, 0.1))

    for step in range(10):
        t = 100.0 + step * 0.50
        eps = 0.025 + 0.004 * (step % 3)
        rows = [
            row(0, 101, blend(a0, jitter, eps), 30.0),
            row(0, 102, blend(b0, jitter, eps * 0.8), 180.0),
            row(3, 201, blend(a3, jitter, eps * 0.7), 35.0),
            row(3, 202, blend(b3, jitter, eps * 0.9), 185.0),
        ]
        manager.update_active_tracks(rows, t)
        adaptive.observe_rows(rows, t)
        manager.observe(rows, t)
        adaptive.reconcile(t)

    def gid(key):
        binding = manager.bindings[key]
        return manager._resolve(binding.global_id)

    ga0, ga3 = gid((0, 101)), gid((3, 201))
    gb0, gb3 = gid((0, 102)), gid((3, 202))
    snap = adaptive.snapshot()

    ok = ga0 == ga3 and gb0 == gb3 and ga0 != gb0
    print(f"A={ga0}/{ga3} B={gb0}/{gb3}")
    print(
        "adaptive "
        f"merges={snap['adaptive_merges']} score={snap['last_pair_score']:.3f} "
        f"margin={snap['last_pair_margin']:.3f} threshold={snap['last_threshold']:.3f} "
        f"dup_skip={snap['duplicate_skip']} samples={snap['bank_samples']}"
    )
    print("ADAPTIVE_REID_PREFLIGHT=" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
