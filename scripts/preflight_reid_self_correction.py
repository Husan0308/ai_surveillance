from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.global_reid import GlobalReIDManager


def _unit(a: float, b: float) -> tuple[float, ...]:
    vector = np.zeros(256, dtype=np.float32)
    vector[0] = a
    vector[1] = b
    vector /= max(1e-8, float(np.linalg.norm(vector)))
    return tuple(float(v) for v in vector)


def _color(index: int) -> tuple[float, ...]:
    vector = np.zeros(96, dtype=np.float32)
    vector[index] = 1.0
    return tuple(float(v) for v in vector)


def _row(source_id: int, object_id: int, feature, color, bbox=(100.0, 100.0, 180.0, 300.0)):
    return {
        "source_id": source_id,
        "object_id": object_id,
        "feature": feature,
        "color_feature": color,
        "bbox": bbox,
        "confidence": 0.90,
        "tracker_confidence": 0.70,
    }


def _labels(manager: GlobalReIDManager) -> dict[tuple[int, int], str]:
    return {(sid, oid): label for sid, oid, label in manager.label_assignments()}


def _feed(manager, row, times):
    for now in times:
        manager.update_active_tracks([row], now=now)
        manager.observe([row], now=now)


def main() -> int:
    manager = GlobalReIDManager()
    person_a = _unit(1.0, 0.0)
    person_b = _unit(0.0, 1.0)
    misleading_b = _unit(0.80, 0.60)
    color_a = _color(3)
    color_b = _color(17)

    # Build two established people first. A owns Global ID 1 in room 1.
    a = _row(2, 101, person_a, color_a)
    _feed(manager, a, (90.0, 90.7, 91.4))
    a_id = _labels(manager)[(2, 101)]

    # B owns a different Global ID from another room.
    b_old = _row(4, 202, person_b, color_b, (360.0, 100.0, 450.0, 320.0))
    _feed(manager, b_old, (92.0, 92.7, 93.4))
    b_id = _labels(manager)[(4, 202)]
    if a_id == b_id:
        raise RuntimeError("setup failed: two different synthetic people share an ID")

    # A becomes active in one peer camera. B appears in the paired camera with an
    # initially misleading appearance that can temporarily look closer to A.
    manager.update_active_tracks([a], now=100.0)
    manager.observe([a], now=100.0)
    b_new_misleading = _row(5, 303, misleading_b, color_b, (400.0, 90.0, 490.0, 315.0))
    for now in (100.1, 100.8, 101.5):
        manager.update_active_tracks([a, b_new_misleading], now=now)
        manager.observe([b_new_misleading], now=now)

    transient = _labels(manager).get((5, 303))
    if transient is None:
        raise RuntimeError("misleading track never received a provisional identity")

    # Later clean embeddings prove that this is person B. The manager must revoke
    # a wrong provisional/current assignment, preserve A's original ID, and recover
    # B's previous ID rather than creating another permanent fragment.
    b_new_clean = _row(5, 303, person_b, color_b, (410.0, 92.0, 500.0, 317.0))
    for now in (102.2, 102.9, 103.6, 104.3, 105.0):
        manager.update_active_tracks([a, b_new_clean], now=now)
        manager.observe([b_new_clean], now=now)

    labels = _labels(manager)
    if labels.get((2, 101)) != a_id:
        raise RuntimeError(f"true owner lost its original ID: expected={a_id} labels={labels}")
    if labels.get((5, 303)) != b_id:
        raise RuntimeError(
            "wrong assignment was not self-corrected to the previously known person: "
            f"expected={b_id} labels={labels}"
        )
    if labels.get((5, 303)) == labels.get((2, 101)):
        raise RuntimeError("cannot-link correction failed: two different active people still share one ID")

    snapshot = manager.snapshot()
    stats = snapshot["stats"]
    if stats.get("corrections", 0) < 1 or stats.get("reassigned", 0) < 1:
        raise RuntimeError(f"correction path was not exercised: stats={stats}")
    if snapshot.get("provisional_bindings", 0) < 0:
        raise RuntimeError("invalid provisional binding count")

    print(
        "REID_SELF_CORRECTION=PASS "
        f"owner={a_id} recovered={b_id} transient={transient} "
        f"corrections={stats.get('corrections')} reassigned={stats.get('reassigned')} "
        f"peer_reject={stats.get('peer_reject')} rollbacks={stats.get('rollbacks')} "
        f"provisional={snapshot.get('provisional_bindings')} confirmed={snapshot.get('confirmed_bindings')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
