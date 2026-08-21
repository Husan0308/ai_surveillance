#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.legacy_unknown_overlay import _visible_tracks
from services.camera_v2.old_ui_detection_backend import OldUIBoxManager


def main() -> int:
    manager = OldUIBoxManager(2560, 1440)
    t0 = time.monotonic()

    # High-confidence person needs two consistent observations in the exact
    # Core-v1 birth policy.  The second observation should create local track 1.
    manager.update("CAM-01", t0, [((900.0, 300.0, 1260.0, 1200.0), 0.88)])
    first = _visible_tracks(manager, "CAM-01", t0 + 0.02)
    if first:
        raise RuntimeError(f"track became visible before Core-v1 confirmation: {first}")

    manager.update("CAM-01", t0 + 0.12, [((920.0, 305.0, 1280.0, 1205.0), 0.90)])
    second = _visible_tracks(manager, "CAM-01", t0 + 0.14)
    if len(second) != 1:
        raise RuntimeError(f"expected one confirmed visible track, got {second}")
    track_id = int(second[0][5])
    if track_id != 1:
        raise RuntimeError(f"expected first local track id=1, got {track_id}")

    manager.update("CAM-01", t0 + 0.24, [((950.0, 310.0, 1310.0, 1210.0), 0.91)])
    third = _visible_tracks(manager, "CAM-01", t0 + 0.26)
    if len(third) != 1 or int(third[0][5]) != track_id:
        raise RuntimeError(
            f"local Unknown ID changed across normal motion: before={track_id} after={third}"
        )

    print(
        "LEGACY_UNKNOWN_TRACK_TEST=PASS "
        f"label=Unknown_C1_{track_id:02d} stable_id=1 confirmed_after=2-hits",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
