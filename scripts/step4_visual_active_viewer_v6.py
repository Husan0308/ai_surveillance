#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

# This file is launched both as `python scripts/...py` and as a module. When Python
# executes a file path, sys.path[0] is the scripts/ directory, not the repository root,
# so absolute imports such as `from scripts...` fail unless ROOT is added explicitly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication

from scripts.step4_visual_debug_viewer_v6 import (
    ViewerV6,
    build_source_and_tracks,
    parse_args,
)


class ActiveOnlyTracks:
    """Render only confirmed live observations; keep shadow memory internal."""

    def __init__(self, base) -> None:
        self.base = base

    @property
    def last_error(self) -> str:
        return self.base.last_error

    def start(self) -> None:
        self.base.start()

    def stop(self) -> None:
        self.base.stop()

    def join(self, timeout: float = 1.5) -> None:
        self.base.join(timeout)

    def snapshot(self):
        return [
            row
            for row in self.base.snapshot()
            if row.confirmed and not row.predicted and row.state == "tracked"
        ]


def main() -> int:
    args = parse_args()
    source, base_tracks = build_source_and_tracks(args)
    tracks = ActiveOnlyTracks(base_tracks)

    print(
        "STEP4_V6_VISUAL_ACTIVE_ONLY "
        f"camera={args.camera} tracker_log={args.track_log} "
        "render=confirmed-tracked-only shadow_memory=internal shadow_render=0 "
        "body_envelope=1 metadata_lag_comp=1 size_prediction=0",
        flush=True,
    )

    app = QApplication(sys.argv)
    tracks.start()
    source.start()
    viewer = ViewerV6(source, tracks, args.camera)
    viewer.setWindowTitle(f"Step 4 V6 Body Envelope - {args.camera} - ACTIVE ONLY")
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
