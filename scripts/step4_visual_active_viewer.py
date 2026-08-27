#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication

from services.shared.camera_config import load_settings
from scripts.step4_visual_debug_viewer import MainStreamSource, TrackLogTailer, Viewer


class ActiveOnlyTracks:
    """Presentation-only filter: keep tracker shadow memory internal, do not render it."""

    def __init__(self, base: TrackLogTailer) -> None:
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
    parser = argparse.ArgumentParser(
        description="Show one smooth main stream with confirmed active Step 4 tracks only."
    )
    parser.add_argument("--camera", default="CAM-01")
    parser.add_argument("--track-log", default="/tmp/ML_STEP4_V3_VISUAL.log")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--latency-ms", type=int, default=80)
    args = parser.parse_args()

    settings = load_settings()
    by_id = {camera.camera_id: camera for camera in settings.cameras}
    if args.camera not in by_id:
        raise SystemExit(f"unknown camera {args.camera}; available={','.join(by_id)}")

    base_tracks = TrackLogTailer(Path(args.track_log), args.camera)
    tracks = ActiveOnlyTracks(base_tracks)
    source = MainStreamSource(
        by_id[args.camera],
        max(320, args.width),
        max(180, args.height),
        max(40, args.latency_ms),
    )

    print(
        "STEP4_VISUAL_ACTIVE_ONLY "
        f"camera={args.camera} tracker_log={args.track_log} "
        "render=confirmed-tracked-only shadow_memory=internal shadow_render=0 "
        "detector=unchanged tracker=unchanged camera_service=unchanged",
        flush=True,
    )

    app = QApplication(sys.argv)
    tracks.start()
    source.start()
    viewer = Viewer(source, tracks, args.camera)
    viewer.setWindowTitle(f"Step 4 Visual Acceptance - {args.camera} - ACTIVE ONLY")
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
