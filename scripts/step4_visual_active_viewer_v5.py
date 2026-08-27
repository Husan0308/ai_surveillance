#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication

from scripts.step4_visual_debug_viewer_v5 import (
    FreshTrackLogTailer,
    QualityMainStreamSource,
    ViewerV5,
)
from services.shared.camera_config import load_settings


class ActiveOnlyTracks:
    """Render only confirmed live tracks; keep shadow recovery internal."""

    def __init__(self, base: FreshTrackLogTailer) -> None:
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
        description="Sharp 720p main stream with confirmed Step 4 V5 tracks only."
    )
    parser.add_argument("--camera", default="CAM-01")
    parser.add_argument("--track-log", default="/tmp/ML_STEP4_V5_VISUAL.log")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--latency-ms", type=int, default=120)
    args = parser.parse_args()

    settings = load_settings()
    by_id = {camera.camera_id: camera for camera in settings.cameras}
    if args.camera not in by_id:
        raise SystemExit(f"unknown camera {args.camera}; available={','.join(by_id)}")

    base_tracks = FreshTrackLogTailer(Path(args.track_log), args.camera)
    tracks = ActiveOnlyTracks(base_tracks)
    source = QualityMainStreamSource(
        by_id[args.camera],
        max(640, args.width),
        max(360, args.height),
        max(80, args.latency_ms),
    )

    print(
        "STEP4_V5_VISUAL_ACTIVE_ONLY "
        f"camera={args.camera} tracker_log={args.track_log} "
        f"main={args.width}x{args.height} latency={args.latency_ms}ms "
        "render=confirmed-tracked-only shadow_memory=internal shadow_render=0 "
        "predict=center-only-bounded size_prediction=0 stale_box_guard=1",
        flush=True,
    )

    app = QApplication(sys.argv)
    tracks.start()
    source.start()
    viewer = ViewerV5(source, tracks, args.camera)
    viewer.setWindowTitle(f"Step 4 V5 Visual Acceptance - {args.camera} - ACTIVE ONLY")
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
