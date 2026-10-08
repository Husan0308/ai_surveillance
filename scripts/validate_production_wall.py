"""Opt-in evidence runner for the actual production MainWindow on a real display."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from services.frontend.app.main import MainWindow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--grid-only", action="store_true",
        help="capture the normal six-tile wall without triggering fullscreen",
    )
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    window = MainWindow()
    window.showMaximized()
    fullscreen_checked = False
    grid_captured = False
    with args.output.open("x", buffering=1) as log:
        def sample():
            nonlocal fullscreen_checked, grid_captured
            rows = {c: {"sequence": tile.last_version, "connected": tile.is_connected(),
                        "status": tile.status.text(), "fps": tile.measured_fps,
                        "painted_sequence": tile.video.preview_last_logged_sequence,
                        "current_frame_num": tile.current_source_frame_num,
                        "current_pts_ns": tile.current_pts_ns,
                        "current_image_size": ([tile.current_image.width(), tile.current_image.height()]
                                                if tile.current_image is not None else None),
                        "tracked_objects_for_current_pts": [
                            {"native_track_id": row.get("native_track_id"),
                             "pts_ns": row.get("pts_ns"), "bbox": row.get("bbox"),
                             "bbox_source": row.get("bbox_source"),
                             "tracker_confidence": row.get("tracker_confidence"),
                             "detector_confidence": row.get("detector_confidence")}
                            for row in tile.tracked_objects
                            if tile.current_pts_ns is not None
                            and int(row.get("pts_ns", -1)) == int(tile.current_pts_ns)
                        ],
                        "network_reader": tile.reader is not None}
                    for c, tile in window.camera_wall.tiles.items()}
            log.write(json.dumps({"mono_ns": time.monotonic_ns(), "cameras": rows,
                "api": window.api_status.text(), "ml": window.ml_status.text(),
                "wall": window.camera_count.text(), "dev_room": window.dev_room_status.text(),
                "grid_only": args.grid_only,
                "grid_captured": grid_captured,
                "fullscreen_checked": fullscreen_checked}) + "\n")
            if (not grid_captured and len(rows) == 6
                    and all(r["connected"] for r in rows.values())):
                window.grab().save(str(args.output.with_name("wall.png")))
                grid_captured = True
                if not args.grid_only:
                    tile = window.camera_wall.tiles["CAM-06"]
                    tile._open_fullscreen()
                    QTimer.singleShot(700, tile.fullscreen_dialog.close)
                    fullscreen_checked = True
        timer = QTimer()
        timer.timeout.connect(sample)
        timer.start(500)
        signal.signal(signal.SIGTERM, lambda *_: window.close())
        signal.signal(signal.SIGINT, lambda *_: window.close())
        result = app.exec()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
