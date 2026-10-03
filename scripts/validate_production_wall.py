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
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    window = MainWindow()
    window.showMaximized()
    fullscreen_checked = False
    with args.output.open("x", buffering=1) as log:
        def sample():
            nonlocal fullscreen_checked
            rows = {c: {"sequence": tile.last_version, "connected": tile.is_connected(),
                        "status": tile.status.text(), "fps": tile.measured_fps,
                        "painted_sequence": tile.video.preview_last_logged_sequence,
                        "network_reader": tile.reader is not None}
                    for c, tile in window.camera_wall.tiles.items()}
            log.write(json.dumps({"mono_ns": time.monotonic_ns(), "cameras": rows,
                "api": window.api_status.text(), "ml": window.ml_status.text(),
                "wall": window.camera_count.text(), "dev_room": window.dev_room_status.text(),
                "fullscreen_checked": fullscreen_checked}) + "\n")
            if not fullscreen_checked and len(rows) == 6 and all(r["connected"] for r in rows.values()):
                window.grab().save(str(args.output.with_name("wall.png")))
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
