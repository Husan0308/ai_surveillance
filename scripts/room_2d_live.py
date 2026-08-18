#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.ml_service.core_v1.room_projection import RoomProjection


class RoomMap(QLabel):
    def __init__(self, map_path: Path, projections: dict[str, RoomProjection], backend: str):
        super().__init__()
        self.map_path = map_path
        self.projections = projections
        self.backend = backend.rstrip("/")
        self.base = QPixmap(str(map_path))
        if self.base.isNull():
            raise RuntimeError(f"room map ochilmadi: {map_path}")
        self.points = []
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#02050a;")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_tracks)
        self.timer.start(100)
        self.render()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.render()

    def refresh_tracks(self):
        try:
            with urllib.request.urlopen(f"{self.backend}/tracks", timeout=0.7) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return

        points = []
        cameras = payload.get("cameras") or {}
        for camera_id, projection in self.projections.items():
            rows = (cameras.get(camera_id) or {}).get("tracks") or []
            for row in rows:
                bbox = row.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                mapped = projection.project_bbox_footpoint(bbox)
                if mapped is None or not projection.inside_map(mapped, margin=25):
                    continue
                points.append(
                    {
                        "camera_id": camera_id,
                        "display_id": row.get("display_id") or str(row.get("track_id", "")),
                        "point": mapped,
                    }
                )
        self.points = points
        self.render()

    def render(self):
        if self.width() < 2 or self.height() < 2:
            return
        scaled = self.base.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        canvas = QPixmap(self.size())
        canvas.fill(Qt.black)
        x0 = (self.width() - scaled.width()) // 2
        y0 = (self.height() - scaled.height()) // 2

        painter = QPainter(canvas)
        painter.drawPixmap(x0, y0, scaled)
        sx = scaled.width() / max(1, self.base.width())
        sy = scaled.height() / max(1, self.base.height())

        for item in self.points:
            x, y = item["point"]
            dx = x0 + x * sx
            dy = y0 + y * sy
            pen = QPen(Qt.red if item["camera_id"] == "CAM-01" else Qt.cyan)
            pen.setWidth(3)
            painter.setPen(pen)
            painter.setBrush(pen.color())
            painter.drawEllipse(int(dx - 7), int(dy - 7), 14, 14)
            painter.drawText(int(dx + 11), int(dy - 9), f'{item["camera_id"]} {item["display_id"]}')
        painter.end()
        self.setPixmap(canvas)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", default="DEV-ROOM")
    parser.add_argument("--backend", default="http://127.0.0.1:8001")
    parser.add_argument("--map", type=Path, default=ROOT / "assets/rooms/dev_room_2d.svg")
    parser.add_argument("--cameras", nargs="+", default=["CAM-01", "CAM-04"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    projections = {}
    for camera_id in args.cameras:
        path = ROOT / "config/room_2d" / args.room / f"{camera_id}.json"
        if not path.exists():
            raise SystemExit(f"Calibration yo‘q: {path}\nAvval scripts/calibrate_room_2d.py bilan calibrate qiling.")
        projections[camera_id] = RoomProjection.load(path)

    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle(f"{args.room} · Live 2D")
    window.resize(1200, 900)
    window.setCentralWidget(RoomMap(args.map.expanduser().resolve(), projections, args.backend))
    window.show()
    raise SystemExit(app.exec())
