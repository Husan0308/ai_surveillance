#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.ml_service.core_v1.room_projection import reprojection_errors, solve_homography


class ClickImage(QLabel):
    clicked = Signal(float, float)

    def __init__(self, title: str):
        super().__init__(title)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(520, 360)
        self.setStyleSheet("background:#06090f;color:#8893a3;border:1px solid #253044;")
        self._source = QPixmap()
        self._points: list[tuple[float, float]] = []

    def set_source(self, pixmap: QPixmap):
        self._source = pixmap
        self._render()

    def set_points(self, points):
        self._points = list(points)
        self._render()

    def _target_rect(self):
        if self._source.isNull():
            return None
        scaled = self._source.size().scaled(self.size(), Qt.KeepAspectRatio)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        return x, y, scaled.width(), scaled.height()

    def _render(self):
        if self._source.isNull():
            return
        rect = self._target_rect()
        if rect is None:
            return
        x0, y0, w, h = rect
        canvas = QPixmap(self.size())
        canvas.fill(Qt.black)
        painter = QPainter(canvas)
        painter.drawPixmap(x0, y0, self._source.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        pen = QPen(Qt.red)
        pen.setWidth(3)
        painter.setPen(pen)
        sx = w / max(1, self._source.width())
        sy = h / max(1, self._source.height())
        for index, (px, py) in enumerate(self._points, 1):
            dx = x0 + px * sx
            dy = y0 + py * sy
            painter.drawEllipse(int(dx - 6), int(dy - 6), 12, 12)
            painter.drawText(int(dx + 9), int(dy - 8), str(index))
        painter.end()
        self.setPixmap(canvas)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    def mousePressEvent(self, event):
        if self._source.isNull():
            return
        rect = self._target_rect()
        if rect is None:
            return
        x0, y0, w, h = rect
        x, y = event.position().x(), event.position().y()
        if not (x0 <= x <= x0 + w and y0 <= y <= y0 + h):
            return
        px = (x - x0) * self._source.width() / w
        py = (y - y0) * self._source.height() / h
        self.clicked.emit(float(px), float(py))


class CalibrationWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.camera_points: list[tuple[float, float]] = []
        self.map_points: list[tuple[float, float]] = []
        self.pending_camera: tuple[float, float] | None = None
        self.setWindowTitle(f"2D Calibration · {args.room} · {args.camera}")
        self.resize(1450, 760)

        self.camera_view = ClickImage(args.camera)
        self.map_view = ClickImage(args.room)
        self.camera_view.clicked.connect(self.camera_click)
        self.map_view.clicked.connect(self.map_click)

        self.status = QLabel("Camera nuqtani bosing, keyin 2D mapdagi aynan o‘sha fizik joyni bosing. 6–10 juft nuqta yaxshi.")
        self.status.setStyleSheet("color:#d5dbea;padding:8px;")

        refresh = QPushButton("Refresh camera")
        refresh.clicked.connect(self.load_camera)
        undo = QPushButton("Undo")
        undo.clicked.connect(self.undo)
        reset = QPushButton("Reset")
        reset.clicked.connect(self.reset)
        save = QPushButton("Solve + Save")
        save.clicked.connect(self.save)

        top = QHBoxLayout()
        top.addWidget(self.camera_view, 1)
        top.addWidget(self.map_view, 1)
        buttons = QHBoxLayout()
        for button in (refresh, undo, reset, save):
            buttons.addWidget(button)
        layout = QVBoxLayout()
        layout.addLayout(top, 1)
        layout.addWidget(self.status)
        layout.addLayout(buttons)
        root = QWidget()
        root.setLayout(layout)
        root.setStyleSheet("background:#02050a;")
        self.setCentralWidget(root)

        self.load_map()
        self.load_camera()

    def load_map(self):
        pixmap = QPixmap(str(self.args.map))
        if pixmap.isNull():
            raise RuntimeError(f"room map ochilmadi: {self.args.map}")
        self.map_view.set_source(pixmap)

    def load_camera(self):
        try:
            with urllib.request.urlopen(f"{self.args.backend}/frame/{self.args.camera}", timeout=3) as response:
                payload = response.read()
            image = QImage.fromData(payload, "JPG")
            if image.isNull():
                raise RuntimeError("camera JPEG decode bo‘lmadi")
            self.camera_view.set_source(QPixmap.fromImage(image))
            self.status.setText(f"{self.args.camera} snapshot tayyor. Camera point → map point tartibida bosing.")
        except Exception as exc:
            self.status.setText(f"Camera snapshot xato: {exc}")

    def camera_click(self, x, y):
        if self.pending_camera is not None:
            self.status.setText("Avval pending camera nuqta uchun mapdagi mos nuqtani bosing.")
            return
        self.pending_camera = (x, y)
        self.camera_view.set_points(self.camera_points + [self.pending_camera])
        self.status.setText(f"Camera nuqta {len(self.camera_points)+1} tanlandi. Endi mapdagi aynan shu joyni bosing.")

    def map_click(self, x, y):
        if self.pending_camera is None:
            self.status.setText("Avval camera rasmidagi fizik pol nuqtasini bosing.")
            return
        self.camera_points.append(self.pending_camera)
        self.map_points.append((x, y))
        self.pending_camera = None
        self.camera_view.set_points(self.camera_points)
        self.map_view.set_points(self.map_points)
        self.status.setText(f"{len(self.camera_points)} juft nuqta. Kamida 4, tavsiya 6–10.")

    def undo(self):
        if self.pending_camera is not None:
            self.pending_camera = None
        elif self.camera_points:
            self.camera_points.pop()
            self.map_points.pop()
        self.camera_view.set_points(self.camera_points)
        self.map_view.set_points(self.map_points)
        self.status.setText(f"{len(self.camera_points)} juft nuqta qoldi.")

    def reset(self):
        self.pending_camera = None
        self.camera_points.clear()
        self.map_points.clear()
        self.camera_view.set_points([])
        self.map_view.set_points([])
        self.status.setText("Reset qilindi.")

    def save(self):
        if len(self.camera_points) < 4:
            QMessageBox.warning(self, "Calibration", "Kamida 4 juft nuqta kerak.")
            return
        matrix, mask = solve_homography(self.camera_points, self.map_points, self.args.ransac)
        errors = reprojection_errors(self.camera_points, self.map_points, matrix)
        payload = {
            "room_id": self.args.room,
            "camera_id": self.args.camera,
            "map_asset": str(self.args.map.relative_to(ROOT) if self.args.map.is_relative_to(ROOT) else self.args.map),
            "map_size": [self.map_view._source.width(), self.map_view._source.height()],
            "camera_size": [self.camera_view._source.width(), self.camera_view._source.height()],
            "camera_points": [[round(x, 3), round(y, 3)] for x, y in self.camera_points],
            "map_points": [[round(x, 3), round(y, 3)] for x, y in self.map_points],
            "homography": np.asarray(matrix).tolist(),
            "inliers": [int(v) for v in np.asarray(mask).reshape(-1).tolist()],
            "reprojection_error_px": {
                "mean": float(errors.mean()),
                "max": float(errors.max()),
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.args.output.parent.mkdir(parents=True, exist_ok=True)
        self.args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.status.setText(f"Saved: {self.args.output} · mean error={errors.mean():.2f}px max={errors.max():.2f}px")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", required=True)
    parser.add_argument("--room", default="DEV-ROOM")
    parser.add_argument("--backend", default="http://127.0.0.1:8001")
    parser.add_argument("--map", type=Path, default=ROOT / "assets/rooms/dev_room_2d.svg")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ransac", type=float, default=4.0)
    args = parser.parse_args()
    args.map = args.map.expanduser().resolve()
    if args.output is None:
        args.output = ROOT / "config/room_2d" / args.room / f"{args.camera}.json"
    else:
        args.output = args.output.expanduser().resolve()
    return args


if __name__ == "__main__":
    args = parse_args()
    app = QApplication(sys.argv)
    window = CalibrationWindow(args)
    window.show()
    raise SystemExit(app.exec())
