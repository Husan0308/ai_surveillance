from __future__ import annotations

import sys

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget

from services.frontend.app.api_client import ApiClient
from services.frontend.app.camera_wall import CameraWall
from services.frontend.app.config import load_settings


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = load_settings()
        self.setWindowTitle("AI Surveillance")
        self.resize(1500, 920)

        self.api_status = QLabel("API: checking...")
        self.ml_status = QLabel("ML: checking...")
        self.camera_count = QLabel("Cameras: checking...")

        status_row = QHBoxLayout()
        status_row.addWidget(self.api_status)
        status_row.addWidget(self.ml_status)
        status_row.addWidget(self.camera_count)
        status_row.addStretch(1)

        self.camera_wall = CameraWall(self.settings)

        layout = QVBoxLayout()
        layout.addLayout(status_row)
        layout.addWidget(self.camera_wall, 1)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.api = ApiClient(self.settings.api_base_url, self)
        self.api.api_health_received.connect(self._on_api_health)
        self.api.ml_health_received.connect(self._on_ml_health)
        self.api.cameras_received.connect(self._on_cameras)
        self.api.tracks_received.connect(self.camera_wall.update_tracks)
        self.api.request_failed.connect(self._on_request_failed)

        self.api_timer = QTimer(self)
        self.api_timer.setInterval(self.settings.refresh_interval_ms)
        self.api_timer.timeout.connect(self.api.refresh_all)
        self.api_timer.start()

        self.track_timer = QTimer(self)
        self.track_timer.setInterval(self.settings.track_refresh_interval_ms)
        self.track_timer.timeout.connect(self.api.refresh_tracks)
        self.track_timer.start()

        # Proven mmap wall behavior: poll quickly with a PreciseTimer, but each
        # tile repaints only when its mmap sequence changes. No frame backlog.
        self.frame_timer = QTimer(self)
        self.frame_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.frame_timer.setInterval(self.settings.frame_refresh_interval_ms)
        self.frame_timer.timeout.connect(self.camera_wall.refresh_frames)
        self.frame_timer.start()

        self.api.refresh_all()
        self.api.refresh_tracks()

    def _on_api_health(self, data: dict) -> None:
        self.api_status.setText(f"API: {data.get('status', 'unknown')}")

    def _on_ml_health(self, data: dict) -> None:
        status = data.get("status", "unknown")
        online = data.get("online_camera_count", "?")
        total = data.get("camera_count", "?")
        self.ml_status.setText(f"ML: {status} | online={online}/{total}")

    def _on_cameras(self, data: dict) -> None:
        cameras = [row for row in data.get("cameras", []) if isinstance(row, dict)]
        online = sum(1 for camera in cameras if camera.get("online"))
        self.camera_count.setText(f"Cameras: {online}/{data.get('count', len(cameras))}")
        self.camera_wall.set_cameras(cameras)

    def _on_request_failed(self, request_name: str, reason: str) -> None:
        if request_name == "api_health":
            self.api_status.setText(f"API: unavailable ({reason})")
        elif request_name == "ml_health":
            self.ml_status.setText(f"ML: unavailable ({reason})")
        elif request_name == "cameras":
            self.camera_count.setText(f"Cameras: unavailable ({reason})")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.camera_wall.close_readers()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
