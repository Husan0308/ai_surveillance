from __future__ import annotations

import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QListWidget,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from services.frontend.app.api_client import ApiClient
from services.frontend.app.config import load_settings


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = load_settings()

        self.setWindowTitle("AI Surveillance")
        self.resize(720, 520)

        self.api_status = QLabel("API: checking...")
        self.ml_status = QLabel("ML: checking...")
        self.camera_count = QLabel("Cameras: checking...")
        self.camera_list = QListWidget()

        title = QLabel("AI Surveillance — Service Status")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(self.api_status)
        layout.addWidget(self.ml_status)
        layout.addWidget(self.camera_count)
        layout.addWidget(self.camera_list, 1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.api = ApiClient(self.settings.api_base_url, self)
        self.api.api_health_received.connect(self._on_api_health)
        self.api.ml_health_received.connect(self._on_ml_health)
        self.api.cameras_received.connect(self._on_cameras)
        self.api.request_failed.connect(self._on_request_failed)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(self.settings.refresh_interval_ms)
        self.refresh_timer.timeout.connect(self.api.refresh_all)
        self.refresh_timer.start()

        self.api.refresh_all()

    def _on_api_health(self, data: dict) -> None:
        status = data.get("status", "unknown")
        self.api_status.setText(f"API: {status}")

    def _on_ml_health(self, data: dict) -> None:
        status = data.get("status", "unknown")
        camera_count = data.get("camera_count", "?")
        last_error = data.get("last_error")

        text = f"ML: {status} | cameras={camera_count}"
        if last_error:
            text += f" | error={last_error}"
        self.ml_status.setText(text)

    def _on_cameras(self, data: dict) -> None:
        cameras = data.get("cameras", [])
        self.camera_count.setText(f"Cameras: {data.get('count', len(cameras))}")

        camera_ids = [str(camera.get("id", "unknown")) for camera in cameras]
        current_ids = [self.camera_list.item(i).text() for i in range(self.camera_list.count())]
        if camera_ids == current_ids:
            return

        self.camera_list.clear()
        self.camera_list.addItems(camera_ids)

    def _on_request_failed(self, request_name: str, reason: str) -> None:
        if request_name == "api_health":
            self.api_status.setText(f"API: unavailable ({reason})")
        elif request_name == "ml_health":
            self.ml_status.setText(f"ML: unavailable ({reason})")
        elif request_name == "cameras":
            self.camera_count.setText(f"Cameras: unavailable ({reason})")


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
