from __future__ import annotations

import sys

from PySide6.QtCore import QTimer
from PySide6.QtNetwork import QNetworkAccessManager
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget

from services.frontend.app.api_client import ApiClient
from services.frontend.app.camera_wall import CameraWall
from services.frontend.app.config import load_settings
from services.frontend.app.realtime_client import MonitoringSocket
from services.frontend.app.realtime_models import (
    LatestMetadataStore,
    parse_camera_rows,
    parse_track_message,
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = load_settings()
        self.setWindowTitle("AI Surveillance")
        self.resize(1500, 920)

        self.api_status = QLabel("API: checking...")
        self.ml_status = QLabel("ML: checking...")
        self.camera_count = QLabel("Cameras: checking...")
        self.realtime_status = QLabel("Realtime: connecting...")

        status_row = QHBoxLayout()
        status_row.addWidget(self.api_status)
        status_row.addWidget(self.ml_status)
        status_row.addWidget(self.camera_count)
        status_row.addWidget(self.realtime_status)
        status_row.addStretch(1)

        # Keep the six long-lived MJPEG requests on a separate HTTP connection
        # pool from short control-plane API requests. Qt HTTP/1 executes only six
        # requests in parallel per host/port for each QNetworkAccessManager.
        self.control_network = QNetworkAccessManager(self)
        self.stream_network = QNetworkAccessManager(self)

        self.metadata_store = LatestMetadataStore()
        self.camera_wall = CameraWall(
            api_base_url=self.settings.api_base_url,
            manager=self.stream_network,
            decode_interval_ms=self.settings.frame_refresh_interval_ms,
            parent=self,
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addLayout(status_row)
        layout.addWidget(self.camera_wall, 1)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.api = ApiClient(
            self.settings.api_base_url,
            self.control_network,
            self,
        )
        self.api.api_health_received.connect(self._on_api_health)
        self.api.ml_health_received.connect(self._on_ml_health)
        self.api.cameras_received.connect(self._on_cameras)
        self.api.request_failed.connect(self._on_request_failed)

        self.monitoring = MonitoringSocket(
            self.settings.api_base_url,
            reconnect_ms=self.settings.ws_reconnect_ms,
            parent=self,
        )
        self.monitoring.message_received.connect(self._on_monitoring_message)
        self.monitoring.state_changed.connect(self._on_realtime_state)

        self.api_timer = QTimer(self)
        self.api_timer.setInterval(max(500, self.settings.refresh_interval_ms))
        self.api_timer.timeout.connect(self.api.refresh_all)
        self.api_timer.start()

        self.api.refresh_all()
        self.monitoring.start()

    def _on_api_health(self, data: dict) -> None:
        self.api_status.setText(f"API: {data.get('status', 'unknown')}")

    def _on_ml_health(self, data: dict) -> None:
        status = data.get("status", "unknown")
        online = data.get("online_camera_count", "?")
        total = data.get("camera_count", "?")
        tracking = data.get("tracking")
        tracker_ready = ""
        if isinstance(tracking, dict):
            tracker_ready = " | tracking=ready" if tracking.get("ready") else " | tracking=starting"
        self.ml_status.setText(f"ML: {status} | online={online}/{total}{tracker_ready}")

    def _on_cameras(self, data: dict) -> None:
        cameras = parse_camera_rows(data)
        online = sum(1 for camera in cameras if camera.online)
        self.camera_count.setText(f"Cameras: {online}/{len(cameras)}")
        self.camera_wall.set_cameras(cameras)

        # A track update can arrive before the slower REST camera discovery call.
        # Re-apply latest metadata after tiles are created so the first rendered
        # frame already has the freshest available overlay.
        for camera in cameras:
            metadata = self.metadata_store.get(camera.camera_id)
            if metadata is not None:
                self.camera_wall.update_metadata(camera.camera_id, metadata)

    def _on_monitoring_message(self, data: dict) -> None:
        parsed = parse_track_message(data)
        if parsed is not None:
            camera_id, metadata = parsed
            if self.metadata_store.update(camera_id, metadata):
                self.camera_wall.update_metadata(camera_id, metadata)
            return

        if data.get("type") == "service" and data.get("status") == "unavailable":
            reason = str(data.get("reason") or "ML service unavailable")
            self.realtime_status.setText(f"Realtime: ML unavailable ({reason})")

    def _on_realtime_state(self, state: str) -> None:
        self.realtime_status.setText(f"Realtime: {state.lower()}")

    def _on_request_failed(self, request_name: str, reason: str) -> None:
        if request_name == "api_health":
            self.api_status.setText(f"API: unavailable ({reason})")
        elif request_name == "ml_health":
            self.ml_status.setText(f"ML: unavailable ({reason})")
        elif request_name == "cameras":
            self.camera_count.setText(f"Cameras: unavailable ({reason})")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.api_timer.stop()
        self.monitoring.stop()
        self.camera_wall.close_streams()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
