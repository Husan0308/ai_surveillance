from __future__ import annotations

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget

from services.frontend.app.api_client import ApiClient
from services.frontend.app.camera_wall import CameraWall
from services.frontend.app.room_pair_panel import RoomPairPanel
from services.frontend.app.config import load_settings


ALL_CAMERAS = tuple(f"CAM-{index:02d}" for index in range(1, 7))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = load_settings()
        self.setWindowTitle("AI Surveillance")
        self.resize(1920, 1080)

        self.api_status = QLabel("API: checking...")
        self.ml_status = QLabel("ML: checking...")
        self.camera_count = QLabel("Camera wall: checking...")
        self.dev_room_status = QLabel("Dev Room: checking...")

        status_row = QHBoxLayout()
        status_row.addWidget(self.api_status)
        status_row.addWidget(self.ml_status)
        status_row.addWidget(self.camera_count)
        status_row.addWidget(self.dev_room_status)
        status_row.addStretch(1)

        self.camera_wall = CameraWall(ml_video_base_url=self.settings.ml_video_base_url)
        self.camera_wall.set_cameras(list(ALL_CAMERAS))
        self.room_pair_panel = RoomPairPanel()

        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(8)
        content.addWidget(self.camera_wall, 3)
        content.addWidget(self.room_pair_panel, 1)

        layout = QVBoxLayout()
        layout.addLayout(status_row)
        layout.addLayout(content, 1)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.api = ApiClient(self.settings.api_base_url, self)
        self.api.api_health_received.connect(self._on_api_health)
        self.api.ml_health_received.connect(self._on_ml_health)
        self.api.cameras_received.connect(self._on_cameras)
        self.api.room_pair_received.connect(self._on_room_pair)
        self.api.acceptance_candidates_received.connect(self._on_acceptance_candidates)
        self.api.request_failed.connect(self._on_request_failed)

        self.acceptance_mode_enabled = False
        self.room_pair_panel.acceptance_mode_changed.connect(self._on_acceptance_mode_changed)
        self.room_pair_panel.acceptance_candidates_requested.connect(self._on_acceptance_mode_changed)
        self.room_pair_panel.acceptance_start_requested.connect(self._start_acceptance_subtest)
        self.room_pair_panel.acceptance_candidates_changed.connect(self.camera_wall.set_acceptance_candidates)
        self.room_pair_panel.acceptance_labels_changed.connect(self.camera_wall.set_acceptance_labels)
        self.camera_wall.acceptance_bbox_selected.connect(self.room_pair_panel.handle_acceptance_bbox_selected)

        self.api_timer = QTimer(self)
        self.api_timer.setInterval(self.settings.refresh_interval_ms)
        self.api_timer.timeout.connect(self._refresh_api)
        self.api_timer.start()

        self.frame_timer = QTimer(self)
        self.frame_timer.setInterval(self.settings.frame_refresh_interval_ms)
        self.frame_timer.timeout.connect(self._refresh_frames)
        self.frame_timer.start()

        self.api.refresh_all()

    def _refresh_api(self) -> None:
        self.api.refresh_all(self.acceptance_mode_enabled)

    def _on_acceptance_mode_changed(self, enabled: bool) -> None:
        self.acceptance_mode_enabled = bool(enabled)
        self.camera_wall.set_acceptance_mode(self.acceptance_mode_enabled)

    def _on_acceptance_candidates(self, payload: dict) -> None:
        self.room_pair_panel.update_acceptance_candidates(payload)

    def _start_acceptance_subtest(self) -> None:
        required = ("CAM-01", "CAM-04")
        unhealthy = [
            camera_id for camera_id in required
            if camera_id not in self.camera_wall.tiles or not self.camera_wall.tiles[camera_id].is_connected()
        ]
        if unhealthy:
            self.room_pair_panel.acceptance_status.setText(
                "Acceptance timer not started: waiting for live preview from " + ", ".join(unhealthy)
            )
            return
        self.room_pair_panel.start_subtest()

    def _refresh_frames(self) -> None:
        self.camera_wall.refresh_frames()
        online, total = self.camera_wall.connection_counts()
        self.camera_count.setText(f"Camera wall: {online}/{total}")

    def _on_api_health(self, data: dict) -> None:
        self.api_status.setText(f"API: {data.get('status', 'unknown')}")

    def _on_ml_health(self, data: dict) -> None:
        status = data.get("status", "unknown")
        self.ml_status.setText(f"ML: {status}")

    def _on_cameras(self, data: dict) -> None:
        cameras = data.get("cameras", [])
        camera_ids = [str(camera.get("id", "unknown")) for camera in cameras]
        # API camera registry status is not the V11 preview transport status.
        # The wall header is updated from frames actually displayed below.
        ordered_camera_ids = [camera_id for camera_id in ALL_CAMERAS if camera_id in camera_ids]
        self.camera_wall.set_cameras(ordered_camera_ids or list(ALL_CAMERAS))

    def _on_room_pair(self, data: dict) -> None:
        self.room_pair_panel.update_snapshot(data)
        readiness = data.get("readiness", {})
        self.dev_room_status.setText("Dev Room: ready" if readiness.get("ready") else "Dev Room: not ready")
        self.camera_wall.set_source_mode(str(data.get("source_mode", "live")))
        self.camera_wall.set_identity_snapshot(data.get("people", []))

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
    window.showMaximized()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
