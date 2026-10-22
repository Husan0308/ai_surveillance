from __future__ import annotations

import json
import math
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QUrl, Signal, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtNetwork import QAbstractSocket, QNetworkAccessManager, QNetworkRequest
from PySide6.QtWebSockets import QWebSocket
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QTableWidget,
    QTableWidgetItem, QSlider, QVBoxLayout, QWidget,
)

from services.mv3dt_staging.presentation import display_state, update_active_observations, world_to_map_pixel


CAMERA_COLORS = {
    "CAM-01": QColor("#ffb347"), "CAM-04": QColor("#42d6c7"),
    "CAM-02": QColor("#ff7f50"), "CAM-05": QColor("#69a7ff"),
}


class MapCanvas(QWidget):
    def __init__(self, map_path: str | Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.map_pixmap = QPixmap(str(map_path))
        self.transform: list[float] | None = None
        self.observations: list[dict[str, Any]] = []
        self.primary_markers: list[dict[str, Any]] = []
        self.current_observations: list[dict[str, Any]] = []
        self.show_camera_observations = False
        self.show_display_aliases = False
        self.separate_by_camera = False
        self.show_trails = False
        self.show_marker_labels = True
        self.trail_window_frames = 180
        self.trail_max_gap_frames = 30
        self._trail_last_frame: int | None = None
        self.trails: dict[tuple[str, str], deque[tuple[int, tuple[float, float]]]] = {}
        self.setMinimumSize(QSize(560, 400))

    def set_transform(self, matrix: list[float] | None) -> None:
        self.transform = matrix if matrix and len(matrix) == 9 else None
        self.update()

    def set_observations(self, rows: list[dict[str, Any]], *, current_frame: int | None = None,
                         visual_review: dict[str, Any] | None = None,
                         use_display_aliases: bool = False,
                         separate_by_camera: bool = False) -> None:
        self.observations = list(rows)
        self.show_display_aliases = use_display_aliases
        self.separate_by_camera = separate_by_camera
        frame = current_frame if current_frame is not None else max((int(row["frame_id"]) for row in rows), default=0)
        self._record_trail_samples(rows, frame)
        state = display_state(rows, current_frame=frame, visual_review=visual_review,
                              use_display_aliases=use_display_aliases,
                              separate_by_camera=separate_by_camera)
        self.primary_markers = state["primary_markers"]
        self.current_observations = state["current_observations"]
        self.update()

    def reset_trails(self) -> None:
        self.trails.clear()
        self._trail_last_frame = None
        self.update()

    def _record_trail_samples(self, rows: list[dict[str, Any]], frame: int) -> None:
        if self._trail_last_frame is not None and frame < self._trail_last_frame:
            # Seeking or replaying backwards must not draw a false connecting path.
            self.trails.clear()
        self._trail_last_frame = frame
        for row in rows:
            try:
                row_frame = int(row["frame_id"])
                world = row.get("world_position_unverified_units")
                if row_frame != frame or not isinstance(world, list) or len(world) != 2:
                    continue
                position = (float(world[0]), float(world[1]))
                if not all(math.isfinite(value) for value in position):
                    continue
                camera = str(row.get("camera_id", ""))
                raw_id = str(row.get("raw_global_id", row.get("mv3dt_global_id", "")))
                if not camera or not raw_id:
                    continue
                history = self.trails.setdefault((camera, raw_id), deque())
                if history and history[-1][0] == frame:
                    history[-1] = (frame, position)
                else:
                    history.append((frame, position))
                while history and frame - history[0][0] > self.trail_window_frames:
                    history.popleft()
            except (KeyError, TypeError, ValueError, OverflowError):
                continue

    def _image_rect(self) -> QRectF:
        area = QRectF(self.rect()).adjusted(8, 8, -8, -8)
        if self.map_pixmap.isNull():
            return area
        image_size = self.map_pixmap.size()
        image_aspect = image_size.width() / max(1, image_size.height())
        area_aspect = area.width() / max(1.0, area.height())
        if area_aspect > image_aspect:
            height = area.height()
            width = height * image_aspect
        else:
            width = area.width()
            height = width / image_aspect
        return QRectF(area.center().x() - width / 2, area.center().y() - height / 2, width, height)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101820"))
        target = self._image_rect()
        if self.map_pixmap.isNull():
            painter.setPen(QColor("#ffffff"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Dev Room map unavailable")
            return
        painter.drawPixmap(target, self.map_pixmap, QRectF(self.map_pixmap.rect()))
        if self.transform is None:
            painter.setPen(QColor("#ffffff"))
            painter.drawText(target, Qt.AlignmentFlag.AlignCenter, "Waiting for map transform metadata")
            return
        image_w = self.map_pixmap.width()
        image_h = self.map_pixmap.height()
        if self.show_trails:
            self._draw_trails(painter, target, image_w, image_h)
        render_rows = [(row, True) for row in self.primary_markers]
        if self.show_camera_observations:
            render_rows += [(row, False) for row in self.current_observations]
        for row, primary in render_rows:
            world = row.get("world_position_unverified_units")
            if not isinstance(world, list) or len(world) != 2:
                continue
            try:
                px, py = world_to_map_pixel(world[0], world[1], self.transform)
            except (TypeError, ValueError):
                continue
            if not (0 <= px < image_w and 0 <= py < image_h):
                continue
            point = QPointF(target.x() + (px / image_w) * target.width(),
                            target.y() + (py / image_h) * target.height())
            camera = str(row.get("camera_id", ""))
            color = CAMERA_COLORS.get(camera, QColor("#ffffff"))
            painter.setPen(QPen(QColor("#14212a"), 2))
            painter.setBrush(color if primary else Qt.BrushStyle.NoBrush)
            painter.drawEllipse(point, 8 if primary else 4, 8 if primary else 4)
            painter.setFont(QFont("Sans Serif", 9, QFont.Weight.Bold))
            painter.setPen(QColor("#111111"))
            if primary and self.show_marker_labels:
                raw_ids = ",".join(row.get("raw_global_ids", []))
                if self.separate_by_camera:
                    label = f"{camera} · MV3DT output ID {raw_ids or row.get('mv3dt_global_id')}"
                else:
                    sources = "/".join(row.get("observing_cameras", [camera]))
                    aliases = row.get("reviewed_duplicate_ids", [])
                    raw_label = f" (raw {raw_ids})" if raw_ids and raw_ids != str(row.get("display_global_id")) else ""
                    label = f"Display ID {row.get('display_global_id', row.get('mv3dt_global_id'))}{raw_label} · {sources}"
                    if aliases:
                        label += f" (+{','.join(aliases)} reviewed duplicate)"
                painter.drawText(QRectF(point.x() + 10, point.y() - 21, 360, 24), label)
            elif not primary and self.show_marker_labels:
                raw_id = row.get("raw_global_id", row.get("mv3dt_global_id"))
                display_id = row.get("display_global_id", raw_id) if self.show_display_aliases else raw_id
                label = f"obs {camera} raw {raw_id}"
                if display_id != raw_id:
                    label += f" → display {display_id}"
                painter.drawText(QRectF(point.x() + 7, point.y() + 5, 260, 22), label)

    def _draw_trails(self, painter: QPainter, target: QRectF, image_w: int, image_h: int) -> None:
        for (camera, _raw_id), samples in self.trails.items():
            if len(samples) < 2:
                continue
            color = QColor(CAMERA_COLORS.get(camera, QColor("#ffffff")))
            color.setAlpha(165)
            painter.setPen(QPen(color, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            for previous, current in zip(samples, list(samples)[1:]):
                previous_frame, previous_world = previous
                current_frame, current_world = current
                if current_frame <= previous_frame or current_frame - previous_frame > self.trail_max_gap_frames:
                    continue
                try:
                    pxy = world_to_map_pixel(previous_world[0], previous_world[1], self.transform)
                    cxy = world_to_map_pixel(current_world[0], current_world[1], self.transform)
                except (TypeError, ValueError):
                    continue
                if not (0 <= pxy[0] < image_w and 0 <= pxy[1] < image_h
                        and 0 <= cxy[0] < image_w and 0 <= cxy[1] < image_h):
                    continue
                p = QPointF(target.x() + pxy[0] / image_w * target.width(),
                            target.y() + pxy[1] / image_h * target.height())
                c = QPointF(target.x() + cxy[0] / image_w * target.width(),
                            target.y() + cxy[1] / image_h * target.height())
                painter.drawLine(p, c)


class MainWindow(QMainWindow):
    replay_complete = Signal()

    def __init__(self, map_path: str | Path, ws_url: str, parent: QWidget | None = None,
                 *, auto_start: bool = True, room_title: str = "Dev Room",
                 review_video_path: str | Path | None = None,
                 camera_scoped_ids: bool = False,
                 show_map_with_video: bool = False,
                 calibration_status: str | None = None,
                 standalone_demo: bool = False) -> None:
        super().__init__(parent)
        if standalone_demo and review_video_path is None:
            raise ValueError("standalone recorded demo requires a local three-panel review video")
        parsed_ws_url = urlsplit(ws_url)
        if parsed_ws_url.scheme != "ws" or parsed_ws_url.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("MV3DT staging Qt client only connects to a loopback ws:// endpoint")
        self.room_title = room_title
        self.standalone_demo = standalone_demo
        self.camera_scoped_ids = camera_scoped_ids
        self.show_map_with_video = show_map_with_video and review_video_path is not None
        self.calibration_status = calibration_status or (
            "CALIBRATION: UNVERIFIED — map scale, pairwise geometry and independent floor-point accuracy are not established."
        )
        self.setWindowTitle(
            f"{room_title} — DEMO / RECORDED VIDEO — NOT LIVE"
            if standalone_demo else f"{room_title} — MV3DT offline staging (experimental)"
        )
        self.resize(1720 if self.show_map_with_video else 1500, 1040 if self.show_map_with_video else 960)
        self.current_frame = 0
        self.received_frame_batches = 0
        self.received_observations = 0
        self._replay_completed = False
        self.display_aliases_available = False
        self._alias_toggle_initialized = False
        self._suppressed_adoption_rows = 0
        self._play_video_on_stream_start = False
        self._seek_video_on_stream_start = False
        self._resume_after_seek = False
        self.last_replay_frame_id: int | None = None
        self.nominal_fps = 30.0
        self.transform: list[float] | None = None
        self.active: list[dict[str, Any]] = []
        self.visual_review: dict[str, Any] | None = None
        self.last_camera_frame: dict[str, int] = {}
        self.ws_url = ws_url
        host = f"[{parsed_ws_url.hostname}]" if ":" in (parsed_ws_url.hostname or "") else parsed_ws_url.hostname
        port = f":{parsed_ws_url.port}" if parsed_ws_url.port else ""
        http_scheme = "https" if parsed_ws_url.scheme == "wss" else "http"
        self.http_base_url = f"{http_scheme}://{host}{port}"
        self._frame_mode_initialized = False
        self._frame_mode_autoplay = False
        self._frame_mode_first_frame = 0
        self._frame_mode_last_frame = 0
        self._frame_reply = None
        self._pending_exact_frame: int | None = None
        self._network = QNetworkAccessManager(self) if self.show_map_with_video else None
        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(100)
        self._frame_timer.timeout.connect(self._request_video_timeline_frame)

        title = QLabel(
            f"{room_title} · DEMO / RECORDED VIDEO — NOT LIVE"
            if standalone_demo else f"{room_title} · MV3DT offline replay"
        )
        title.setStyleSheet("font-size: 22px; font-weight: 700; color: #f3f7fa")
        self.warning = QLabel(
            ("EXPERIMENTAL / UNCALIBRATED — recorded CAM-02/CAM-05 playback only; camera-local IDs stay separate. "
             "Markers are illustrative; map scale and independent floor-point accuracy are unverified. "
             "The two recordings are paired by file-frame index, not a verified common exposure clock.")
            if standalone_demo else
            (("EXPERIMENTAL / UNCALIBRATED — camera-scoped MV3DT output IDs are kept separate. "
             "Markers are illustrative projected foot positions, not metric locations; map scale, pairwise geometry "
             "and source-time synchronization are unverified. The event log has no source PTS.")
            if camera_scoped_ids else
            ("EXPERIMENTAL — MV3DT output IDs and projected positions are not identity or geometry ground truth. "
             "Map scale is unverified; source PTS is not present in the event log."))
        )
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("background:#5a3512;color:#fff2d6;padding:9px;border-radius:4px")
        self.status = QLabel("Offline replay: connecting…")
        self.status.setStyleSheet("color:#cee0e8;padding:4px")
        self.counters = QLabel("Camera observations: 0 · Camera-scoped output labels: 0 · Confirmed visible people: UNAVAILABLE")
        self.counters.setStyleSheet("font-size:15px;font-weight:600;padding:4px")
        self.review_status = QLabel("People confirmation is available only for explicitly reviewed source frames.")
        self.review_status.setWordWrap(True)
        self.map_canvas = MapCanvas(map_path)
        self.map_canvas.show_marker_labels = not (camera_scoped_ids and self.show_map_with_video)
        self.review_video_path = Path(review_video_path).expanduser().resolve() if review_video_path else None
        if self.review_video_path is not None and not self.review_video_path.is_file():
            raise ValueError("review video path must be an existing local file")
        self.review_video_widget: QVideoWidget | None = None
        self.video_player: QMediaPlayer | None = None
        self.video_play_button: QPushButton | None = None
        self.video_stop_button: QPushButton | None = None
        self.video_slider: QSlider | None = None
        self.video_position_label: QLabel | None = None
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "Camera", "Source", "Raw Global ID", "Display ID", "Frame / nominal",
            "World units (unverified)", "Alias provenance / review",
        ])
        if camera_scoped_ids:
            self.table.setHorizontalHeaderItem(3, QTableWidgetItem("Camera-scoped label"))
        self.table.setColumnWidth(0, 78)
        self.table.setColumnWidth(1, 56)
        self.table.setColumnWidth(2, 112)
        self.table.setColumnWidth(3, 82)
        self.table.setColumnWidth(4, 112)
        self.table.setColumnWidth(5, 175)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumWidth(640)
        self.replay_button = QPushButton("Replay from beginning")
        self.replay_button.clicked.connect(self.replay)
        self.clear_button = QPushButton("Clear markers")
        self.clear_button.clicked.connect(self.clear_markers)
        self.observation_toggle = QCheckBox("Show camera observation points")
        self.observation_toggle.toggled.connect(self._toggle_observations)
        self.trail_toggle = QCheckBox("Show short movement trails (6 s nominal)")
        self.trail_toggle.toggled.connect(self._toggle_trails)
        self.map_label_toggle = QCheckBox("Show marker labels on map")
        self.map_label_toggle.setChecked(self.map_canvas.show_marker_labels)
        self.map_label_toggle.toggled.connect(self._toggle_map_labels)
        self.alias_toggle = QCheckBox("Apply confirmed ADOPTED_ID aliases (display only)")
        self.alias_toggle.setEnabled(False)
        if camera_scoped_ids:
            self.alias_toggle.setText("Cross-camera adoption aliases disabled (camera-scoped mode)")
        self.alias_toggle.toggled.connect(self._on_alias_toggle)
        self.alias_status = QLabel("Cross-camera ID merging is disabled" if camera_scoped_ids
                                   else "ADOPTED_ID display aliasing unavailable")
        self.alias_status.setWordWrap(True)
        controls = QHBoxLayout()
        controls.addWidget(self.replay_button)
        controls.addWidget(self.clear_button)
        controls.addWidget(self.observation_toggle)
        controls.addWidget(self.trail_toggle)
        controls.addWidget(self.map_label_toggle)
        controls.addWidget(self.alias_toggle)
        controls.addStretch(1)
        if self.review_video_path is not None:
            self.review_video_widget = QVideoWidget()
            if standalone_demo:
                # The pre-rendered review is an intentionally wide 1760x360
                # three-panel composite. Keep the widget near that aspect ratio
                # instead of letterboxing it in a map/table-sized tall layout.
                self.review_video_widget.setMinimumHeight(340)
                self.review_video_widget.setMaximumHeight(420)
            self.video_player = QMediaPlayer(self)
            self.video_audio = QAudioOutput(self)
            self.video_audio.setVolume(0.0)
            self.video_player.setAudioOutput(self.video_audio)
            self.video_player.setVideoOutput(self.review_video_widget)
            self.video_player.setSource(QUrl.fromLocalFile(str(self.review_video_path)))
            self.video_play_button = QPushButton("Play")
            self.video_play_button.clicked.connect(self._toggle_review_playback)
            if standalone_demo:
                self.video_stop_button = QPushButton("Stop")
                self.video_stop_button.clicked.connect(self._stop_standalone_demo)
            self.video_slider = QSlider(Qt.Orientation.Horizontal)
            self.video_slider.setRange(0, 0)
            self.video_slider.sliderPressed.connect(self._on_review_seek_started)
            self.video_slider.sliderReleased.connect(self._on_review_seek_finished)
            self.video_position_label = QLabel("0:00 / 3:00")
            self.video_player.durationChanged.connect(self._on_video_duration_changed)
            self.video_player.positionChanged.connect(self._on_video_position_changed)
            self.video_player.playbackStateChanged.connect(self._on_video_state_changed)
            self.video_player.errorOccurred.connect(self._on_review_video_error)
            playback_controls = QHBoxLayout()
            playback_controls.addWidget(self.video_play_button)
            if self.video_stop_button is not None:
                playback_controls.addWidget(self.video_stop_button)
            playback_controls.addWidget(self.video_slider, 1)
            playback_controls.addWidget(self.video_position_label)
        else:
            playback_controls = None
        body = QHBoxLayout()
        if standalone_demo:
            body.addWidget(self.review_video_widget, 1)
            layout = QVBoxLayout()
        elif self.show_map_with_video:
            video_panel = QVBoxLayout()
            video_heading = QLabel("CAM-02 (left)  |  CAM-05 (right) · rectified files, same frame index")
            video_heading.setStyleSheet("font-weight:700;padding:4px")
            video_panel.addWidget(video_heading)
            video_panel.addWidget(self.review_video_widget, 1)
            map_panel = QVBoxLayout()
            map_heading = QLabel("Entrance Room · illustrative map projection · EXPERIMENTAL / UNCALIBRATED")
            map_heading.setStyleSheet("font-weight:700;padding:4px")
            map_panel.addWidget(map_heading)
            self.map_calibration_status = QLabel(self.calibration_status)
            self.map_calibration_status.setWordWrap(True)
            self.map_calibration_status.setStyleSheet("background:#5a3512;color:#fff2d6;padding:5px")
            map_panel.addWidget(self.map_calibration_status)
            map_panel.addWidget(self.map_canvas, 1)
            body.addLayout(video_panel, 3)
            body.addLayout(map_panel, 2)
            self.table.setMinimumWidth(0)
            layout = QVBoxLayout()
        else:
            body.addWidget(self.review_video_widget or self.map_canvas, 3)
            body.addWidget(self.table, 2)
            layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(self.warning)
        layout.addWidget(self.status)
        if not standalone_demo:
            layout.addWidget(self.counters)
            layout.addWidget(self.review_status)
            layout.addWidget(self.alias_status)
            layout.addLayout(controls)
        if playback_controls is not None:
            layout.addLayout(playback_controls)
        layout.addLayout(body, 1 if standalone_demo else (3 if self.show_map_with_video else 1))
        if self.show_map_with_video:
            layout.addWidget(self.table, 2)
        elif not standalone_demo:
            layout.addWidget(self.table, 2)
        root = QWidget()
        root.setLayout(layout)
        root.setStyleSheet("QWidget{background:#17232c;color:#e9f1f5} QTableWidget{background:#202e38;gridline-color:#50616b} QHeaderView::section{background:#293b47;color:#f0f5f8;padding:6px}")
        self.setCentralWidget(root)

        self.socket = QWebSocket()
        self.socket.connected.connect(lambda: self.status.setText("Offline replay: connected to local staging API"))
        self.socket.disconnected.connect(self._on_disconnected)
        self.socket.textMessageReceived.connect(self._on_message)
        self.socket.errorOccurred.connect(self._on_socket_error)
        if standalone_demo:
            self.status.setText("Recorded DEMO ready · no RTSP, detector or live tracking process is running")
            QTimer.singleShot(0, self._play_standalone_demo)
        elif auto_start:
            QTimer.singleShot(0, self.replay)

    def replay(self) -> None:
        if self.standalone_demo:
            self._play_standalone_demo(restart=True)
            return
        if self.show_map_with_video:
            self._begin_video_master_replay()
            return
        if self.review_video_path is not None and self.video_player is not None:
            self.video_player.pause()
            self.video_player.setPosition(0)
        self._start_replay_stream(start_frame=0, clear=True,
                                  start_video=self.review_video_path is not None,
                                  seek_video=self.review_video_path is not None,
                                  reset_counts=True)

    def _start_replay_stream(self, *, start_frame: int, clear: bool,
                             start_video: bool, seek_video: bool,
                             reset_counts: bool = False, end_frame: int | None = None) -> None:
        self._replay_completed = True
        self.socket.abort()
        self._replay_completed = False
        self.current_frame = max(0, int(start_frame))
        if reset_counts:
            self.received_frame_batches = 0
            self.received_observations = 0
        if clear:
            self.active = []
            self.visual_review = None
            self.map_canvas.reset_trails()
            self.map_canvas.set_observations([], separate_by_camera=self.camera_scoped_ids)
            self._render_table()
            self._render_counts()
        self._play_video_on_stream_start = start_video
        self._seek_video_on_stream_start = seek_video
        self._replay_completed = False
        self.last_camera_frame = {}
        self.status.setText("Offline replay: connecting…")
        separator = "&" if "?" in self.ws_url else "?"
        replay_url = f"{self.ws_url}{separator}start_frame={max(0, int(start_frame))}"
        if end_frame is not None:
            replay_url += f"&end_frame={max(0, int(end_frame))}"
        self.socket.open(QUrl(replay_url))

    def _toggle_review_playback(self) -> None:
        if self.video_player is None:
            return
        if self.standalone_demo:
            if self.video_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self.video_player.pause()
            else:
                self.video_player.play()
            return
        if self.show_map_with_video:
            if self.video_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self.video_player.pause()
                self._frame_timer.stop()
            else:
                self._request_video_timeline_frame(force=True)
                self.video_player.play()
                self._frame_timer.start()
            return
        if self.video_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.video_player.pause()
            self._stop_replay_stream()
            return
        if self.socket.state() != QAbstractSocket.SocketState.ConnectedState:
            if self.last_replay_frame_id is not None and self.current_frame < self.last_replay_frame_id:
                position_ms = self.video_player.position()
                resume_frame = max(0, int((position_ms / 1000.0) * self.nominal_fps))
                self._start_replay_stream(start_frame=resume_frame, clear=True,
                                          start_video=True, seek_video=True)
            else:
                self.video_player.play()
        else:
            self.video_player.play()

    def _stop_replay_stream(self) -> None:
        self._replay_completed = True
        self._play_video_on_stream_start = False
        self._seek_video_on_stream_start = False
        if self.socket.state() != QAbstractSocket.SocketState.UnconnectedState:
            self.socket.abort()

    def _on_review_seek_started(self) -> None:
        if self.video_player is None:
            return
        self._resume_after_seek = self.video_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        self.video_player.pause()
        if self.show_map_with_video:
            self._frame_timer.stop()
            return
        self._stop_replay_stream()

    def _on_review_seek_finished(self) -> None:
        if self.video_player is None or self.video_slider is None:
            return
        target_ms = self.video_slider.value()
        self.video_player.setPosition(target_ms)
        if self.standalone_demo:
            if self._resume_after_seek:
                self.video_player.play()
            return
        if self.show_map_with_video:
            self._request_video_timeline_frame(force=True)
            if self._resume_after_seek:
                self.video_player.play()
                self._frame_timer.start()
            return
        target_frame = int((target_ms / 1000.0) * self.nominal_fps)
        if self.last_replay_frame_id is not None:
            target_frame = min(target_frame, self.last_replay_frame_id)
        self._start_replay_stream(start_frame=target_frame, clear=True,
                                  start_video=self._resume_after_seek,
                                  seek_video=True, reset_counts=False,
                                  end_frame=None if self._resume_after_seek else target_frame)

    def _on_video_duration_changed(self, duration_ms: int) -> None:
        if self.video_slider is not None:
            self.video_slider.setRange(0, max(0, duration_ms))
        self._update_video_time_label(0)
        if duration_ms > 0:
            print(f"MV3DT review video ready: duration_ms={duration_ms}", flush=True)

    def _on_review_video_error(self, error, error_string: str) -> None:
        if self.video_position_label is not None:
            self.video_position_label.setText("Video error")
        self.status.setText(f"Review video error: {error_string}")
        print(f"MV3DT review video error: {error} {error_string}", flush=True)

    def _play_standalone_demo(self, *, restart: bool = False) -> None:
        if self.video_player is None:
            return
        if restart:
            self.video_player.setPosition(0)
        self.video_player.play()

    def _stop_standalone_demo(self) -> None:
        if self.video_player is None:
            return
        self.video_player.pause()
        self.video_player.setPosition(0)
        if self.video_play_button is not None:
            self.video_play_button.setText("Play")
        self.status.setText("Demo stopped · press Play to restart or close the window")

    def _on_video_position_changed(self, position_ms: int) -> None:
        if self.video_slider is not None and not self.video_slider.isSliderDown():
            self.video_slider.setValue(position_ms)
        self._update_video_time_label(position_ms)
        if self.show_map_with_video and not self._replay_completed:
            self.status.setText(self._status_text())

    def _begin_video_master_replay(self) -> None:
        if self.video_player is None or self._network is None:
            return
        self.video_player.pause()
        self._frame_timer.stop()
        self.socket.abort()
        self.active = []
        self.visual_review = None
        self.current_frame = 0
        self._replay_completed = False
        self.last_camera_frame = {}
        self.map_canvas.reset_trails()
        self.map_canvas.set_observations([], separate_by_camera=True)
        self._render_table()
        self._render_counts()
        self.status.setText("Offline video-master replay: loading frame index and map metadata…")
        self._frame_mode_autoplay = True
        request = QNetworkRequest(QUrl(f"{self.http_base_url}/api/v1/summary"))
        reply = self._network.get(request)
        reply.finished.connect(lambda current=reply: self._on_video_master_summary(current))

    def _on_video_master_summary(self, reply) -> None:
        try:
            payload = json.loads(bytes(reply.readAll()).decode("utf-8"))
            frame_range = payload.get("frame_id_range")
            map_details = payload.get("map") or {}
            transform = map_details.get("transform_world_to_pixel_row_major")
            if not isinstance(frame_range, list) or len(frame_range) != 2:
                raise ValueError("offline summary lacks a frame range")
            if not isinstance(transform, list) or len(transform) != 9:
                raise ValueError("offline summary lacks a valid map transform")
            self._frame_mode_first_frame, self._frame_mode_last_frame = map(int, frame_range)
            if self._frame_mode_first_frame > self._frame_mode_last_frame:
                raise ValueError("offline summary frame range is invalid")
            self.nominal_fps = float(payload.get("nominal_fps") or 30.0)
            self.transform = transform
            self.map_canvas.set_transform(transform)
            self._frame_mode_initialized = True
            counts = map_details.get("projected_observation_counts", {})
            outside = counts.get("outside_map")
            self.warning.setText(
                "EXPERIMENTAL / UNCALIBRATED — camera-scoped MV3DT output IDs stay separate. "
                "Illustrative projected foot positions only; map scale and pairwise geometry are unverified. "
                "Video and map use normalized file-frame indices, not a common capture clock; source PTS is absent."
                + (f" {outside} logged positions fell outside map bounds and are not drawn."
                   if isinstance(outside, int) and outside else "")
            )
            self.status.setText("Offline video-master replay · requesting map rows for the video frame index")
            self.video_player.setPosition(round(self._frame_mode_first_frame * 1000.0 / self.nominal_fps))
            self._request_video_timeline_frame(force=True)
            if self._frame_mode_autoplay:
                self.video_player.play()
                self._frame_timer.start()
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.status.setText(f"Offline preview metadata error: {exc}")
            self._replay_completed = True
        finally:
            reply.deleteLater()

    def _video_timeline_frame(self) -> int:
        if self.video_player is None or self._frame_mode_last_frame < self._frame_mode_first_frame:
            return self._frame_mode_first_frame
        frame = int((self.video_player.position() / 1000.0) * self.nominal_fps)
        return max(self._frame_mode_first_frame, min(self._frame_mode_last_frame, frame))

    def _request_video_timeline_frame(self, *, force: bool = False) -> None:
        if not self.show_map_with_video or not self._frame_mode_initialized or self._network is None:
            return
        target_frame = self._video_timeline_frame()
        if not force and target_frame == self.current_frame:
            return
        if self._frame_reply is not None:
            self._pending_exact_frame = target_frame
            return
        self._pending_exact_frame = None
        self._frame_reply = self._network.get(
            QNetworkRequest(QUrl(f"{self.http_base_url}/api/v1/frames/{target_frame}"))
        )
        reply = self._frame_reply
        reply.finished.connect(lambda current=reply, requested=target_frame:
                               self._on_video_timeline_frame(current, requested))

    def _on_video_timeline_frame(self, reply, requested_frame: int) -> None:
        if self._frame_reply is reply:
            self._frame_reply = None
        try:
            payload = json.loads(bytes(reply.readAll()).decode("utf-8"))
            returned_frame = int(payload.get("frame_id", -1))
            if returned_frame != requested_frame:
                raise ValueError(f"requested frame {requested_frame}, API returned frame {returned_frame}")
            current_video_frame = self._video_timeline_frame()
            if (self.video_player is not None
                    and self.video_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
                    and requested_frame != current_video_frame):
                self._pending_exact_frame = current_video_frame
                return
            self.current_frame = returned_frame
            self.visual_review = payload.get("visual_review")
            frame_messages = payload.get("frame_messages", [])
            self.last_camera_frame = {str(item["camera_id"]): int(item["frame_id"])
                                      for item in frame_messages}
            observations = payload.get("observations", [])
            self.active = update_active_observations([], observations,
                                                     current_frame=returned_frame, expiry_frames=0)
            self.map_canvas.set_observations(
                self.active, current_frame=returned_frame, visual_review=self.visual_review,
                use_display_aliases=False, separate_by_camera=True,
            )
            self._render_table()
            self._render_counts()
            self.status.setText(self._status_text())
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.status.setText(f"Offline frame-index mismatch/error: {exc}")
        finally:
            reply.deleteLater()
            pending = self._pending_exact_frame
            self._pending_exact_frame = None
            if pending is not None and pending != self.current_frame:
                self._request_video_timeline_frame(force=True)

    def _update_video_time_label(self, position_ms: int) -> None:
        if self.video_position_label is None or self.video_player is None:
            return
        duration_ms = max(0, self.video_player.duration())
        def stamp(value: int) -> str:
            seconds = max(0, value // 1000)
            return f"{seconds // 60}:{seconds % 60:02d}"
        self.video_position_label.setText(f"{stamp(position_ms)} / {stamp(duration_ms)}")

    def _on_video_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if self.video_play_button is not None:
            self.video_play_button.setText("Pause" if state == QMediaPlayer.PlaybackState.PlayingState else "Play")
        if self.standalone_demo:
            self.status.setText(
                "DEMO playing · recorded only · no RTSP or inference is running"
                if state == QMediaPlayer.PlaybackState.PlayingState
                else "Recorded DEMO paused · no RTSP or inference is running"
            )

    def clear_markers(self) -> None:
        self.active = []
        self.visual_review = None
        self.map_canvas.reset_trails()
        self.map_canvas.set_observations([], separate_by_camera=self.camera_scoped_ids)
        self._render_table()
        self._render_counts()

    def _toggle_observations(self, enabled: bool) -> None:
        self.map_canvas.show_camera_observations = enabled
        self.map_canvas.update()

    def _toggle_trails(self, enabled: bool) -> None:
        self.map_canvas.show_trails = enabled
        self.map_canvas.update()

    def _toggle_map_labels(self, enabled: bool) -> None:
        self.map_canvas.show_marker_labels = enabled
        self.map_canvas.update()

    def _on_alias_toggle(self, enabled: bool) -> None:
        self.map_canvas.set_observations(self.active, current_frame=self.current_frame,
                                         visual_review=self.visual_review,
                                         use_display_aliases=enabled,
                                         separate_by_camera=self.camera_scoped_ids)
        self._render_table()
        self._render_counts()

    def _on_disconnected(self) -> None:
        if not self._replay_completed:
            self.status.setText(f"Offline replay disconnected at frame {self.current_frame}")

    def _on_socket_error(self, _error) -> None:
        if not self._replay_completed:
            self.status.setText(f"Offline replay API error: {self.socket.errorString()}")

    def _on_message(self, payload: str) -> None:
        try:
            message = json.loads(payload)
        except json.JSONDecodeError:
            self.status.setText("Offline replay API sent invalid JSON")
            return
        kind = message.get("type")
        if kind == "start":
            map_details = message.get("map", {})
            self.transform = map_details.get("transform_world_to_pixel_row_major") or message.get(
                "transform_world_to_pixel_row_major"
            )
            self.map_canvas.set_transform(self.transform)
            self.display_aliases_available = (
                bool(message.get("display_aliasing_available", False)) and not self.camera_scoped_ids
            )
            self.alias_toggle.setEnabled(self.display_aliases_available)
            alias_count = int(message.get("display_alias_count", 0) or 0)
            if self.camera_scoped_ids:
                self.alias_status.setText(
                    "Disabled for this preview: camera-scoped labels stay separate; cross-camera adoption aliases are not applied."
                )
            elif self.display_aliases_available:
                self.alias_status.setText(
                    f"{alias_count} hash-bound ADOPTED_ID event(s) available. Aliases are retrospective for this "
                    "offline replay only; raw MV3DT IDs, timestamps, boxes and map observations are unchanged. "
                    "They do not prove physical identity correctness."
                )
                if not self._alias_toggle_initialized and not self.camera_scoped_ids:
                    self.alias_toggle.setChecked(True)
                    self._alias_toggle_initialized = True
            else:
                self.alias_status.setText("No source-hash-bound ADOPTED_ID aliases are configured.")
            counts = map_details.get("projected_observation_counts", {})
            outside = counts.get("outside_map")
            if isinstance(outside, int) and outside:
                self.warning.setText(
                    self.warning.text() + f" {outside} logged projected observations fall outside the map image and are not drawn."
                )
            self.status.setText(
                f"Offline replay · nominal {message.get('nominal_fps')} FPS · frame-index timeline only · "
                "source synchronization is not established"
            )
            self.nominal_fps = float(message.get("nominal_fps") or 30.0)
            frame_range = message.get("frame_id_range") or [0, 0]
            self.last_replay_frame_id = int(frame_range[-1])
            if self.video_player is not None:
                if self._seek_video_on_stream_start:
                    first_frame = int(frame_range[0])
                    self.video_player.setPosition(round(first_frame * 1000.0 / self.nominal_fps))
                if self._play_video_on_stream_start:
                    self.video_player.play()
                self._seek_video_on_stream_start = False
                self._play_video_on_stream_start = False
        elif kind == "frame":
            self.current_frame = int(message["frame_id"])
            self.visual_review = message.get("visual_review")
            self.received_frame_batches += 1
            self.received_observations += len(message.get("observations", []))
            for frame_message in message.get("frame_messages", []):
                self.last_camera_frame[str(frame_message["camera_id"])] = self.current_frame
            self.active = update_active_observations(
                self.active, message.get("observations", []), current_frame=self.current_frame, expiry_frames=30
            )
            self.map_canvas.set_observations(self.active, current_frame=self.current_frame,
                                             visual_review=self.visual_review,
                                             use_display_aliases=self.alias_toggle.isChecked(),
                                             separate_by_camera=self.camera_scoped_ids)
            self._render_table()
            self._render_counts()
            self.status.setText(self._status_text())
        elif kind == "complete":
            self._replay_completed = True
            state = display_state(self.active, current_frame=self.current_frame, visual_review=self.visual_review,
                                  use_display_aliases=self.alias_toggle.isChecked(),
                                  separate_by_camera=self.camera_scoped_ids)
            self.status.setText(
                f"Replay complete at frame {message.get('last_frame_id')}; markers and counters describe that saved frame. "
                "Recent observations remain available in the table."
            )
            print(
                "MV3DT staging Qt replay complete: "
                f"last_frame={message.get('last_frame_id')} "
                f"frame_batches={self.received_frame_batches} "
                f"object_observations={self.received_observations} "
                f"active_camera_observations={len(self.active)} "
                f"unique_global_ids={state['unique_global_ids']} "
                f"primary_markers={len(state['primary_markers'])} "
                f"confirmed_visible_people={state['confirmed_visible_people']}",
                flush=True,
            )
            self.replay_complete.emit()

    def _status_text(self) -> str:
        cameras = " · ".join(
            f"{camera}: frame {frame}" for camera, frame in sorted(self.last_camera_frame.items())
        ) or "no camera event yet"
        message = (f"Offline replay frame {self.current_frame} ({self.current_frame / 30.0:.2f}s nominal) · "
                   f"{cameras}")
        if self.video_player is not None:
            video_frame = int((self.video_player.position() / 1000.0) * self.nominal_fps)
            message += (f" · video nominal frame {video_frame} · event/video frame delta "
                        f"{self.current_frame - video_frame:+d}")
        return message

    def _render_counts(self) -> None:
        state = display_state(self.active, current_frame=self.current_frame, visual_review=self.visual_review,
                              use_display_aliases=self.alias_toggle.isChecked(),
                              separate_by_camera=self.camera_scoped_ids)
        confirmed = state["confirmed_visible_people"]
        confirmed_text = f"{confirmed} (exact-frame visual review)" if confirmed is not None else "UNAVAILABLE"
        if self.camera_scoped_ids:
            self.counters.setText(
                f"Current camera observations: {state['camera_observations']} · "
                f"Camera-scoped MV3DT output labels: {state['camera_scoped_track_labels']} · "
                "Cross-camera merge: OFF · Confirmed people: UNAVAILABLE"
                + (f" · Recent observations: {state['recent_observations']} (excluded)"
                   if state["recent_observations"] else "")
            )
        else:
            self.counters.setText(
            f"Camera observations: {state['camera_observations']} · Unique global IDs: {state['unique_global_ids']} (raw) · "
            f"Display IDs: {state['display_unique_ids']} ({'ADOPTED_ID aliases ON' if state['display_aliases_enabled'] else 'raw labels'}) · "
            f"Confirmed visible people: {confirmed_text} · Primary markers: {len(state['primary_markers'])}"
            + (f" · Recent observations: {state['recent_observations']} (excluded from current counts)"
               if state["recent_observations"] else "")
            )
        if state["visual_review"]:
            aliases = ", ".join(f"ID {alias} → ID {primary}" for alias, primary in state["duplicate_global_ids"].items())
            self.review_status.setText(f"Reviewed frame {self.current_frame}: {aliases} is a duplicate track of the same visible person. "
                                      "All output IDs and camera rows are retained below; observation points can be inspected.")
            self.review_status.setToolTip(state["visual_review"]["note"])
        elif state["visual_review_suppressed_reason"]:
            self.review_status.setText(state["visual_review_suppressed_reason"])
        elif state["display_aliases_enabled"]:
            self.review_status.setText(
                "Markers are grouped by the selected display ID using source-hash-bound ADOPTED_ID events. "
                f"{self._suppressed_adoption_rows} superseded pre-adoption raw row(s) are omitted from the active "
                "table when their adopted rows are present; all raw observations remain in the offline event "
                "store/API. Event-derived aliases do not certify identity."
            )
        elif self.camera_scoped_ids:
            self.review_status.setText(
                "Each marker is camera-scoped and displays a raw MV3DT output ID. "
                "These are not local NvDCF IDs or verified person identities; same-number IDs in different cameras stay separate."
            )
        else:
            self.review_status.setText("One primary marker per current MV3DT output ID. Different IDs remain separate until "
                                      "an exact-frame visual review establishes a duplicate. Confirmed people: unavailable.")

    def _render_table(self) -> None:
        use_aliases = self.alias_toggle.isChecked()
        state = display_state(self.active, current_frame=self.current_frame, visual_review=self.visual_review,
                              use_display_aliases=use_aliases,
                              separate_by_camera=self.camera_scoped_ids)
        visible_rows = list(self.active)
        if use_aliases:
            current_display_keys = {
                (str(row["camera_id"]), str(row.get("display_global_id", row.get("raw_global_id", row["mv3dt_global_id"]))))
                for row in self.active if int(row["frame_id"]) == self.current_frame
            }
            retained = []
            for row in visible_rows:
                raw_id = str(row.get("raw_global_id", row["mv3dt_global_id"]))
                display_id = str(row.get("display_global_id", raw_id))
                lineage = row.get("display_alias_lineage", [])
                adopted = any(int(item.get("adoption_frame", self.current_frame + 1)) <= self.current_frame
                              for item in lineage)
                superseded = (int(row["frame_id"]) < self.current_frame and adopted
                              and (str(row["camera_id"]), display_id) in current_display_keys)
                if not superseded:
                    retained.append(row)
            self._suppressed_adoption_rows = len(visible_rows) - len(retained)
            visible_rows = retained
        else:
            self._suppressed_adoption_rows = 0
        self.table.setRowCount(len(visible_rows))
        for index, row in enumerate(visible_rows):
            position = row.get("world_position_unverified_units")
            world_text = (f"({float(position[0]):.3f}, {float(position[1]):.3f})"
                          if isinstance(position, list) and len(position) == 2 else "unavailable")
            raw_id = str(row.get("raw_global_id", row["mv3dt_global_id"]))
            event_lineage = row.get("display_alias_lineage", [])
            display_id = (f"{row['camera_id']} / {raw_id}" if self.camera_scoped_ids else
                          (str(row.get("display_global_id", raw_id)) if use_aliases else raw_id))
            alias_of = state["duplicate_global_ids"].get(raw_id)
            if alias_of is not None:
                scope = f"visually reviewed duplicate of raw ID {alias_of}"
            elif use_aliases and row.get("display_alias_applied"):
                provenance = ", ".join(f"ADOPTED_ID@{item['adoption_frame']}" for item in event_lineage)
                scope = f"display alias ({provenance}); raw ID retained"
            elif self.display_aliases_available and row.get("display_alias_applied"):
                scope = f"alias available: {raw_id} → {row.get('display_global_id')} (off)"
            else:
                scope = ("current" if int(row["frame_id"]) == self.current_frame else
                         f"recent, {self.current_frame - int(row['frame_id'])} frames old")
            nominal = row.get("media_time_seconds_nominal")
            nominal_text = f"{float(row['frame_id'])} / {float(nominal):.2f}s" if nominal is not None else str(row["frame_id"])
            source_id = row.get("source_id", "unavailable")
            if self.camera_scoped_ids:
                scope = "source-scoped MV3DT output; not NvDCF local ID or cross-camera identity"
            values = [row["camera_id"], source_id, raw_id, display_id, nominal_text, world_text, scope]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setToolTip(
                    "Display ID is an optional presentation label derived from documented ADOPTED_ID events; "
                    "it never replaces the raw tracker/MV3DT observation and does not establish real-person identity. "
                    f"Camera/source: {row['camera_id']}/{source_id}; raw ID: {raw_id}; display ID: {display_id}. "
                    f"Frame: {row.get('frame_id')}; nominal media time: {nominal}; "
                    f"event publication timestamp: {row.get('event_timestamp', 'unavailable')}; "
                    f"bbox: {row.get('bbox')}; foot 2D: {row.get('foot_location_2d_px')}; "
                    f"world units (unverified): {row.get('world_position_unverified_units')}; "
                    f"alias lineage: {event_lineage}. Observation scope: {scope}. "
                    "The normalized Kafka log has no original source PTS."
                )
                self.table.setItem(index, column, item)
        self.table.resizeRowsToContents()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._frame_timer.stop()
        if self.standalone_demo and self.video_player is not None:
            self.video_player.stop()
        if self._frame_reply is not None:
            self._frame_reply.abort()
        self.socket.close()
        super().closeEvent(event)


def main() -> None:
    map_path = os.getenv("MV3DT_STAGING_MAP")
    if not map_path or not Path(map_path).is_file():
        raise SystemExit("Set MV3DT_STAGING_MAP to the preserved Dev Room map.png")
    api_url = os.getenv("MV3DT_STAGING_WS", "ws://127.0.0.1:18120/ws/replay")
    review_video = os.getenv("MV3DT_STAGING_REVIEW_VIDEO")
    show_map_with_video = os.getenv("MV3DT_STAGING_SHOW_MAP_WITH_VIDEO", "0") == "1"
    standalone_demo = os.getenv("MV3DT_STAGING_STANDALONE_DEMO", "0") == "1"
    if standalone_demo and not review_video:
        raise SystemExit("standalone demo requires MV3DT_STAGING_REVIEW_VIDEO")
    if standalone_demo and show_map_with_video:
        raise SystemExit("standalone demo video already embeds the two cameras and map; disable separate API map mode")
    default_speed = "1" if review_video and show_map_with_video else "10"
    speed = float(os.getenv("MV3DT_STAGING_SPEED", default_speed))
    if not 0.1 <= speed <= 100:
        raise SystemExit("MV3DT_STAGING_SPEED must be in 0.1..100")
    separator = "&" if "?" in api_url else "?"
    api_url = f"{api_url}{separator}speed={speed:g}"
    app = QApplication(sys.argv)
    room_title = os.getenv("MV3DT_STAGING_ROOM_TITLE", "Dev Room")
    window = MainWindow(map_path, api_url, room_title=room_title,
                        review_video_path=review_video or None,
                        camera_scoped_ids=os.getenv("MV3DT_STAGING_CAMERA_SCOPED_IDS", "0") == "1",
                        show_map_with_video=show_map_with_video,
                        calibration_status=os.getenv(
                            "MV3DT_STAGING_CALIBRATION_STATUS",
                            "CALIBRATION: UNVERIFIED — map scale and independent floor-point accuracy are not established.",
                        ), standalone_demo=standalone_demo, auto_start=not standalone_demo)
    window.trail_toggle.setChecked(os.getenv("MV3DT_STAGING_SHOW_TRAILS", "0") == "1")
    if standalone_demo:
        window.resize(1840, 620)
        screen = app.primaryScreen()
        if screen is not None:
            bounds = screen.availableGeometry()
            window.resize(min(window.width(), bounds.width()), min(window.height(), bounds.height()))
            window.move(bounds.x() + (bounds.width() - window.width()) // 2,
                        bounds.y() + (bounds.height() - window.height()) // 2)
        window.show()
    else:
        window.showMaximized()
    if os.getenv("MV3DT_STAGING_AUTOCLOSE", "0") == "1":
        window.replay_complete.connect(lambda: QTimer.singleShot(250, app.quit))
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()

