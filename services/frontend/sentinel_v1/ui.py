from __future__ import annotations

# The Sentinel UI source is stored in line-preserving parts so the exact supplied
# interface can be carried without dropping pages/controls while cameras are wired
# incrementally. The parts are concatenated and executed as this module.
from pathlib import Path as _Path
import os as _os
import time as _time

_parts_dir = _Path(__file__).with_name("ui_parts")
_source = "".join(path.read_text(encoding="utf-8") for path in sorted(_parts_dir.glob("part_*.pyfrag")))
exec(compile(_source, str(_parts_dir / "sentinel_ui_combined.py"), "exec"), globals(), globals())

from services.frontend.sentinel_v1.monitoring_client_v1 import MonitoringTelemetryClient

# Keep the supplied page/widget construction untouched. CAM-01 and CAM-02 become
# real at the CameraView boundary itself, so Monitoring/fullscreen/expand views
# consume the same latest-only previews without opening another RTSP session.
LIVE_PREVIEW_CAMERAS = tuple(
    dict.fromkeys(
        part.strip()
        for part in _os.environ.get(
            "SENTINEL_LIVE_PREVIEW_CAMERAS",
            _os.environ.get("SENTINEL_CAMERA_IDS", "CAM-01,CAM-02"),
        ).split(",")
        if part.strip()
    )
)
MONITORING_REALTIME = _os.environ.get("SENTINEL_MONITORING_REALTIME", "0").strip().lower() in {
    "1", "true", "yes", "on"
}
MONITORING_WS_URL = _os.environ.get(
    "SENTINEL_MONITORING_WS_URL", "ws://127.0.0.1:8000/ws/v1/monitoring"
)
_camera_view_init = CameraView.__init__


def _preview_path_for_camera(camera_id: str, _env=_os.environ) -> str:
    env_key = f"V11_UI_PREVIEW_PATH_{camera_id.upper().replace('-', '')}"
    slug = camera_id.lower().replace("-", "")
    return _env.get(env_key, f"/dev/shm/v11_ui_preview_{slug}_v1.bin")


def _camera_view_init_staged_live(
    self,
    camera,
    people=None,
    show_room=True,
    occupancy=None,
    display_name=None,
    heatmap_enabled=False,
    realtime_camera_id=None,
    parent=None,
):
    camera_id = getattr(camera, "id", None)
    selected_live_id = realtime_camera_id
    if selected_live_id is None and camera_id in LIVE_PREVIEW_CAMERAS:
        selected_live_id = camera_id

    _camera_view_init(
        self,
        camera,
        people=people,
        show_room=show_room,
        occupancy=occupancy,
        display_name=display_name,
        heatmap_enabled=heatmap_enabled,
        realtime_camera_id=None,
        parent=parent,
    )

    if selected_live_id is None:
        return

    self.realtime_camera_id = selected_live_id
    self.live_reader = PreviewFrameReader(_preview_path_for_camera(selected_live_id))
    self.live_image = QImage()
    self.live_sequence = -1
    self.live_last_frame_mono = 0.0
    self.live_object_count = getattr(self.camera, "current_box_count", None) if MONITORING_REALTIME else 0
    if not MONITORING_REALTIME:
        self.camera.online = False
        self.camera.fps = 0.0
        self.camera.last_error = "Ulanish kutilmoqda"
    self.live_timer = QTimer(self)
    self.live_timer.setInterval(33)
    self.live_timer.timeout.connect(self._poll_live_preview)
    self.live_timer.start()


def _camera_view_poll_live_preview_fresh(self):
    if self.live_reader is None:
        return
    now = _time.monotonic()
    frame = self.live_reader.read_latest()
    if frame is not None and frame.sequence != self.live_sequence:
        image = QImage(
            frame.payload, frame.width, frame.height, frame.stride, QImage.Format_RGB32
        ).copy()
        if not image.isNull():
            self.live_image = image
            self.live_sequence = frame.sequence
            self.live_last_frame_mono = now
            if not MONITORING_REALTIME:
                self.live_object_count = frame.object_count
                self.camera.online = True
                self.camera.fps = frame.fps
                self.camera.last_error = None
            self.update()
            return
    if self.live_last_frame_mono <= 0.0 or (now - self.live_last_frame_mono) > 1.20:
        changed = not self.live_image.isNull()
        self.live_image = QImage()
        self.live_sequence = -1
        if not MONITORING_REALTIME:
            changed = changed or self.camera.online
            self.live_object_count = 0
            self.camera.online = False
            self.camera.fps = 0.0
            self.camera.last_error = "Ulanish kutilmoqda"
        if changed:
            self.update()


def _camera_view_draw_live_frame_qsize_safe(self, painter, rect):
    if self.live_image.isNull():
        return False
    target_size = rect.size()
    if hasattr(target_size, "toSize"):
        target_size = target_size.toSize()
    scaled = self.live_image.scaled(
        target_size,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation,
    )
    if scaled.isNull():
        return False
    painter.fillRect(rect, QColor("#05080b"))
    x = rect.left() + (rect.width() - scaled.width()) / 2.0
    y = rect.top() + (rect.height() - scaled.height()) / 2.0
    painter.drawImage(QRectF(x, y, scaled.width(), scaled.height()), scaled)
    return True


CameraView.__init__ = _camera_view_init_staged_live
CameraView._poll_live_preview = _camera_view_poll_live_preview_fresh
CameraView._draw_live_frame = _camera_view_draw_live_frame_qsize_safe


def _metric_number(value, default=None):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _camera_error(row, online):
    if not online:
        return "Source stopped"
    errors = []
    for key, title in (
        ("pipeline_errors", "pipeline"),
        ("infer_errors", "infer"),
        ("meta_errors", "meta"),
        ("copy_errors", "copy"),
        ("preview_errors", "preview"),
    ):
        value = _metric_number(row.get(key), 0)
        if value and value > 0:
            errors.append(f"{title}={int(value)}")
    return ", ".join(errors) or None


def _update_monitoring_views(window):
    for view in window.findChildren(CameraView):
        view.live_object_count = getattr(view.camera, "current_box_count", None)
        view.setToolTip(
            "source={source} fps · render={render} fps · infer={infer} Hz · "
            "queue={queue} · drops={drops} · result_age={age} ms".format(
                source="—" if view.camera.source_fps is None else f"{view.camera.source_fps:.1f}",
                render="—" if view.camera.render_fps is None else f"{view.camera.render_fps:.1f}",
                infer="—" if view.camera.infer_hz is None else f"{view.camera.infer_hz:.1f}",
                queue="—" if view.camera.queue is None else view.camera.queue,
                drops="—" if view.camera.dropped is None else view.camera.dropped,
                age="—" if view.camera.latency is None else view.camera.latency,
            )
        )
        view.update()


def _main_window_apply_monitoring_snapshot(self, payload):
    status = payload.get("telemetry_status", "offline")
    runtime_status = payload.get("runtime", {}).get("status")
    fresh = status == "fresh" and runtime_status == "live"
    rows = {row.get("camera_id"): row for row in payload.get("cameras", [])}
    detector_enabled = payload.get("runtime", {}).get("detector_enabled")
    visible = 0
    for camera in CAMERAS:
        row = rows.get(camera.id, {})
        camera.telemetry_status = status
        camera.detector_enabled = detector_enabled if isinstance(detector_enabled, bool) else None
        if fresh:
            camera.online = bool(row.get("online"))
            camera.source_fps = _metric_number(row.get("source_fps"))
            camera.render_fps = _metric_number(row.get("render_fps"))
            camera.infer_hz = _metric_number(row.get("infer_hz"))
            camera.fps = camera.render_fps or 0.0
            camera.queue = _metric_number(row.get("queue_depth"))
            camera.dropped = _metric_number(row.get("detector_drops"))
            camera.reconnects = _metric_number(row.get("reconnects"))
            age = _metric_number(row.get("result_age_ms"))
            camera.latency = int(round(age)) if age is not None else None
            camera.current_box_count = _metric_number(row.get("current_box_count"))
            if camera.current_box_count is not None:
                visible += int(camera.current_box_count)
            camera.last_error = _camera_error(row, camera.online)
        else:
            camera.online = False
            camera.fps = 0.0
            camera.source_fps = None
            camera.render_fps = None
            camera.infer_hz = None
            camera.queue = None
            camera.dropped = None
            camera.reconnects = None
            camera.latency = None
            camera.current_box_count = None
            if runtime_status == "degraded":
                camera.last_error = "ML service unavailable"
            else:
                camera.last_error = "Telemetry stale" if status == "stale" else "Telemetry offline"
    monitoring = self.pages[0]
    monitoring.total_people_value.setText(str(visible) if fresh else "—")
    monitoring.known_people_value.setText("—")
    monitoring.unknown_people_value.setText("—")
    monitoring.recent_status_value.setText(f"TELEMETRY {status.upper()}")
    if self.stack.currentIndex() == 0:
        online = sum(camera.online for camera in CAMERAS)
        self.subtitle.setText(f"{len(CAMERAS)} ta kamera · {online} online · realtime telemetry")
    _update_monitoring_views(self)


def _main_window_monitoring_status(self, status):
    if status not in {"disconnected", "stale", "offline", "invalid"}:
        return
    for camera in CAMERAS:
        camera.online = False
        camera.fps = 0.0
        camera.source_fps = None
        camera.render_fps = None
        camera.infer_hz = None
        camera.queue = None
        camera.dropped = None
        camera.reconnects = None
        camera.latency = None
        camera.current_box_count = None
        camera.telemetry_status = status
        camera.last_error = "API disconnected" if status == "disconnected" else f"Telemetry {status}"
    monitoring = self.pages[0]
    monitoring.total_people_value.setText("—")
    monitoring.known_people_value.setText("—")
    monitoring.unknown_people_value.setText("—")
    monitoring.recent_status_value.setText(f"TELEMETRY {status.upper()}")
    _update_monitoring_views(self)


_main_window_init = MainWindow.__init__
_main_window_close_event = MainWindow.closeEvent


def _main_window_init_realtime(self):
    _main_window_init(self)
    self.monitoring_client = None
    if MONITORING_REALTIME:
        self.monitoring_client = MonitoringTelemetryClient(
            MONITORING_WS_URL, [camera.id for camera in CAMERAS], self
        )
        self.monitoring_client.snapshotChanged.connect(self._apply_monitoring_snapshot)
        self.monitoring_client.statusChanged.connect(self._monitoring_status_changed)
        self.monitoring_client.start()


def _main_window_close_realtime(self, event):
    if self.monitoring_client is not None:
        self.monitoring_client.stop()
    _main_window_close_event(self, event)


MainWindow.__init__ = _main_window_init_realtime
MainWindow._apply_monitoring_snapshot = _main_window_apply_monitoring_snapshot
MainWindow._monitoring_status_changed = _main_window_monitoring_status
MainWindow.closeEvent = _main_window_close_realtime

del _source, _parts_dir, _Path, _os
