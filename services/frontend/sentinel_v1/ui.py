from __future__ import annotations

# The Sentinel UI source is stored in line-preserving parts so the exact supplied
# interface can be carried without dropping pages/controls while cameras are wired
# incrementally. The parts are concatenated and executed as this module.
from pathlib import Path as _Path
import os as _os

_parts_dir = _Path(__file__).with_name("ui_parts")
_source = "".join(path.read_text(encoding="utf-8") for path in sorted(_parts_dir.glob("part_*.pyfrag")))
exec(compile(_source, str(_parts_dir / "sentinel_ui_combined.py"), "exec"), globals(), globals())

# Keep the supplied page/widget construction untouched. CAM-01 and CAM-02 become
# real at the CameraView boundary itself, so Monitoring/fullscreen/expand views
# consume the same latest-only previews without opening another RTSP session.
LIVE_PREVIEW_CAMERAS = tuple(
    dict.fromkeys(
        part.strip()
        for part in _os.environ.get("SENTINEL_LIVE_PREVIEW_CAMERAS", "CAM-01,CAM-02").split(",")
        if part.strip()
    )
)
_camera_view_init = CameraView.__init__


def _preview_path_for_camera(camera_id: str) -> str:
    env_key = f"V11_UI_PREVIEW_PATH_{camera_id.upper().replace('-', '')}"
    slug = camera_id.lower().replace("-", "")
    return _os.environ.get(env_key, f"/dev/shm/v11_ui_preview_{slug}_v1.bin")


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

    # Build the supplied widget normally first, but do not let its single-path
    # legacy live-preview hook create the reader. Multi-camera preview readers
    # are attached below using one SHM path per runtime camera.
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
    self.live_object_count = 0
    self.camera.online = False
    self.camera.fps = 0.0
    self.camera.last_error = "Ulanish kutilmoqda"
    self.live_timer = QTimer(self)
    self.live_timer.setInterval(33)
    self.live_timer.timeout.connect(self._poll_live_preview)
    self.live_timer.start()


# PySide6 QRect.size() returns QSize directly. Normalize both QSize/QSizeF safely
# so a paint exception cannot leave QPainter save()/restore() unbalanced.
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
CameraView._draw_live_frame = _camera_view_draw_live_frame_qsize_safe

del _source, _parts_dir, _Path, _os
