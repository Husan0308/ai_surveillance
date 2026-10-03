from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TextIO

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import QDialog, QFrame, QGridLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameReader


_TIMING_HANDLES: dict[Path, TextIO] = {}


def _write_timing_row(output: Path, row: dict) -> None:
    """Append diagnostic telemetry without opening the file on every paint."""
    try:
        handle = _TIMING_HANDLES.get(output)
        if handle is None or handle.closed:
            output.parent.mkdir(parents=True, exist_ok=True)
            handle = output.open("a", encoding="utf-8", buffering=1)
            _TIMING_HANDLES[output] = handle
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    except OSError:
        # Timing telemetry is diagnostic-only and must never disrupt UI.
        pass


VERIFIED_LIVE_SOURCE_DIMENSIONS = {
    "CAM01": (2560, 1440),
    "CAM04": (3200, 1800),
}


def _preview_path(camera_id: str) -> str:
    key = f"V11_UI_PREVIEW_PATH_{camera_id.upper().replace('-', '')}"
    slug = camera_id.lower().replace('-', '')
    return os.getenv(key, f"/dev/shm/v11_ui_preview_{slug}_v1.bin")


def overlay_source_dimensions(camera_id: str, source_mode: str = "live") -> tuple[int, int]:
    """Return the pixel coordinate space used by MV3DT boxes for this source.

    Offline files use the calibrated 1920x1080 canvas by default. Live cameras
    may negotiate different native dimensions, so deployments can override
    each camera independently (for example CAM01=2560x1440, CAM04=3200x1800).
    """
    key = camera_id.upper().replace("-", "")
    default_width = os.getenv("MV3DT_OVERLAY_WIDTH", "1920")
    default_height = os.getenv("MV3DT_OVERLAY_HEIGHT", "1080")
    if source_mode != "live":
        return max(1, int(default_width)), max(1, int(default_height))
    verified_width, verified_height = VERIFIED_LIVE_SOURCE_DIMENSIONS.get(key, (int(default_width), int(default_height)))
    width = os.getenv(f"MV3DT_OVERLAY_WIDTH_{key}", verified_width)
    height = os.getenv(f"MV3DT_OVERLAY_HEIGHT_{key}", verified_height)
    return max(1, int(width)), max(1, int(height))


def scale_overlay_bbox(
    bbox: list[float] | tuple[float, ...],
    image_width: int,
    image_height: int,
    source_width: int,
    source_height: int,
) -> tuple[int, int, int, int]:
    """Scale a source-space bbox into the displayed frame's pixel space."""
    left, top, right, bottom = (float(value) for value in bbox[:4])
    sx = image_width / max(1, source_width)
    sy = image_height / max(1, source_height)
    return int(left * sx), int(top * sy), int(right * sx), int(bottom * sy)


class CameraTile(QFrame):
    acceptance_bbox_selected = Signal(str, object, object)

    def __init__(self, camera_id: str, ml_video_base_url: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.camera_id = camera_id
        # Shared memory is the sole video contract. The old MJPEG endpoint is
        # intentionally unsupported; never open another network video reader.
        self.use_shared_memory = True
        self.preview_reader = PreviewFrameReader(_preview_path(camera_id))
        self.reader = None
        self.last_version = 0
        self.last_frame_mono = 0.0
        self.fps_sample_version = 0
        self.fps_sample_mono = 0.0
        self.measured_fps = 0.0
        self.identity_rows: list[dict] = []
        self.acceptance_candidates: list[dict] = []
        self.acceptance_labels: dict[tuple[str, int], str] = {}
        self.acceptance_enabled = False
        self.acceptance_source_dimensions: tuple[int, int] | None = None
        self.current_image: QImage | None = None
        self.current_payload: bytes | None = None
        self.current_frame_pixmap: QPixmap | None = None
        self.preview_diagnostic_log = Path(os.environ["MV3DT_PREVIEW_LATENCY_LOG"]) if os.getenv("MV3DT_PREVIEW_LATENCY_LOG") else None
        self.startup_capture_dir = Path(os.environ["MV3DT_UI_STARTUP_CAPTURE_DIR"]) if os.getenv("MV3DT_UI_STARTUP_CAPTURE_DIR") else None
        self.startup_capture_stages: set[str] = set()
        self.session_start_ns = time.monotonic_ns()
        self.same_sequence_tile_renders = 0
        self.same_sequence_paint_events = 0
        self._last_tile_render_sequence = 0
        self.displayed_pixmap_size = None
        self.fullscreen_dialog: QDialog | None = None
        self.fullscreen_video: QLabel | None = None
        self.fullscreen_shortcut: QShortcut | None = None
        self.show_native = os.getenv("FRONTEND_SHOW_NATIVE_IDS", "0") == "1"
        self.source_mode = "live"
        self.dev_room_canvas_width, self.dev_room_canvas_height = overlay_source_dimensions(camera_id)

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.title = QLabel(camera_id)
        self.status = QLabel("CONNECTING")
        self.video = _ClickableVideoLabel("Connecting...")
        timing_log = os.getenv("MV3DT_PREVIEW_LATENCY_LOG")
        self.video.preview_timing_log = Path(timing_log) if timing_log else None
        self.video.preview_camera_id = self.camera_id
        self.video.preview_timing_context = None
        self.video.preview_last_logged_sequence = 0
        self.video.clicked.connect(self._on_video_clicked)
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(320, 180)
        self.video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video.setStyleSheet("background: black; color: white;")
        self.fullscreen_button = QPushButton("Fullscreen")
        self.fullscreen_button.setToolTip("Open the latest full-resolution shared preview")
        self.fullscreen_button.clicked.connect(self._open_fullscreen)

        header = QGridLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.title, 0, 0)
        header.addWidget(self.status, 0, 1, alignment=Qt.AlignmentFlag.AlignRight)
        header.addWidget(self.fullscreen_button, 0, 2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addLayout(header)
        layout.addWidget(self.video, 1)

        if self.reader is not None:
            self.reader.start()

    def _draw_frame(self, image: QImage) -> None:
        self.current_image = image
        timing = self.video.preview_timing_context
        if timing is not None:
            timing["t7c_qimage_created_monotonic_ns"] = time.monotonic_ns()
            timing["qimage_width"] = int(image.width())
            timing["qimage_height"] = int(image.height())
            timing["qimage_bytes_per_line"] = int(image.bytesPerLine())
            if "qimage" not in self.startup_capture_stages:
                self._capture_startup_stage("qimage", image.bits().tobytes(), timing, image.bytesPerLine(), "RGB32")
        pixmap_start = time.monotonic_ns()
        self.current_frame_pixmap = QPixmap.fromImage(image)
        if timing is not None and "qpixmap" not in self.startup_capture_stages:
            pixmap_image = self.current_frame_pixmap.toImage()
            timing["t7d_qpixmap_created_monotonic_ns"] = time.monotonic_ns()
            timing["qpixmap_create_duration_ns"] = timing["t7d_qpixmap_created_monotonic_ns"] - pixmap_start
            self._capture_startup_stage("qpixmap", pixmap_image.bits().tobytes(), timing,
                                        pixmap_image.bytesPerLine(), "RGB32")
        elif timing is not None:
            timing["t7d_qpixmap_created_monotonic_ns"] = time.monotonic_ns()
            timing["qpixmap_create_duration_ns"] = timing["t7d_qpixmap_created_monotonic_ns"] - pixmap_start
        self._update_tile_pixmap()
        self._update_fullscreen_pixmap()

    def _capture_startup_stage(
        self, stage: str, payload: bytes, timing: dict, stride: int, pixel_format: str,
        width: int | None = None, height: int | None = None,
    ) -> None:
        """Capture exactly the first consumer-side frame at selected boundaries."""
        if self.startup_capture_dir is None or stage in self.startup_capture_stages:
            return
        try:
            self.startup_capture_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{self.camera_id.lower()}-{stage}"
            (self.startup_capture_dir / f"{stem}.raw").write_bytes(payload)
            meta = {k: timing.get(k) for k in ("camera_id", "sequence", "source_frame_num", "pts_ns",
                "t6_ipc_publish_monotonic_ns", "preview_width", "preview_height")}
            meta.update({"stage": stage,
                "width": int(width if width is not None else timing.get("preview_width", 0)),
                "height": int(height if height is not None else timing.get("preview_height", 0)), "stride": int(stride),
                "format": pixel_format, "payload_bytes": len(payload), "ready": True,
                "session_start_ns": self.session_start_ns, "capture_monotonic_ns": time.monotonic_ns()})
            (self.startup_capture_dir / f"{stem}.json").write_text(json.dumps(meta, separators=(",", ":")) + "\n")
            self.startup_capture_stages.add(stage)
        except OSError:
            # Diagnostic capture must never disrupt the live preview path.
            return

    def _draw_overlays(self, pixmap: QPixmap) -> None:
        painter = QPainter(pixmap)
        for row in self.identity_rows:
            bbox = row.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            if self.camera_id in {"CAM-01", "CAM-04"}:
                coordinate_width, coordinate_height = self.acceptance_source_dimensions or overlay_source_dimensions(
                    self.camera_id, self.source_mode
                )
            else:
                coordinate_width = self.current_image.width() if self.current_image is not None else pixmap.width()
                coordinate_height = self.current_image.height() if self.current_image is not None else pixmap.height()
            left, top, right, bottom = scale_overlay_bbox(
                bbox,
                pixmap.width(),
                pixmap.height(),
                coordinate_width,
                coordinate_height,
            )
            color = QColor(80, 220, 80) if str(row.get("application_id")) == "Person_01" else QColor(40, 165, 255)
            painter.setPen(QPen(color, 3))
            painter.drawRect(left, top, right - left, bottom - top)
            label = f"{row.get('application_id', 'Unknown')} | {self.camera_id}"
            painter.drawText(left, max(20, top - 8), label)
            if self.show_native:
                native = row.get("debug", {}).get("native_mv3dt_id", "?")
                painter.drawText(left, min(pixmap.height() - 6, bottom + 18), f"native:{native}")
        if self.acceptance_enabled:
            for row in self.acceptance_candidates:
                bbox = row.get("bbox")
                native = row.get("native_track_id")
                if not isinstance(bbox, (list, tuple)) or len(bbox) < 4 or native is None:
                    continue
                if row.get("camera_id") != self.camera_id:
                    continue
                coordinate_width, coordinate_height = self.acceptance_source_dimensions or overlay_source_dimensions(self.camera_id, self.source_mode)
                left, top, right, bottom = scale_overlay_bbox(
                    bbox, pixmap.width(), pixmap.height(), coordinate_width, coordinate_height
                )
                tester = self.acceptance_labels.get((self.camera_id, int(native)))
                if tester:
                    painter.setPen(QPen(QColor(255, 70, 220), 4))
                    painter.drawRect(left, top, right - left, bottom - top)
                    painter.drawText(left, max(20, top - 10), f"{tester} · ACCEPTANCE ONLY")
        painter.end()

    def _update_tile_pixmap(self) -> None:
        if self.current_frame_pixmap is None:
            return
        target = self.video.size()
        pixmap = self.current_frame_pixmap
        scale_start = time.monotonic_ns()
        if target.width() > 0 and target.height() > 0:
            pixmap = pixmap.scaled(
                target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        else:
            pixmap = pixmap.copy()
        timing = self.video.preview_timing_context
        if timing is not None and timing.get("sequence") == self.last_version:
            timing["t7e_scale_completed_monotonic_ns"] = time.monotonic_ns()
            timing["scale_duration_ns"] = timing["t7e_scale_completed_monotonic_ns"] - scale_start
            timing["tile_source_pixmap_width"] = int(self.current_frame_pixmap.width())
            timing["tile_source_pixmap_height"] = int(self.current_frame_pixmap.height())
            timing["tile_target_width"] = int(target.width())
            timing["tile_target_height"] = int(target.height())
            timing["tile_scaled_width"] = int(pixmap.width())
            timing["tile_scaled_height"] = int(pixmap.height())
            timing["tile_transformation"] = "SmoothTransformation"
        elif self.last_version and self._last_tile_render_sequence == self.last_version:
            self.same_sequence_tile_renders += 1
        overlay_start = time.monotonic_ns()
        self._draw_overlays(pixmap)
        if timing is not None and timing.get("sequence") == self.last_version:
            timing["t7f_overlay_completed_monotonic_ns"] = time.monotonic_ns()
            timing["overlay_duration_ns"] = timing["t7f_overlay_completed_monotonic_ns"] - overlay_start
            if "displayed" not in self.startup_capture_stages:
                display_image = pixmap.toImage()
                self._capture_startup_stage("displayed", display_image.bits().tobytes(), timing,
                                            display_image.bytesPerLine(), "RGB32",
                                            display_image.width(), display_image.height())
        self.displayed_pixmap_size = pixmap.size()
        setpixmap_start = time.monotonic_ns()
        self.video.setPixmap(pixmap)
        if timing is not None and timing.get("sequence") == self.last_version:
            timing["t8_setpixmap_return_monotonic_ns"] = time.monotonic_ns()
            timing["setpixmap_duration_ns"] = timing["t8_setpixmap_return_monotonic_ns"] - setpixmap_start
            timing["t8_label_dimensions"] = [int(self.video.width()), int(self.video.height())]
        # QLabel normally defers this paint until the GUI event loop returns.
        # Under the six-camera refresh timer that added a measured 10–19 ms
        # after setPixmap. Paint the newly received latest frame now, while
        # retaining the same pixmap, overlays, and latest-only semantics.
        if self.video.isVisible():
            self.video.repaint()
        self._last_tile_render_sequence = self.last_version

    def _update_fullscreen_pixmap(self) -> None:
        if self.fullscreen_video is None or self.current_frame_pixmap is None:
            return
        target = self.fullscreen_video.size()
        pixmap = self.current_frame_pixmap
        if target.width() > 0 and target.height() > 0:
            pixmap = pixmap.scaled(
                target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        else:
            pixmap = pixmap.copy()
        self._draw_overlays(pixmap)
        self.fullscreen_video.setPixmap(pixmap)

    def _open_fullscreen(self) -> None:
        if self.current_frame_pixmap is None:
            return
        if self.fullscreen_dialog is None:
            dialog = QDialog(self.window(), Qt.WindowType.Window)
            dialog.setWindowTitle(f"{self.camera_id} — live preview")
            dialog.setStyleSheet("background: black;")
            label = QLabel(dialog)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("background: black;")
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(label)
            self.fullscreen_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), dialog)
            self.fullscreen_shortcut.activated.connect(dialog.close)
            self.fullscreen_dialog = dialog
            self.fullscreen_video = label
        self._update_fullscreen_pixmap()
        self.fullscreen_dialog.showFullScreen()
        self._update_fullscreen_pixmap()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_tile_pixmap()
        self._update_fullscreen_pixmap()

    def refresh(self) -> None:
        if self.preview_reader is not None:
            refresh_start = time.monotonic_ns()
            frame = self.preview_reader.read_latest(max_age_sec=0.25, after_sequence=self.last_version)
            if frame is not None and frame.sequence != self.last_version:
                reader_return_ns = time.monotonic_ns()
                self.current_payload = frame.payload
                qimage_start = time.monotonic_ns()
                image = QImage(self.current_payload, frame.width, frame.height, frame.stride, QImage.Format.Format_RGB32)
                if not image.isNull():
                    skipped_frames = max(0, frame.sequence - self.last_version - 1) if self.last_version else 0
                    now = time.monotonic()
                    if frame.sequence < self.fps_sample_version:
                        self.fps_sample_version = frame.sequence
                        self.fps_sample_mono = now
                        self.measured_fps = 0.0
                        # The publisher restarts its per-file sequence at 1.
                        # Reset diagnostic deduplication too, or paint samples
                        # are suppressed until the new run passes the old ID.
                        self.video.preview_last_logged_sequence = 0
                    elif self.fps_sample_mono <= 0.0:
                        self.fps_sample_version = frame.sequence
                        self.fps_sample_mono = now
                    elif now - self.fps_sample_mono >= 0.5:
                        elapsed = now - self.fps_sample_mono
                        sampled = max(0.0, (frame.sequence - self.fps_sample_version) / elapsed)
                        self.measured_fps = sampled if self.measured_fps <= 0.0 else (0.65 * self.measured_fps + 0.35 * sampled)
                        self.fps_sample_version = frame.sequence
                        self.fps_sample_mono = now
                    self.last_version = frame.sequence
                    self.last_frame_mono = now
                    self.video.preview_timing_context = {
                        "camera_id": self.camera_id,
                        "sequence": frame.sequence,
                        "t6_ipc_publish_monotonic_ns": frame.timestamp_ns,
                        "t0_decoder_reference_monotonic_ns": frame.decoder_reference_ns,
                        "t1_decoder_out_monotonic_ns": frame.decoder_out_ns,
                        "pts_ns": frame.pts_ns,
                        "source_frame_num": frame.source_frame_num,
                        "t7_ui_receive_monotonic_ns": time.monotonic_ns(),
                        "t7_ui_receive_wall_ns": time.time_ns(),
                        "t7a_reader_poll_begin_monotonic_ns": refresh_start,
                        "t7b_reader_return_monotonic_ns": reader_return_ns,
                        "t7c_qimage_create_begin_monotonic_ns": qimage_start,
                        "ui_skipped_preview_frames": skipped_frames,
                        "preview_width": frame.width,
                        "preview_height": frame.height,
                        "preview_stride": frame.stride,
                        "preview_format": "RGB32",
                        "preview_payload_bytes": len(frame.payload),
                        "preview_reader_poll_duration_ns": reader_return_ns - refresh_start,
                        "session_start_ns": self.session_start_ns,
                    }
                    if "shared-memory" not in self.startup_capture_stages:
                        self._capture_startup_stage("shared-memory", frame.payload, self.video.preview_timing_context,
                                                    frame.stride, "BGRA/RGB32 shared payload")
                    self._draw_frame(image)
                    self.video.preview_timing_context["camera_refresh_duration_ns"] = time.monotonic_ns() - refresh_start
                    state = "REPLAY" if self.source_mode == "replay" and self.camera_id in {"CAM-01", "CAM-04"} else "LIVE"
                    displayed_fps = self.measured_fps if self.measured_fps > 0.0 else frame.fps
                    self.status.setText(f"{state} · {displayed_fps:.1f} FPS")
                    return
            if self.last_version == 0:
                self.status.setText("CONNECTING")
            elif time.monotonic() - self.last_frame_mono > 0.25:
                self.status.setText("STALE · 0.0 FPS")
                self.current_image = None
                self.current_frame_pixmap = None
                self.displayed_pixmap_size = None
                self.video.clear()
            return

    def is_connected(self) -> bool:
        if self.preview_reader is not None:
            return self.last_version > 0 and time.monotonic() - self.last_frame_mono <= 0.25
        return False

    def close_reader(self) -> None:
        if self.fullscreen_dialog is not None:
            self.fullscreen_dialog.close()
        if self.reader is not None:
            self.reader.stop()
            self.reader.join()
        if self.preview_reader is not None:
            self.preview_reader.close()

    def set_identity_rows(self, rows: list[dict]) -> None:
        self.identity_rows = list(rows)

    def set_source_mode(self, mode: str) -> None:
        self.source_mode = mode if mode in {"replay", "live"} else "live"
        replay = self.source_mode == "replay" and self.camera_id in {"CAM-01", "CAM-04"}
        self.title.setText(f"{self.camera_id} | REPLAY" if replay else self.camera_id)

    def set_acceptance_mode(self, enabled: bool) -> None:
        self.acceptance_enabled = bool(enabled)
        if not self.acceptance_enabled:
            self.acceptance_candidates = []
            self.acceptance_labels = {}
            self.acceptance_source_dimensions = None
        if self.current_image is not None:
            self._draw_frame(self.current_image)
        else:
            self.update()

    def set_acceptance_candidates(self, candidates: list[dict]) -> None:
        self.acceptance_candidates = [
            row for row in candidates if row.get("camera_id") == self.camera_id
        ] if self.acceptance_enabled else []
        if self.acceptance_enabled:
            dimension_row = next((row for row in self.acceptance_candidates
                                  if int(row.get("source_frame_width") or 0) > 0
                                  and int(row.get("source_frame_height") or 0) > 0), None)
            if dimension_row is not None:
                self.acceptance_source_dimensions = (
                    int(dimension_row["source_frame_width"]),
                    int(dimension_row["source_frame_height"]),
                )
        if self.current_image is not None:
            self._draw_frame(self.current_image)

    def set_acceptance_labels(self, labels: dict[tuple[str, int], str]) -> None:
        self.acceptance_labels = dict(labels) if self.acceptance_enabled else {}
        if self.current_image is not None:
            self._draw_frame(self.current_image)

    def _on_video_clicked(self, x: int, y: int) -> None:
        if not self.acceptance_enabled or self.current_image is None or self.displayed_pixmap_size is None:
            return
        pixmap_width = max(1, int(self.displayed_pixmap_size.width()))
        pixmap_height = max(1, int(self.displayed_pixmap_size.height()))
        offset_x = (self.video.width() - pixmap_width) / 2.0
        offset_y = (self.video.height() - pixmap_height) / 2.0
        px = (float(x) - offset_x) * self.current_image.width() / pixmap_width
        py = (float(y) - offset_y) * self.current_image.height() / pixmap_height
        if not (0.0 <= px < self.current_image.width() and 0.0 <= py < self.current_image.height()):
            return
        coordinate_width, coordinate_height = self.acceptance_source_dimensions or overlay_source_dimensions(self.camera_id, self.source_mode)
        source_x = px * coordinate_width / self.current_image.width()
        source_y = py * coordinate_height / self.current_image.height()
        matches = []
        for row in self.acceptance_candidates:
            bbox = row.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            left, top, right, bottom = (float(v) for v in bbox[:4])
            if left <= source_x <= right and top <= source_y <= bottom:
                matches.append((max(0.0, right - left) * max(0.0, bottom - top), row))
        if not matches:
            return
        row = min(matches, key=lambda item: item[0])[1]
        bbox = row["bbox"]
        left = max(0, int(float(bbox[0]) * self.current_image.width() / coordinate_width))
        top = max(0, int(float(bbox[1]) * self.current_image.height() / coordinate_height))
        right = min(self.current_image.width(), int(float(bbox[2]) * self.current_image.width() / coordinate_width + 0.999))
        bottom = min(self.current_image.height(), int(float(bbox[3]) * self.current_image.height() / coordinate_height + 0.999))
        if right <= left or bottom <= top:
            return
        crop = self.current_image.copy(left, top, right - left, bottom - top)
        self.acceptance_bbox_selected.emit(self.camera_id, row, crop)


class _ClickableVideoLabel(QLabel):
    clicked = Signal(int, int)

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        timing = self.preview_timing_context
        output = self.preview_timing_log
        if not timing or output is None:
            return
        sequence = int(timing.get("sequence", 0))
        if sequence <= self.preview_last_logged_sequence:
            tile = self.parentWidget()
            if isinstance(tile, CameraTile):
                tile.same_sequence_paint_events += 1
            return
        self.preview_last_logged_sequence = sequence
        row = {
            **timing,
            "t8_ui_paint_monotonic_ns": time.monotonic_ns(),
            "t8_ui_paint_wall_ns": time.time_ns(),
            "widget_width": int(self.width()),
            "widget_height": int(self.height()),
        }
        _write_timing_row(output, row)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(int(event.position().x()), int(event.position().y()))
        super().mousePressEvent(event)


class CameraWall(QWidget):
    acceptance_bbox_selected = Signal(str, object, object)

    def __init__(self, ml_video_base_url: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ml_video_base_url = ml_video_base_url
        self.source_mode = "live"
        self.tiles: dict[str, CameraTile] = {}
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(6)
        for row in range(2):
            self.grid.setRowStretch(row, 1)
        for column in range(3):
            self.grid.setColumnStretch(column, 1)

    def set_cameras(self, camera_ids: list[str]) -> None:
        if list(self.tiles) == camera_ids:
            return
        for tile in self.tiles.values():
            tile.close_reader()
            self.grid.removeWidget(tile)
            tile.deleteLater()
        self.tiles.clear()
        for index, camera_id in enumerate(camera_ids):
            tile = CameraTile(camera_id, self.ml_video_base_url, self)
            tile.set_source_mode(self.source_mode)
            tile.acceptance_bbox_selected.connect(self.acceptance_bbox_selected.emit)
            self.tiles[camera_id] = tile
            row, column = divmod(index, 3)
            self.grid.addWidget(tile, row, column)

    def refresh_frames(self) -> dict:
        tick_start = time.monotonic_ns()
        per_camera = {}
        changed = 0
        same_sequence_renders_before = 0
        same_sequence_paints_before = 0
        for tile in self.tiles.values():
            previous = tile.last_version
            render_count = tile.same_sequence_tile_renders
            paint_count = tile.same_sequence_paint_events
            camera_start = time.monotonic_ns()
            tile.refresh()
            per_camera[tile.camera_id] = {
                "duration_ns": time.monotonic_ns() - camera_start,
                "sequence_before": previous,
                "sequence_after": tile.last_version,
                "new_sequence": tile.last_version != previous,
                "same_sequence_tile_renders": tile.same_sequence_tile_renders - render_count,
                "same_sequence_paint_events": tile.same_sequence_paint_events - paint_count,
                "image_dimensions": [tile.current_image.width(), tile.current_image.height()] if tile.current_image else None,
                "target_dimensions": [tile.video.width(), tile.video.height()],
            }
            changed += int(tile.last_version != previous)
            same_sequence_renders_before += tile.same_sequence_tile_renders - render_count
            same_sequence_paints_before += tile.same_sequence_paint_events - paint_count
        return {
            "duration_ns": time.monotonic_ns() - tick_start,
            "tiles_with_new_sequence": changed,
            "tiles_unchanged": len(self.tiles) - changed,
            "same_sequence_tile_renders": same_sequence_renders_before,
            "same_sequence_paint_events": same_sequence_paints_before,
            "cameras": per_camera,
        }

    def connection_counts(self) -> tuple[int, int]:
        return sum(tile.is_connected() for tile in self.tiles.values()), len(self.tiles)

    def set_identity_snapshot(self, people: list[dict]) -> None:
        by_camera = {}
        for person in people:
            by_camera.setdefault(person.get("camera_id"), []).append(person)
        for camera_id, tile in self.tiles.items():
            tile.set_identity_rows(by_camera.get(camera_id, []))

    def set_source_mode(self, mode: str) -> None:
        self.source_mode = mode if mode in {"replay", "live"} else "live"
        for tile in self.tiles.values():
            tile.set_source_mode(self.source_mode)

    def set_acceptance_mode(self, enabled: bool) -> None:
        for tile in self.tiles.values():
            tile.set_acceptance_mode(enabled)

    def set_acceptance_candidates(self, candidates: list[dict]) -> None:
        for tile in self.tiles.values():
            tile.set_acceptance_candidates(candidates)

    def set_acceptance_labels(self, labels: dict[tuple[str, int], str]) -> None:
        for tile in self.tiles.values():
            tile.set_acceptance_labels(labels)

    def close_readers(self) -> None:
        for tile in self.tiles.values():
            tile.close_reader()
