from __future__ import annotations

import atexit
import json
import os
import sys
import time
from pathlib import Path
from typing import TextIO

from PySide6.QtCore import QSize, Qt, Signal, QSocketNotifier, QTimer, QRectF, Slot
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import QDialog, QFrame, QGridLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameNotifyReceiver, PreviewFrameReader


_TIMING_HANDLES: dict[Path, TextIO] = {}


def _flush_timing_handles() -> None:
    for handle in tuple(_TIMING_HANDLES.values()):
        try:
            handle.flush()
            handle.close()
        except OSError:
            pass
    _TIMING_HANDLES.clear()


atexit.register(_flush_timing_handles)


def _write_timing_row(output: Path, row: dict) -> None:
    """Append diagnostic telemetry without opening the file on every paint."""
    try:
        handle = _TIMING_HANDLES.get(output)
        if handle is None or handle.closed:
            output.parent.mkdir(parents=True, exist_ok=True)
            # Buffered diagnostics keep per-frame file flushes off the Qt paint
            # critical path. Normal process shutdown flushes all accepted rows.
            handle = output.open("a", encoding="utf-8", buffering=1024 * 1024)
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


FULLSCREEN_PREVIEW_PATH = "/dev/shm/v11_ui_preview_fullscreen_v1.bin"
FULLSCREEN_REQUEST_PATH = Path("/dev/shm/v11_ui_fullscreen_request_v1")


def _set_fullscreen_preview_request(camera_id: str | None) -> None:
    """Select at most one native high-resolution preview branch."""
    value = "" if camera_id is None else camera_id
    temporary = FULLSCREEN_REQUEST_PATH.with_name(FULLSCREEN_REQUEST_PATH.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(value + "\n", encoding="ascii")
        os.replace(temporary, FULLSCREEN_REQUEST_PATH)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass


def overlay_source_dimensions(camera_id: str, source_mode: str = "live",
                             frame_snapshot: dict | None = None) -> tuple[int, int]:
    """Return the pixel coordinate space used by MV3DT boxes for this source.

    Offline files use the calibrated 1920x1080 canvas by default. Live cameras
    may negotiate different native dimensions. Prefer dimensions carried with
    the exact image/tracker snapshot; the static per-camera map is only a
    fallback for older feeds that lack source geometry.
    """
    if source_mode == "live" and isinstance(frame_snapshot, dict):
        try:
            if frame_snapshot.get("camera_id") == camera_id:
                width = int(frame_snapshot.get("source_frame_width", 0))
                height = int(frame_snapshot.get("source_frame_height", 0))
                if width > 0 and height > 0:
                    return width, height
        except (TypeError, ValueError):
            pass
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


def bbox_coordinate_dimensions(
    row: dict, fallback: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """Return the bbox's declared pixel space, never guessing pipeline space.

    A DeepStream pipeline-space rectangle is measured on the nvstreammux
    canvas. Its decoded source dimensions are descriptive metadata only and
    must never be used to scale that rectangle. If a pipeline-space record is
    incomplete or internally inconsistent, return None so the caller can
    fail closed instead of painting a displaced box.
    """
    try:
        coordinate_space = row.get("bbox_coordinate_space")
        bbox_width = int(row.get("bbox_coordinate_width") or 0)
        bbox_height = int(row.get("bbox_coordinate_height") or 0)
        pipeline_width = int(row.get("pipeline_width") or 0)
        pipeline_height = int(row.get("pipeline_height") or 0)
        if coordinate_space == "pipeline":
            width = bbox_width or pipeline_width
            height = bbox_height or pipeline_height
            if width <= 0 or height <= 0:
                return None
            if ((pipeline_width > 0 and width != pipeline_width)
                    or (pipeline_height > 0 and height != pipeline_height)):
                return None
            return width, height

        if coordinate_space == "source":
            width = bbox_width or int(row.get("source_frame_width") or 0)
            height = bbox_height or int(row.get("source_frame_height") or 0)
        else:
            width = bbox_width or pipeline_width or int(row.get("source_frame_width") or 0)
            height = bbox_height or pipeline_height or int(row.get("source_frame_height") or 0)
    except (TypeError, ValueError):
        return None
    if width > 0 and height > 0:
        return width, height
    if fallback is not None and fallback[0] > 0 and fallback[1] > 0:
        return fallback
    return None


def snapshot_bbox_coordinate_dimensions(
    snapshot: dict, fallback: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """Validate the frame-level bbox contract before joining UI objects."""
    dimensions = bbox_coordinate_dimensions(snapshot, fallback)
    if snapshot.get("bbox_coordinate_space") == "pipeline":
        try:
            pipeline_dimensions = (int(snapshot["pipeline_width"]),
                                  int(snapshot["pipeline_height"]))
        except (KeyError, TypeError, ValueError):
            return None
        if dimensions is None or dimensions != pipeline_dimensions:
            return None
    return dimensions


def snapshot_bbox_rows(
    snapshot: dict, key: str, dimensions: tuple[int, int] | None,
) -> list[dict]:
    """Return only rows whose bbox space agrees with the frame contract."""
    if dimensions is None:
        return []
    rows = snapshot.get(key, [])
    if not isinstance(rows, list):
        return []
    clean: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_dimensions = bbox_coordinate_dimensions(row, dimensions)
        if row_dimensions != dimensions:
            continue
        clean.append(row | {"bbox_coordinate_width": dimensions[0],
                            "bbox_coordinate_height": dimensions[1]})
    return clean


class CameraTile(QFrame):
    acceptance_bbox_selected = Signal(str, object, object)

    def __init__(self, camera_id: str, ml_video_base_url: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.camera_id = camera_id
        # Shared memory is the sole video contract. The old MJPEG endpoint is
        # intentionally unsupported; never open another network video reader.
        self.use_shared_memory = True
        # Read on the existing Qt socket-notification event. The mmap reader is
        # latest-only and copies pixels only for a new committed sequence; a
        # second worker/signal handoff previously added up to ~22 ms at P95.
        self.preview_reader = PreviewFrameReader(_preview_path(camera_id))
        self.fullscreen_reader: PreviewFrameReader | None = None
        self.fullscreen_last_sequence = 0
        self.fullscreen_current_image: QImage | None = None
        self.fullscreen_current_payload: bytes | None = None
        self.fullscreen_last_frame_mono = 0.0
        self.fullscreen_tracked_objects: list[dict] = []
        self.fullscreen_current_frame: dict | None = None
        self.fullscreen_source_dimensions: tuple[int, int] | None = None
        self.fullscreen_sync_misses = 0
        self.preview_sync_misses = 0
        self.reader = None
        self.last_version = 0
        self.last_frame_mono = 0.0
        self.fps_sample_version = 0
        self.fps_sample_mono = 0.0
        self.measured_fps = 0.0
        self.identity_rows: list[dict] = []
        self.tracked_objects: list[dict] = []
        self.detector_proposals: list[dict] = []
        self.box_diagnostics = os.getenv("FRONTEND_BOX_DIAGNOSTICS", "0") == "1"
        # The canonical local tracker IPC is the sole box source. An empty or
        # stale feed means no box; identity rows never become geometry.
        self.tracked_object_feed_supported = True
        self.current_source_frame_num: int | None = None
        self.current_pts_ns: int | None = None
        self.current_source_dimensions: tuple[int, int] | None = None
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
        capture_frame = os.getenv("MV3DT_UI_CAPTURE_FRAME_NUM")
        self.startup_capture_frame_num = int(capture_frame) if capture_frame and capture_frame.isdigit() else None
        self.session_start_ns = time.monotonic_ns()
        self.preview_notify_received_ns = 0
        self.same_sequence_tile_renders = 0
        self.same_sequence_paint_events = 0
        self._last_tile_render_sequence = 0
        self.displayed_pixmap_size = None
        self.fullscreen_dialog: QDialog | None = None
        self.fullscreen_video: QLabel | None = None
        self.fullscreen_shortcut: QShortcut | None = None
        self.show_native = os.getenv("FRONTEND_SHOW_NATIVE_IDS", "0") == "1"
        self.tracker_osd_baked_into_preview = False
        self.source_mode = "live"
        self.dev_room_canvas_width, self.dev_room_canvas_height = overlay_source_dimensions(camera_id)

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.title = QLabel(camera_id)
        self.status = QLabel("CONNECTING")
        self.video = _ClickableVideoLabel("Connecting...")
        self.video.camera_tile = self
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


    def _draw_frame(self, image: QImage, *, paint: bool = True) -> None:
        self.current_image = image
        # Full-resolution QPixmap copies are only needed while that camera is
        # actually fullscreen. Normal wall tiles draw the shared-memory QImage
        # directly with smooth painter scaling, avoiding an 8 MB QPixmap copy
        # just to shrink it.
        self.current_frame_pixmap = None
        timing = self.video.preview_timing_context
        if timing is not None:
            timing["t7c_qimage_created_monotonic_ns"] = time.monotonic_ns()
            timing["qimage_width"] = int(image.width())
            timing["qimage_height"] = int(image.height())
            timing["qimage_bytes_per_line"] = int(image.bytesPerLine())
            if "qimage" not in self.startup_capture_stages:
                self._capture_startup_stage("qimage", image.bits().tobytes(), timing, image.bytesPerLine(), "RGB32")
        if paint:
            self._update_tile_pixmap()

    @Slot()
    def _consume_latest_preview(self, notify_received_ns: int = 0) -> None:
        """Read and paint one newest committed frame in the notifier event."""
        poll_start_ns = time.monotonic_ns()
        frame = self.preview_reader.read_latest(
            max_age_sec=0.25, after_sequence=self.last_version
        )
        poll_return_ns = time.monotonic_ns()
        if frame is None:
            return
        reader_timing = {
            "t7_notify_received_monotonic_ns": int(notify_received_ns) or poll_start_ns,
            "t7a_reader_poll_begin_monotonic_ns": poll_start_ns,
            "t7b_reader_return_monotonic_ns": poll_return_ns,
            "preview_reader_poll_duration_ns": poll_return_ns - poll_start_ns,
        }
        if frame.sequence == self.last_version:
            return

        receive_ns = time.monotonic_ns()
        snapshot = frame.tracker_snapshot
        if (snapshot is not None and (not isinstance(snapshot, dict)
                or snapshot.get("camera_id") != self.camera_id
                or int(snapshot.get("pts_ns", -1)) != frame.pts_ns
                or int(snapshot.get("source_frame_num", -1)) != frame.source_frame_num)):
            self.preview_sync_misses += 1
            return

        qimage_start = time.monotonic_ns()
        image = QImage(frame.payload, frame.width, frame.height, frame.stride,
                       QImage.Format.Format_RGB32)
        if image.isNull():
            self.preview_sync_misses += 1
            return

        skipped_frames = max(0, frame.sequence - self.last_version - 1) if self.last_version else 0
        now = time.monotonic()
        if frame.sequence < self.fps_sample_version:
            self.fps_sample_version = frame.sequence
            self.fps_sample_mono = now
            self.measured_fps = 0.0
            # The publisher restarts its per-file sequence at 1. Reset
            # diagnostic deduplication too, or paint samples are suppressed
            # until the new run passes the old ID.
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

        self.current_payload = frame.payload
        self.last_version = frame.sequence
        self.current_source_frame_num = frame.source_frame_num
        self.current_pts_ns = frame.pts_ns
        self.current_source_dimensions = (
            overlay_source_dimensions(self.camera_id, self.source_mode, snapshot)
            if isinstance(snapshot, dict)
            else overlay_source_dimensions(self.camera_id, self.source_mode)
        )
        feed = snapshot if isinstance(snapshot, dict) else None
        feed_frame_matches = bool(feed is not None
            and int(feed.get("pts_ns", -1)) == int(frame.pts_ns)
            and int(feed.get("source_frame_num", frame.source_frame_num)) == frame.source_frame_num)
        feed_bbox_dimensions = (snapshot_bbox_coordinate_dimensions(
            feed, self.current_source_dimensions or (frame.width, frame.height))
            if feed_frame_matches else None)
        if feed_frame_matches and feed is not None and feed_bbox_dimensions is not None:
            self.tracked_objects = snapshot_bbox_rows(feed, "objects", feed_bbox_dimensions)
            self.detector_proposals = snapshot_bbox_rows(
                feed, "detector_proposals", feed_bbox_dimensions)
            for row in self.tracked_objects:
                row["source_frame_width"] = int(feed.get("source_frame_width") or frame.width)
                row["source_frame_height"] = int(feed.get("source_frame_height") or frame.height)
        else:
            # Keep the same-frame video, but fail closed on boxes when the
            # embedded canonical snapshot does not declare the actual bbox
            # coordinate canvas. Never infer mux geometry from camera caps.
            self.tracked_objects = []
            self.detector_proposals = []
        self.last_frame_mono = now
        self.video.preview_timing_context = {
            "camera_id": self.camera_id,
            "sequence": frame.sequence,
            "t6_ipc_publish_monotonic_ns": frame.timestamp_ns,
            "t0_decoder_reference_monotonic_ns": frame.decoder_reference_ns,
            "t1_decoder_out_monotonic_ns": frame.decoder_out_ns,
            "t1_preview_prepare_start_monotonic_ns": frame.preview_prepare_start_ns,
            "t2_preview_prepare_complete_monotonic_ns": frame.preview_prepare_complete_ns,
            "t3_tracker_metadata_ready_monotonic_ns": frame.tracker_ready_ns,
            "t4_shared_publish_monotonic_ns": frame.timestamp_ns,
            "pts_ns": frame.pts_ns,
            "source_frame_num": frame.source_frame_num,
            "bbox_metadata_pts_ns": int(snapshot["pts_ns"]) if isinstance(snapshot, dict) else None,
            "bbox_metadata_source_frame_num": int(snapshot["source_frame_num"]) if isinstance(snapshot, dict) else None,
            "image_bbox_pts_match": isinstance(snapshot, dict) and frame.pts_ns == int(snapshot["pts_ns"]),
            "image_bbox_frame_match": isinstance(snapshot, dict) and frame.source_frame_num == int(snapshot["source_frame_num"]),
            "t7_ui_receive_monotonic_ns": receive_ns,
            "t7_ui_receive_wall_ns": time.time_ns(),
            "t7a_reader_poll_begin_monotonic_ns": reader_timing["t7a_reader_poll_begin_monotonic_ns"],
            "t7b_reader_return_monotonic_ns": reader_timing["t7b_reader_return_monotonic_ns"],
            "t7c_qimage_create_begin_monotonic_ns": qimage_start,
            "ui_skipped_preview_frames": skipped_frames,
            "preview_width": frame.width,
            "preview_height": frame.height,
            "preview_stride": frame.stride,
            "preview_format": "RGB32",
            "preview_payload_bytes": len(frame.payload),
            "preview_reader_poll_duration_ns": reader_timing["preview_reader_poll_duration_ns"],
            "session_start_ns": self.session_start_ns,
            "t7_notify_received_monotonic_ns": reader_timing["t7_notify_received_monotonic_ns"],
        }
        if "shared-memory" not in self.startup_capture_stages:
            self._capture_startup_stage("shared-memory", frame.payload, self.video.preview_timing_context,
                                        frame.stride, "BGRA/RGB32 shared payload")
        self._draw_frame(image)
        self.video.preview_timing_context["camera_refresh_duration_ns"] = time.monotonic_ns() - receive_ns
        state = "REPLAY" if self.source_mode == "replay" and self.camera_id in {"CAM-01", "CAM-04"} else "LIVE"
        displayed_fps = self.measured_fps if self.measured_fps > 0.0 else frame.fps
        self.status.setText(f"{state} · {displayed_fps:.1f} FPS")

    def _capture_startup_stage(
        self, stage: str, payload: bytes, timing: dict, stride: int, pixel_format: str,
        width: int | None = None, height: int | None = None,
    ) -> None:
        """Capture exactly the first consumer-side frame at selected boundaries."""
        if self.startup_capture_dir is None or stage in self.startup_capture_stages:
            return
        if (self.startup_capture_frame_num is not None
                and int(timing.get("source_frame_num", -1)) != self.startup_capture_frame_num):
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

    def _paint_overlays(self, painter: QPainter, canvas_width: int, canvas_height: int) -> None:
        if self.tracked_object_feed_supported:
            if self.box_diagnostics:
                for proposal in self.detector_proposals:
                    if proposal.get("camera_id") != self.camera_id:
                        continue
                    try:
                        if (int(proposal.get("frame_num", -1)) != int(self.current_source_frame_num)
                                or int(proposal.get("pts_ns", -1)) != int(self.current_pts_ns)):
                            continue
                    except (TypeError, ValueError):
                        continue
                    bbox = proposal.get("bbox")
                    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                        continue
                    # The approved fallback rectangle is this exact current PGIE
                    # proposal. Label it once as FALLBACK instead of painting a
                    # visually duplicate DET rectangle over the same pixels.
                    fallback_duplicate = any(
                        row.get("bbox_source") == "PGIE_SHADOW_FALLBACK"
                        and row.get("camera_id") == self.camera_id
                        and all(abs(float(row["bbox"][i]) - float(bbox[i])) <= 1.0 for i in range(4))
                        for row in self.tracked_objects
                        if isinstance(row.get("bbox"), (list, tuple)) and len(row["bbox"]) >= 4
                    )
                    if fallback_duplicate:
                        continue
                    dimensions = bbox_coordinate_dimensions(
                        proposal, self.current_source_dimensions or
                        overlay_source_dimensions(self.camera_id, self.source_mode))
                    if dimensions is None:
                        continue
                    coordinate_width, coordinate_height = dimensions
                    left, top, right, bottom = scale_overlay_bbox(
                        bbox, canvas_width, canvas_height, coordinate_width, coordinate_height
                    )
                    painter.setPen(QPen(QColor(255, 170, 35), 2, Qt.PenStyle.DashLine))
                    painter.drawRect(left, top, right - left, bottom - top)
                    painter.drawText(left, max(20, top - 8), f"DET {float(proposal.get('confidence', 0.0)):.2f}")
            for row in self.tracked_objects:
                if row.get("camera_id") != self.camera_id:
                    continue
                native = row.get("native_track_id")
                pts = row.get("pts_ns")
                if native is None or pts is None or self.current_pts_ns is None:
                    continue
                # The feed and pixels are published from the same decoded
                # source frame. Exact PTS matching prevents stale geometry.
                try:
                    if int(pts) != int(self.current_pts_ns):
                        continue
                except (TypeError, ValueError):
                    continue
                bbox = row.get("bbox")
                if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                    continue
                dimensions = bbox_coordinate_dimensions(
                    row, self.current_source_dimensions or
                    overlay_source_dimensions(self.camera_id, self.source_mode))
                if dimensions is None:
                    continue
                coordinate_width, coordinate_height = dimensions
                left, top, right, bottom = scale_overlay_bbox(
                    bbox, canvas_width, canvas_height, coordinate_width, coordinate_height
                )
                identity = self._matching_identity_row(row)
                if self.box_diagnostics:
                    if row.get("bbox_source") in {
                            "PGIE_SHADOW_FALLBACK", "SHADOW_CONTEXT_VERIFIED"}:
                        color, prefix = QColor(40, 220, 255), "FALLBACK"
                    else:
                        color, prefix = QColor(70, 240, 100), "TRACK"
                    label = f"{prefix} native:{native}"
                elif identity is None:
                    color = QColor(255, 210, 40)
                    label = f"Unknown | native:{native}"
                else:
                    color = QColor(80, 220, 80) if str(identity.get("application_id")) == "Person_01" else QColor(40, 165, 255)
                    label = f"{identity.get('application_id', 'Unknown')} | native:{native}"
                painter.setPen(QPen(color, 3))
                painter.drawRect(left, top, right - left, bottom - top)
                painter.drawText(left, max(20, top - 8), label)
        else:
            # Compatibility for older room-pair payloads that predate the
            # distinct tracked-object feed.
            for row in self.identity_rows:
                bbox = row.get("bbox")
                if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                    continue
                if self.camera_id in {"CAM-01", "CAM-04"}:
                    coordinate_width, coordinate_height = (
                        self.acceptance_source_dimensions or self.current_source_dimensions
                        or overlay_source_dimensions(self.camera_id, self.source_mode)
                    )
                else:
                    coordinate_width, coordinate_height = (
                        self.current_source_dimensions or (
                            self.current_image.width() if self.current_image is not None else canvas_width,
                            self.current_image.height() if self.current_image is not None else canvas_height,
                        )
                    )
                left, top, right, bottom = scale_overlay_bbox(
                    bbox,
                    canvas_width,
                    canvas_height,
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
                    painter.drawText(left, min(canvas_height - 6, bottom + 18), f"native:{native}")
        if self.acceptance_enabled:
            for row in self.acceptance_candidates:
                bbox = row.get("bbox")
                native = row.get("native_track_id")
                if not isinstance(bbox, (list, tuple)) or len(bbox) < 4 or native is None:
                    continue
                if row.get("camera_id") != self.camera_id:
                    continue
                dimensions = bbox_coordinate_dimensions(
                    row, self.acceptance_source_dimensions or self.current_source_dimensions or
                    overlay_source_dimensions(self.camera_id, self.source_mode))
                if dimensions is None:
                    continue
                coordinate_width, coordinate_height = dimensions
                left, top, right, bottom = scale_overlay_bbox(
                    bbox, canvas_width, canvas_height, coordinate_width, coordinate_height
                )
                tester = self.acceptance_labels.get((self.camera_id, int(native)))
                if tester:
                    painter.setPen(QPen(QColor(255, 70, 220), 4))
                    painter.drawRect(left, top, right - left, bottom - top)
                    painter.drawText(left, max(20, top - 10), f"{tester} · ACCEPTANCE ONLY")

    def _draw_overlays(self, pixmap: QPixmap) -> None:
        painter = QPainter(pixmap)
        self._paint_overlays(painter, pixmap.width(), pixmap.height())
        painter.end()

    def _matching_identity_row(self, tracked: dict) -> dict | None:
        try:
            native = int(tracked["native_track_id"])
            frame = int(tracked.get("frame", tracked.get("source_frame_number")))
        except (KeyError, TypeError, ValueError):
            return None
        matches = []
        for row in self.identity_rows:
            if row.get("camera_id") != self.camera_id:
                continue
            try:
                row_native = row.get("native_track_id")
                if row_native is None:
                    row_native = (row.get("debug") or {}).get("native_mv3dt_id")
                if row_native is None or int(row_native) != native:
                    continue
                row_frame = int(row["frame"])
                object_generation = tracked.get("track_generation")
                row_generation = row.get("track_generation")
                if (object_generation is not None and row_generation is not None
                        and int(object_generation) != int(row_generation)):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            delta = abs(row_frame - frame)
            if delta <= 2 and str(row.get("application_id", "")).startswith("Person_"):
                matches.append((delta, row))
        return min(matches, key=lambda item: item[0])[1] if matches else None

    def _update_tile_pixmap(self) -> None:
        if self.current_image is None:
            return
        target = self.video.size()
        image = self.current_image
        timing = self.video.preview_timing_context
        if self.last_version and self._last_tile_render_sequence == self.last_version:
            self.same_sequence_tile_renders += 1
        if target.width() > 0 and target.height() > 0:
            scale = min(target.width() / image.width(), target.height() / image.height())
            scaled_width = max(1, round(image.width() * scale))
            scaled_height = max(1, round(image.height() * scale))
            self.displayed_pixmap_size = QSize(scaled_width, scaled_height)
        # Use exactly one paint request for a new sequence.  In the visible
        # grid, repaint is intentionally synchronous so the latency endpoint
        # is the actual paint of this frame.  Queuing update() first would
        # leave a redundant same-sequence paint behind the immediate repaint.
        # A hidden tile keeps update() so it is ready when shown again.
        if self.video.isVisible():
            self.video.repaint()
        else:
            self.video.update()
        self._last_tile_render_sequence = self.last_version

    def _update_fullscreen_pixmap(self) -> None:
        if (self.fullscreen_video is None or self.fullscreen_current_image is None
                or self.fullscreen_dialog is None or not self.fullscreen_dialog.isVisible()):
            return
        target = self.fullscreen_video.size()
        source_pixmap = QPixmap.fromImage(self.fullscreen_current_image)
        pixmap = source_pixmap
        if target.width() > 0 and target.height() > 0:
            if pixmap.size() != target:
                pixmap = pixmap.scaled(
                    target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
            else:
                pixmap = QPixmap(pixmap)
        else:
            pixmap = pixmap.copy()
        saved_image = self.current_image
        saved_objects = self.tracked_objects
        saved_proposals = self.detector_proposals
        saved_pts = self.current_pts_ns
        saved_frame_num = self.current_source_frame_num
        saved_source_dimensions = self.current_source_dimensions
        self.current_image = self.fullscreen_current_image
        self.current_source_dimensions = self.fullscreen_source_dimensions
        self.tracked_objects = self.fullscreen_tracked_objects
        self.detector_proposals = []
        current_frame = self.fullscreen_current_frame or {}
        self.current_pts_ns = current_frame.get("pts_ns")
        self.current_source_frame_num = current_frame.get("source_frame_num")
        self._draw_overlays(pixmap)
        self.current_image = saved_image
        self.tracked_objects = saved_objects
        self.detector_proposals = saved_proposals
        self.current_pts_ns = saved_pts
        self.current_source_frame_num = saved_frame_num
        self.current_source_dimensions = saved_source_dimensions
        timing = getattr(self.fullscreen_video, "preview_timing_context", None)
        if (timing is not None and self.fullscreen_dialog is not None
                and self.fullscreen_dialog.isVisible()
                and target.width() >= 1000 and target.height() >= 700
                and "fullscreen" not in self.startup_capture_stages):
            fullscreen_image = pixmap.toImage()
            self._capture_startup_stage("fullscreen", fullscreen_image.bits().tobytes(), timing,
                                        fullscreen_image.bytesPerLine(), "RGB32",
                                        fullscreen_image.width(), fullscreen_image.height())
        self.fullscreen_video.setPixmap(pixmap)
        if self.fullscreen_video.isVisible():
            self.fullscreen_video.repaint()

    def _refresh_fullscreen_preview(self, notify_received_ns: int = 0) -> None:
        if (self.fullscreen_reader is None or self.fullscreen_video is None
                or self.fullscreen_dialog is None or not self.fullscreen_dialog.isVisible()):
            return
        poll_start = time.monotonic_ns()
        frame = self.fullscreen_reader.read_latest(max_age_sec=0.25,
                                                   after_sequence=self.fullscreen_last_sequence)
        if frame is None:
            if (self.fullscreen_last_frame_mono > 0.0
                    and time.monotonic() - self.fullscreen_last_frame_mono > 0.25):
                self.fullscreen_current_image = None
                self.fullscreen_current_payload = None
                self.fullscreen_tracked_objects = []
                self.fullscreen_video.clear()
                self.fullscreen_video.setText("STALE · waiting for current 1920×1080 frame")
            return
        self.fullscreen_last_sequence = frame.sequence
        receive_ns = time.monotonic_ns()
        snapshot = frame.tracker_snapshot
        if (frame.width != 1920 or frame.height != 1080 or not isinstance(snapshot, dict)
                or snapshot.get("camera_id") != self.camera_id
                or int(snapshot.get("pts_ns", -1)) != frame.pts_ns
                or int(snapshot.get("source_frame_num", -1)) != frame.source_frame_num):
            self.fullscreen_sync_misses += 1
            return
        image = QImage(frame.payload, frame.width, frame.height, frame.stride,
                       QImage.Format.Format_RGB32)
        if image.isNull():
            self.fullscreen_sync_misses += 1
            return
        self.fullscreen_current_payload = frame.payload
        self.fullscreen_current_image = image
        self.fullscreen_last_frame_mono = time.monotonic()
        self.fullscreen_source_dimensions = overlay_source_dimensions(self.camera_id, self.source_mode, snapshot)
        fullscreen_bbox_dimensions = snapshot_bbox_coordinate_dimensions(
            snapshot, self.fullscreen_source_dimensions)
        self.fullscreen_tracked_objects = snapshot_bbox_rows(
            snapshot, "objects", fullscreen_bbox_dimensions)
        for row in self.fullscreen_tracked_objects:
            row["source_frame_width"] = self.fullscreen_source_dimensions[0]
            row["source_frame_height"] = self.fullscreen_source_dimensions[1]
        self.fullscreen_current_frame = {
            "pts_ns": frame.pts_ns,
            "source_frame_num": frame.source_frame_num,
            "tracker_frame_num": frame.tracker_frame_num,
            "t7_notify_received_monotonic_ns": notify_received_ns,
        }
        paint_context = {
            "camera_id": self.camera_id,
            "sequence": frame.sequence,
            "source_frame_num": frame.source_frame_num,
            "tracker_frame_num": frame.tracker_frame_num,
            "pts_ns": frame.pts_ns,
            "bbox_metadata_pts_ns": int(snapshot["pts_ns"]),
            "bbox_metadata_frame_num": int(snapshot["frame_num"]),
            "bbox_metadata_source_frame_num": int(snapshot["source_frame_num"]),
            "t0_decoder_reference_monotonic_ns": frame.decoder_reference_ns,
            "t1_decoder_out_monotonic_ns": frame.decoder_out_ns,
            "t1_preview_prepare_start_monotonic_ns": frame.preview_prepare_start_ns,
            "t2_preview_prepare_complete_monotonic_ns": frame.preview_prepare_complete_ns,
            "t3_tracker_metadata_ready_monotonic_ns": frame.tracker_ready_ns,
            "t4_shared_publish_monotonic_ns": frame.timestamp_ns,
            "t5_ui_receive_monotonic_ns": receive_ns,
            "t5a_reader_poll_begin_monotonic_ns": poll_start,
            "preview_width": frame.width,
            "preview_height": frame.height,
            "bbox_metadata_objects": len(self.fullscreen_tracked_objects),
            "image_bbox_pts_match": frame.pts_ns == int(snapshot["pts_ns"]),
            "image_bbox_frame_match": frame.source_frame_num == int(snapshot["source_frame_num"]),
        }
        self.fullscreen_video.preview_timing_context = paint_context
        self.fullscreen_video.preview_timing_log = self.video.preview_timing_log
        self.fullscreen_video.preview_last_logged_sequence = 0
        self._update_fullscreen_pixmap()

    def _open_fullscreen(self) -> None:
        if self.current_image is None:
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
            dialog.finished.connect(self._on_fullscreen_closed)
            self.fullscreen_dialog = dialog
            self.fullscreen_video = label
        self.fullscreen_reader = PreviewFrameReader(FULLSCREEN_PREVIEW_PATH)
        self.fullscreen_last_sequence = 0
        _set_fullscreen_preview_request(self.camera_id)
        self.fullscreen_dialog.showFullScreen()
        if self.fullscreen_video is not None:
            self.fullscreen_video.setText("Preparing synchronized 1920×1080 preview…")

    def _on_fullscreen_closed(self, *_args) -> None:
        _set_fullscreen_preview_request(None)
        if self.fullscreen_reader is not None:
            self.fullscreen_reader.close()
            self.fullscreen_reader = None
        self.fullscreen_current_image = None
        self.fullscreen_current_payload = None
        self.fullscreen_last_frame_mono = 0.0
        self.fullscreen_tracked_objects = []
        self.fullscreen_current_frame = None
        self.fullscreen_last_sequence = 0

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_tile_pixmap()
        self._update_fullscreen_pixmap()

    def refresh(self, notify_received_ns: int = 0) -> None:
        """Read only the newest committed sequence and paint it immediately."""
        self._consume_latest_preview(notify_received_ns)
        self._refresh_fullscreen_preview(notify_received_ns)
        if self.last_version == 0:
            self.status.setText("CONNECTING")
        elif time.monotonic() - self.last_frame_mono > 0.25:
            self.status.setText("STALE · 0.0 FPS")
            self.current_image = None
            self.current_payload = None
            self.current_frame_pixmap = None
            self.tracked_objects = []
            self.detector_proposals = []
            self.current_pts_ns = None
            self.displayed_pixmap_size = None
            self.video.clear()

    def is_connected(self) -> bool:
        return self.last_version > 0 and time.monotonic() - self.last_frame_mono <= 0.25

    def close_reader(self) -> None:
        if self.fullscreen_dialog is not None:
            self.fullscreen_dialog.close()
        self._on_fullscreen_closed()
        self.preview_reader.close()
        if self.reader is not None:
            self.reader.stop()
            self.reader.join()

    def set_identity_rows(self, rows: list[dict]) -> None:
        self.identity_rows = list(rows)

    def set_tracked_objects(self, rows: list[dict], supported: bool) -> None:
        # Legacy room-pair object rows are no longer allowed to drive boxes;
        # the frame-locked native tracker IPC is read alongside each preview.
        self.tracked_object_feed_supported = True

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
        coordinate_width, coordinate_height = (
            self.acceptance_source_dimensions or self.current_source_dimensions
            or overlay_source_dimensions(self.camera_id, self.source_mode)
        )
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
        tile = getattr(self, "camera_tile", None)
        image = tile.current_image if isinstance(tile, CameraTile) else None
        if image is None:
            super().paintEvent(event)
        else:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor("black"))
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            widget_width, widget_height = self.width(), self.height()
            scale = min(widget_width / image.width(), widget_height / image.height())
            display_width = max(1, round(image.width() * scale))
            display_height = max(1, round(image.height() * scale))
            offset_x = (widget_width - display_width) // 2
            offset_y = (widget_height - display_height) // 2
            draw_start = time.monotonic_ns()
            painter.drawImage(QRectF(offset_x, offset_y, display_width, display_height), image)
            draw_done = time.monotonic_ns()
            if tile is not None:
                painter.save()
                painter.translate(offset_x, offset_y)
                tile._paint_overlays(painter, display_width, display_height)
                painter.restore()
            painter.end()
            overlay_done = time.monotonic_ns()
            tile.displayed_pixmap_size = QSize(display_width, display_height)
            timing = self.preview_timing_context
            if timing and timing.get("sequence") == tile.last_version:
                timing["t7d_qimage_draw_begin_monotonic_ns"] = draw_start
                timing["t7e_scale_completed_monotonic_ns"] = draw_done
                timing["scale_duration_ns"] = draw_done - draw_start
                timing["t7f_overlay_completed_monotonic_ns"] = overlay_done
                timing["overlay_duration_ns"] = overlay_done - draw_done
                timing["tile_source_pixmap_width"] = int(image.width())
                timing["tile_source_pixmap_height"] = int(image.height())
                timing["tile_target_width"] = int(widget_width)
                timing["tile_target_height"] = int(widget_height)
                timing["tile_scaled_width"] = int(display_width)
                timing["tile_scaled_height"] = int(display_height)
                timing["tile_transformation"] = "QPainter.SmoothPixmapTransform/direct-QImage"
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
        paint_ns = time.monotonic_ns()
        row = {
            **timing,
            "t8_ui_paint_monotonic_ns": paint_ns,
            "t8_ui_paint_wall_ns": time.time_ns(),
            "widget_width": int(self.width()),
            "widget_height": int(self.height()),
        }
        row["t6_ui_paint_monotonic_ns"] = row["t8_ui_paint_monotonic_ns"]
        _write_timing_row(output, row)
        if (isinstance(tile, CameraTile) and tile.startup_capture_dir is not None
                and "displayed" not in tile.startup_capture_stages
                and not getattr(self, "_capture_display_pending", False)):
            self._capture_display_pending = True
            QTimer.singleShot(0, self._capture_display_after_paint)

    def _capture_display_after_paint(self) -> None:
        tile = getattr(self, "camera_tile", None)
        timing = self.preview_timing_context
        if (not isinstance(tile, CameraTile) or not timing
                or "displayed" in tile.startup_capture_stages):
            self._capture_display_pending = False
            return
        try:
            rendered = self.grab().toImage()
            tile._capture_startup_stage("displayed", rendered.bits().tobytes(), timing,
                rendered.bytesPerLine(), "RGB32", rendered.width(), rendered.height())
        except RuntimeError:
            pass
        finally:
            self._capture_display_pending = False

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
        self.preview_notify_receiver: PreviewFrameNotifyReceiver | None = None
        self.preview_notify_notifier: QSocketNotifier | None = None
        try:
            self.preview_notify_receiver = PreviewFrameNotifyReceiver()
            self.preview_notify_notifier = QSocketNotifier(
                self.preview_notify_receiver.fileno(), QSocketNotifier.Type.Read, self
            )
            self.preview_notify_notifier.activated.connect(self._on_preview_notify)
        except (OSError, RuntimeError) as exc:
            # The 16 ms frame timer remains a safe latest-only fallback; the
            # UI never opens a camera source if notification setup fails.
            print(f"preview notification unavailable; retaining timer refresh: {exc}", file=sys.stderr)
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

    def _on_preview_notify(self, *_args) -> None:
        receiver = self.preview_notify_receiver
        if receiver is None:
            return
        received_ns = time.monotonic_ns()
        try:
            events = receiver.drain()
        except OSError:
            return
        for kind, source_id in sorted(events):
            camera_id = f"CAM-{source_id + 1:02d}"
            tile = self.tiles.get(camera_id)
            if tile is None:
                continue
            tile.preview_notify_received_ns = received_ns
            if kind == "tile":
                tile.refresh(received_ns)
            elif kind == "fullscreen":
                tile._refresh_fullscreen_preview(received_ns)

    def connection_counts(self) -> tuple[int, int]:
        return sum(tile.is_connected() for tile in self.tiles.values()), len(self.tiles)

    def set_identity_snapshot(self, people: list[dict]) -> None:
        by_camera = {}
        for person in people:
            by_camera.setdefault(person.get("camera_id"), []).append(person)
        for camera_id, tile in self.tiles.items():
            tile.set_identity_rows(by_camera.get(camera_id, []))

    def set_tracked_object_snapshot(self, objects: list[dict], supported: bool) -> None:
        for camera_id, tile in self.tiles.items():
            tile.set_tracked_objects(objects, supported)

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
        if self.preview_notify_notifier is not None:
            self.preview_notify_notifier.setEnabled(False)
            self.preview_notify_notifier.deleteLater()
            self.preview_notify_notifier = None
        if self.preview_notify_receiver is not None:
            self.preview_notify_receiver.close()
            self.preview_notify_receiver = None
