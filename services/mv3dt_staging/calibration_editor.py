"""Offline, versioned Entrance Room floor-correspondence editor.

This is an operator annotation tool only. It does not alter AMC project files,
estimate camera intrinsics, compensate distortion, or activate map projection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSlider, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from services.mv3dt_staging.calibration_geometry import diagnose_homography, load_amc_alignment
from services.mv3dt_staging.live_tracking.native_frame_bridge import (
    DETECTOR_HEIGHT,
    DETECTOR_WIDTH,
    NativeFullFovScale,
    SOURCE_SIZE,
)


CAMERA_COLOR = QColor("#ffbd59")
HELDOUT_COLOR = QColor("#ed65e8")
SELECTED_COLOR = QColor("#41f0d0")
REFERENCE_COLOR = QColor("#ff6565")
LIVE_NATIVE_SPACE = "live_native_scale_only_1920x1080"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata(video_path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"cannot open offline video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return {"path": str(video_path.resolve()), "sha256": _sha256(video_path),
            "width": width, "height": height, "fps": fps, "frame_count": frames}


def _frame_in_coordinate_space(bgr, camera_key: str, coordinate_space: str,
                              scaler: NativeFullFovScale | None = None):
    """Return a frame in the selected point-coordinate raster.

    Native mode deliberately calls the same validator/resizer as LIVE. This
    makes point clicks and detector foot anchors share one 1920x1080 plane.
    """
    if coordinate_space != LIVE_NATIVE_SPACE:
        return bgr
    camera_id = {"cam02": "CAM-02", "cam05": "CAM-05"}.get(camera_key)
    if camera_id is None:
        raise ValueError(f"unsupported Entrance editor camera: {camera_key}")
    return (scaler or NativeFullFovScale(camera_id))(bgr)


def _eligible_new_holdout(point: dict[str, Any]) -> bool:
    return (point.get("provenance") == "operator_selected_new_floor_landmark"
            and point.get("prior_alignment_reference") is not True)


def load_editor_points(
    path: Path,
    *,
    video_metadata: dict[str, dict[str, Any]],
    map_metadata: dict[str, Any],
    coordinate_space: str,
) -> list[dict[str, Any]]:
    """Load a legacy or versioned alignment without mixing image spaces.

    Camera coordinates whose saved source hash/space does not match the video
    currently shown are retained under ``legacy_camera_coordinates`` and
    hidden from the image/fit. Map clicks are retained as references but must
    be explicitly confirmed in the current layout before fitting.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        points = load_amc_alignment(path)
        saved_sources: dict[str, dict[str, Any]] = {}
    elif isinstance(payload, dict) and isinstance(payload.get("points"), list):
        points = copy.deepcopy(payload["points"])
        saved_sources = payload.get("source_files") or {}
    else:
        raise ValueError("alignment must be an AMC triplet list or a versioned editor object")

    for point in points:
        # Every point loaded from the reference artifact was already consumed
        # in that artifact's fit/review history. It remains useful as a map
        # reference, but can never become a new independent holdout here.
        point["prior_alignment_reference"] = True
        legacy_camera = dict(point.get("legacy_camera_coordinates") or {})
        for key in ("cam02", "cam05"):
            xy = point.get(key)
            source = saved_sources.get(key) or {}
            saved_hash = point.get(f"{key}_source_sha256") or source.get("sha256")
            saved_space = point.get(f"{key}_coordinate_space") or payload.get("coordinate_space")
            matches = (
                isinstance(xy, (tuple, list)) and len(xy) == 2
                and saved_hash == video_metadata[key]["sha256"]
                and saved_space == coordinate_space
            )
            if xy is not None and not matches:
                legacy_camera[key] = {
                    "xy": copy.deepcopy(xy),
                    "source_sha256": saved_hash,
                    "source_path": source.get("path"),
                    "coordinate_space": saved_space or "unrecorded_file_pixel_space",
                    "frame_index": point.get(f"{key}_frame_index"),
                }
                point[key] = None
                point[f"{key}_source_sha256"] = None
                point[f"{key}_coordinate_space"] = None
                point[f"{key}_frame_index"] = None
        if legacy_camera:
            point["legacy_camera_coordinates"] = legacy_camera

        # Preserve old map pixels, but require an explicit operator review on
        # this exact layout before they participate in a new fit.
        point["layout_verified"] = bool(
            point.get("layout_verified")
            and point.get("layout_source_sha256") == map_metadata["sha256"]
        )
        if point.get("layout") is not None and not point.get("layout_source_sha256"):
            prior_layout = saved_sources.get("layout") or payload.get("layout") or {}
            point["legacy_layout_reference"] = {
                "xy": copy.deepcopy(point["layout"]),
                "source_sha256": prior_layout.get("sha256"),
                "coordinate_space": "layout_pixel_reference_unconfirmed",
            }
        point.setdefault("role", "fit")
        point.setdefault("provenance", "saved_amc_alignment_used_by_prior_fit")
        point.setdefault("independent_of_prior_fit", False)
    return points


class ImagePointView(QWidget):
    """Aspect-preserving image widget with coordinate-aware point drag/click."""

    point_changed = Signal(str, str, float, float)
    point_selected = Signal(str)

    def __init__(self, panel_key: str, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.panel_key = panel_key
        self.title = title
        self.pixmap = QPixmap()
        self.points: list[dict[str, Any]] = []
        self.selected_id: str | None = None
        self._drag_id: str | None = None
        self._image_rect = QRectF()
        self.setMinimumSize(350, 360)
        self.setMouseTracking(True)

    def set_image(self, image: QImage) -> None:
        self.pixmap = QPixmap.fromImage(image) if not image.isNull() else QPixmap()
        self.update()

    def set_points(self, points: list[dict[str, Any]], selected_id: str | None) -> None:
        self.points = points
        self.selected_id = selected_id
        self.update()

    def _target_rect(self) -> QRectF:
        area = QRectF(self.rect()).adjusted(8, 34, -8, -8)
        if self.pixmap.isNull():
            return area
        size = self.pixmap.size()
        aspect = size.width() / max(1, size.height())
        if area.width() / max(1.0, area.height()) > aspect:
            width, height = area.height() * aspect, area.height()
        else:
            width, height = area.width(), area.width() / max(aspect, 1e-8)
        return QRectF(area.center().x() - width / 2, area.center().y() - height / 2, width, height)

    def _view_xy(self, image_xy: list[float]) -> QPointF:
        rect = self._target_rect()
        return QPointF(rect.left() + image_xy[0] / max(1, self.pixmap.width()) * rect.width(),
                       rect.top() + image_xy[1] / max(1, self.pixmap.height()) * rect.height())

    def _image_xy(self, position: QPointF) -> list[float] | None:
        rect = self._target_rect()
        if self.pixmap.isNull() or not rect.contains(position):
            return None
        x = (position.x() - rect.left()) / rect.width() * self.pixmap.width()
        y = (position.y() - rect.top()) / rect.height() * self.pixmap.height()
        return [float(min(self.pixmap.width() - 1, max(0.0, x))),
                float(min(self.pixmap.height() - 1, max(0.0, y)))]

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101820"))
        painter.setPen(QColor("#f2f6f8"))
        painter.drawText(QRectF(8, 5, self.width() - 16, 24), self.title)
        target = self._target_rect()
        self._image_rect = target
        if self.pixmap.isNull():
            painter.drawText(target, Qt.AlignmentFlag.AlignCenter, "Image unavailable")
            return
        painter.drawPixmap(target, self.pixmap, QRectF(self.pixmap.rect()))
        for point in self.points:
            xy = point.get(self.panel_key)
            if not isinstance(xy, (list, tuple)) or len(xy) != 2:
                continue
            p = self._view_xy([float(xy[0]), float(xy[1])])
            selected = point.get("point_id") == self.selected_id
            color = SELECTED_COLOR if selected else (
                HELDOUT_COLOR if point.get("role") == "heldout" else
                REFERENCE_COLOR if self.panel_key == "layout" and not point.get("layout_verified") else CAMERA_COLOR
            )
            painter.setPen(QPen(QColor("#17232c"), 2))
            painter.setBrush(color)
            painter.drawEllipse(p, 7 if selected else 5, 7 if selected else 5)
            painter.setPen(QPen(color, 1))
            painter.drawText(QRectF(p.x() + 7, p.y() - 14, 44, 18), str(point.get("point_id", "")))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        position = event.position()
        nearest: tuple[float, str] | None = None
        for point in self.points:
            xy = point.get(self.panel_key)
            if not isinstance(xy, (list, tuple)) or len(xy) != 2:
                continue
            marker = self._view_xy([float(xy[0]), float(xy[1])])
            distance = ((marker.x() - position.x()) ** 2 + (marker.y() - position.y()) ** 2) ** 0.5
            pid = str(point.get("point_id", ""))
            if distance <= 14 and pid and (nearest is None or distance < nearest[0]):
                nearest = (distance, pid)
        if nearest is not None:
            self._drag_id = nearest[1]
            self.selected_id = nearest[1]
            self.point_selected.emit(nearest[1])
            self.update()
            event.accept()
            return
        image_xy = self._image_xy(position)
        if image_xy is not None and self.selected_id:
            self.point_changed.emit(self.panel_key, self.selected_id, image_xy[0], image_xy[1])
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_id is None or not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        image_xy = self._image_xy(event.position())
        if image_xy is not None:
            self.point_changed.emit(self.panel_key, self._drag_id, image_xy[0], image_xy[1])

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_id = None


class CalibrationEditor(QMainWindow):
    def __init__(self, *, cam02: Path, cam05: Path, layout: Path,
                 reference_alignment: Path, output_dir: Path,
                 coordinate_space: str = "project_input_20fps_1920x1080") -> None:
        super().__init__()
        self.setWindowTitle(
            "Entrance Room — LIVE scale-only calibration editor · UNCALIBRATED"
            if coordinate_space == LIVE_NATIVE_SPACE else
            "Entrance Room — rectified MV3DT floor correspondence editor"
            if coordinate_space == "rectified_mv3dt_30fps_1920x1080" else
            "Entrance Room — Offline floor correspondence editor"
        )
        self.resize(1660, 1120)
        self.video_paths = {"cam02": cam02, "cam05": cam05}
        self.layout_path = layout
        self.reference_alignment_path = reference_alignment
        self.output_dir = output_dir
        self.coordinate_space = coordinate_space
        self.output_stem = "LIVE_NATIVE_ALIGNMENT" if coordinate_space == LIVE_NATIVE_SPACE else "alignment"
        self.coordinate_dimensions: dict[str, tuple[int, int]] = {}
        self.frame_scalers: dict[str, NativeFullFovScale] = {}
        self.captures: dict[str, cv2.VideoCapture] = {}
        self.video_metadata: dict[str, dict[str, Any]] = {}
        self.last_frames: dict[str, int] = {"cam02": 0, "cam05": 0}
        self.frame_images: dict[str, QImage] = {}
        for key, path in self.video_paths.items():
            self.video_metadata[key] = _metadata(path)
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise ValueError(f"cannot decode {key} video")
            self.captures[key] = capture
            if coordinate_space == LIVE_NATIVE_SPACE:
                camera_id = "CAM-02" if key == "cam02" else "CAM-05"
                expected_width, expected_height = SOURCE_SIZE[camera_id]
                meta = self.video_metadata[key]
                if (meta["width"], meta["height"]) != (expected_width, expected_height):
                    raise ValueError(
                        f"{camera_id} scale-only editor requires native {expected_width}x{expected_height}; "
                        f"got {meta['width']}x{meta['height']}"
                    )
                self.frame_scalers[key] = NativeFullFovScale(camera_id)
                self.coordinate_dimensions[key] = (DETECTOR_WIDTH, DETECTOR_HEIGHT)
            else:
                self.coordinate_dimensions[key] = (self.video_metadata[key]["width"],
                                                   self.video_metadata[key]["height"])

        self.map_image = QImage(str(layout))
        if self.map_image.isNull():
            raise ValueError(f"cannot read room layout: {layout}")
        self.map_metadata = {"path": str(layout.resolve()), "sha256": _sha256(layout),
                             "width": self.map_image.width(), "height": self.map_image.height()}
        if coordinate_space == "rectified_mv3dt_30fps_1920x1080":
            for key in ("cam02", "cam05"):
                meta = self.video_metadata[key]
                if (meta["width"], meta["height"]) != (1920, 1080) or meta["frame_count"] != 5400:
                    raise ValueError(f"{key} is not the verified 5400-frame 1920x1080 rectified MV3DT input")
        self.points = load_editor_points(
            reference_alignment, video_metadata=self.video_metadata,
            map_metadata=self.map_metadata, coordinate_space=self.coordinate_space,
        )
        self.point_by_id = {str(point["point_id"]): point for point in self.points}
        reference_payload = json.loads(reference_alignment.read_text(encoding="utf-8"))
        self.reference_point_count = (len(reference_payload) if isinstance(reference_payload, list)
                                      else len(reference_payload.get("points", [])))
        self.selected_id: str | None = (
            "P03" if "P03" in self.point_by_id else
            str(self.points[0]["point_id"]) if self.points else None
        )
        self.alignment_hash = _sha256(reference_alignment)
        if coordinate_space == LIVE_NATIVE_SPACE:
            cam02_title = "CAM-02 · native source resized as LIVE · 1920×1080"
            cam05_title = "CAM-05 · native source resized as LIVE · 1920×1080"
        elif coordinate_space == "rectified_mv3dt_30fps_1920x1080":
            cam02_title = "CAM-02 · exact rectified MV3DT input · 1920×1080"
            cam05_title = "CAM-05 · exact rectified MV3DT input · 1920×1080"
        else:
            cam02_title = "CAM-02 · selected project input file pixels"
            cam05_title = "CAM-05 · selected project input file pixels"
        self.cam02_view = ImagePointView("cam02", cam02_title)
        self.cam05_view = ImagePointView("cam05", cam05_title)
        self.map_view = ImagePointView("layout", "Entrance Room layout · pixel coordinates")
        self.map_view.set_image(self.map_image)
        self.views = {"cam02": self.cam02_view, "cam05": self.cam05_view, "layout": self.map_view}
        for view in self.views.values():
            view.point_changed.connect(self._on_point_changed)
            view.point_selected.connect(self.select_point)

        header = QLabel("Entrance Room — manual floor correspondences")
        header.setStyleSheet("font-size:22px;font-weight:700")
        if coordinate_space == LIVE_NATIVE_SPACE:
            warning_text = (
                "OFFLINE / EXPERIMENTAL — original camera recordings are shown after the exact LIVE full-frame resize "
                "to 1920×1080 (CAM-02 0.60×, CAM-05 0.75×, OpenCV INTER_AREA). No crop, padding, undistortion, "
                "rectification or projective warp is applied. Camera coordinates are saved in the LIVE inference plane; "
                "source hashes and original per-camera frame indexes are recorded. alignment_v0011 camera clicks are "
                "preserved as legacy references only and must be reselected. Its existing map marks are shown as map "
                "references, not surveyed ground truth. Reselect the eight FIT locations and add at least two NEW "
                "independent HELD-OUT floor targets. Frame indexes do not establish synchronized exposure."
            )
        elif coordinate_space == "rectified_mv3dt_30fps_1920x1080":
            warning_text = (
                "OFFLINE / EXPERIMENTAL — these are the exact 1920×1080 rectified clips used by the saved MV3DT replay; "
                "tracker boxes and footLocation2D use this image plane. No old source-to-rectified remap is approximated. "
                "Old camera clicks from different file hashes are hidden and preserved as legacy references; reselect them here. "
                "The eight prior points were fit points, not independent validation. Camera frame indexes are per-file and do not prove synchronized exposure."
            )
        else:
            warning_text = (
                "OFFLINE / EXPERIMENTAL — displayed source-file pixel space only. Imported AMC points are historical fit references, "
                "not independent validation. Camera timelines are independent and do not prove synchronized exposure."
            )
        warning = QLabel(warning_text + " Mark the same fixed floor feature in both camera frames and the layout. Fit/LOO values are pixels, not physical distances.")
        warning.setWordWrap(True)
        warning.setStyleSheet("background:#5a3512;color:#fff2d6;padding:9px")
        panels = QHBoxLayout()
        panels.addWidget(self._camera_panel("cam02", self.cam02_view), 1)
        panels.addWidget(self._camera_panel("cam05", self.cam05_view), 1)
        panels.addWidget(self.map_view, 1)

        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels(["Point", "Role", "CAM-02 X", "CAM-02 Y", "Frame", "CAM-05 X", "CAM-05 Y", "Frame", "Map X", "Map Y", "Coordinate status"])
        self.table.setMaximumHeight(220)
        self.table.currentCellChanged.connect(self._on_table_selection)
        self.status = QLabel("Reference alignment loaded read-only. Pixels from a different source hash/coordinate space are hidden from fitting; saved versions are never overwritten.")
        self.status.setWordWrap(True)
        self.diagnostics = QLabel()
        self.diagnostics.setWordWrap(True)
        self.diagnostics.setStyleSheet("background:#202e38;color:#f0f5f8;padding:8px;font-family:monospace")

        buttons = QHBoxLayout()
        self.add_button = QPushButton("Add correspondence")
        self.delete_button = QPushButton("Delete selected point")
        self.fit_button = QPushButton("Mark selected as FIT")
        self.held_button = QPushButton("Mark selected as HELD-OUT")
        self.confirm_new_holdout_button = QPushButton("Confirm new independent floor target")
        self.confirm_layout_button = QPushButton("Confirm retained map mark")
        self.save_button = QPushButton("Save new version (never overwrite)")
        self.add_button.clicked.connect(self.add_point)
        self.delete_button.clicked.connect(self.delete_point)
        self.fit_button.clicked.connect(lambda: self.set_role("fit"))
        self.held_button.clicked.connect(lambda: self.set_role("heldout"))
        self.confirm_new_holdout_button.clicked.connect(self.confirm_new_holdout)
        self.confirm_layout_button.clicked.connect(self.confirm_layout_point)
        self.save_button.clicked.connect(self.save_version)
        for button in (self.add_button, self.delete_button, self.fit_button, self.held_button,
                       self.confirm_new_holdout_button, self.confirm_layout_button, self.save_button):
            buttons.addWidget(button)

        root = QVBoxLayout()
        root.addWidget(header)
        root.addWidget(warning)
        root.addLayout(panels, 1)
        root.addWidget(self.table)
        root.addLayout(buttons)
        root.addWidget(self.status)
        root.addWidget(self.diagnostics)
        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)
        # Frame indices from a different source file are not transferable. Start
        # each exact tracking input at frame zero and require independent seeking.
        self._load_frame("cam02", 0)
        self._load_frame("cam05", 0)
        if self.selected_id == "P03":
            self.status.setText(
                "P03 highlighted for review: prior LOO errors were large in both cameras; it is not declared wrong or removed. "
                + ("Reselect its camera coordinates in this LIVE scale-only 1920×1080 canvas."
                   if coordinate_space == LIVE_NATIVE_SPACE else
                   "Camera coordinates require exact rectified-frame reselection.")
            )
        self._refresh()
        if self.selected_id in self.point_by_id:
            self.table.selectRow(next(i for i, point in enumerate(self.points)
                                      if str(point.get("point_id")) == self.selected_id))

    def _camera_panel(self, key: str, view: ImagePointView) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        camera_label = QLabel(("CAM-02 raw source → live 1920×1080 scale-only" if key == "cam02" else
                               "CAM-05 raw source → live 1920×1080 scale-only") if self.coordinate_space == LIVE_NATIVE_SPACE else
                              ("CAM-02 exact rectified MV3DT file" if key == "cam02" else
                               "CAM-05 exact rectified MV3DT file") if self.coordinate_space == "rectified_mv3dt_30fps_1920x1080" else
                              ("CAM-02 source input" if key == "cam02" else "CAM-05 source input"))
        camera_label.setStyleSheet("font-weight:700")
        frame = QSpinBox()
        frame.setRange(0, max(0, self.video_metadata[key]["frame_count"] - 1))
        frame.setValue(0)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(frame.minimum(), frame.maximum())
        frame.valueChanged.connect(slider.setValue)
        slider.valueChanged.connect(frame.setValue)
        frame.valueChanged.connect(lambda value, camera=key: self._load_frame(camera, value))
        slider.setValue(frame.value())
        fps = self.video_metadata[key]["fps"]
        readout = QLabel(f"File frame index · {fps:g} fps metadata · independent timeline; frame index is not exposure sync")
        self._frame_readouts = getattr(self, "_frame_readouts", {})
        self._frame_readouts[key] = readout
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Frame"))
        controls.addWidget(frame)
        layout.addWidget(camera_label)
        layout.addWidget(view, 1)
        layout.addLayout(controls)
        layout.addWidget(slider)
        layout.addWidget(readout)
        return panel

    def _load_frame(self, key: str, frame_index: int) -> None:
        capture = self.captures[key]
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, bgr = capture.read()
        if not ok:
            self.status.setText(f"Could not decode {key} frame {frame_index}; source is unchanged.")
            return
        decoded_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if decoded_index != int(frame_index):
            self.status.setText(
                f"Frame seek mismatch for {key}: requested {frame_index}, decoded {decoded_index}; point selection is disabled until an exact frame is shown."
            )
            return
        # This is the exact scale-only bridge used by live inference. Source
        # order and frame index are unchanged; no AMC rectification is applied.
        bgr = _frame_in_coordinate_space(bgr, key, self.coordinate_space,
                                         self.frame_scalers.get(key))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, int(rgb.strides[0]), QImage.Format.Format_RGB888).copy()
        self.last_frames[key] = int(frame_index)
        self.frame_images[key] = image
        self.views[key].set_image(image)
        if key in getattr(self, "_frame_readouts", {}):
            self._frame_readouts[key].setText(
                f"Displayed frame {frame_index} · {self.video_metadata[key]['fps']:g} fps · file index only; not exposure sync"
            )
        self._refresh()

    def _on_table_selection(self, row: int, _col: int, _prev_row: int, _prev_col: int) -> None:
        if 0 <= row < len(self.points):
            self.select_point(str(self.points[row]["point_id"]))

    def select_point(self, point_id: str) -> None:
        if point_id not in self.point_by_id:
            return
        self.selected_id = point_id
        self._refresh_views()

    def _on_point_changed(self, panel_key: str, point_id: str, x: float, y: float) -> None:
        point = self.point_by_id.get(point_id)
        if point is None:
            return
        if panel_key == "layout":
            if point.get("layout") is not None and "previous_layout_coordinate" not in point:
                point["previous_layout_coordinate"] = copy.deepcopy(point["layout"])
            point["layout_source_sha256"] = self.map_metadata["sha256"]
            point["layout_coordinate_space"] = "entrance_layout_pixel_xy_down"
            point["layout_verified"] = True
            point["layout_provenance"] = "operator_selected_in_current_editor"
        else:
            if point.get(panel_key) is not None and f"previous_{panel_key}_coordinate" not in point:
                point[f"previous_{panel_key}_coordinate"] = copy.deepcopy(point[panel_key])
            point[f"{panel_key}_source_sha256"] = self.video_metadata[panel_key]["sha256"]
            point[f"{panel_key}_source_path"] = self.video_metadata[panel_key]["path"]
            point[f"{panel_key}_coordinate_space"] = self.coordinate_space
            point[f"{panel_key}_dimensions"] = list(self.coordinate_dimensions[panel_key])
            point[f"{panel_key}_source_dimensions"] = [self.video_metadata[panel_key]["width"],
                                                         self.video_metadata[panel_key]["height"]]
            point[f"{panel_key}_selection_provenance"] = "operator_selected_on_exact_displayed_file_frame"
        point[panel_key] = [round(float(x), 3), round(float(y), 3)]
        if panel_key in self.video_metadata:
            point[f"{panel_key}_frame_index"] = self.last_frames[panel_key]
        self._refresh()

    def add_point(self) -> None:
        numbers = [int(match.group(1)) for item in self.points
                   if (match := re.fullmatch(r"P(\d+)", str(item.get("point_id", ""))))]
        point_id = f"P{max(numbers, default=0) + 1:02d}"
        point = {"point_id": point_id, "cam02": None, "cam05": None, "layout": None,
                 "role": "fit", "provenance": "operator_selected_new_floor_landmark",
                 "cam02_frame_index": None, "cam05_frame_index": None,
                 "layout_verified": False, "independent_of_prior_fit": False,
                 "note": "Select the same fixed physical floor feature in both exact camera files and the layout."}
        self.points.append(point)
        self.point_by_id[point_id] = point
        self.selected_id = point_id
        self.status.setText(f"{point_id} added. Click its floor location in CAM-02, CAM-05 and layout; frame indices are independent.")
        self._refresh()
        self.table.selectRow(len(self.points) - 1)

    def delete_point(self) -> None:
        if self.selected_id is None:
            return
        point = self.point_by_id.pop(self.selected_id, None)
        if point is None:
            return
        self.points.remove(point)
        self.selected_id = str(self.points[0]["point_id"]) if self.points else None
        self.status.setText("Removed from this unsaved editor version only; source AMC evidence is unchanged.")
        self._refresh()

    def set_role(self, role: str) -> None:
        if role not in {"fit", "heldout"} or self.selected_id is None:
            return
        point = self.point_by_id[self.selected_id]
        if role == "heldout" and (point.get("prior_alignment_reference") is True
                                   or "prior_fit" in str(point.get("provenance", ""))):
            self.status.setText("This point is from the prior alignment and cannot become an independent holdout. Add a NEW physical floor target instead.")
            return
        point["role"] = role
        if role == "heldout":
            point["independent_of_prior_fit"] = False
        self.status.setText(
            "Role changed in the staging candidate. A prior-AMC point is still not independent ground truth, even if held out here."
            if self.point_by_id[self.selected_id].get("provenance") == "saved_amc_alignment_used_by_prior_fit"
            else f"{self.selected_id} is now {role.upper()}; held-out points are excluded from homography fitting."
        )
        self._refresh()

    def confirm_new_holdout(self) -> None:
        if self.selected_id is None:
            return
        point = self.point_by_id[self.selected_id]
        if not _eligible_new_holdout(point):
            self.status.setText("Only a newly added landmark can be attested as independent of the prior eight fit points.")
            return
        if not point.get("layout_verified") or any(
            point.get(key) is None or point.get(f"{key}_source_sha256") != self.video_metadata[key]["sha256"]
            or point.get(f"{key}_coordinate_space") != self.coordinate_space
            for key in ("cam02", "cam05")
        ):
            self.status.setText("Select the landmark in both camera files and the layout before marking it held-out.")
            return
        point["role"] = "heldout"
        point["independent_of_prior_fit"] = True
        point["heldout_attestation"] = "operator_attests_new_physical_target_not_in_prior_fit_eight"
        self.status.setText("New target marked HELD-OUT by operator attestation. Its pixel error is still not a surveyed metric-accuracy result.")
        self._refresh()

    def confirm_layout_point(self) -> None:
        if self.selected_id is None:
            return
        point = self.point_by_id[self.selected_id]
        if point.get("layout") is None:
            self.status.setText("Selected point has no retained map coordinate; click its physical floor location on the map.")
            return
        point["layout_verified"] = True
        point["layout_source_sha256"] = self.map_metadata["sha256"]
        point["layout_coordinate_space"] = "entrance_layout_pixel_xy_down"
        point["layout_provenance"] = "operator_confirmed_retained_map_mark_in_current_layout"
        self.status.setText("Retained map pixel confirmed by operator for this point; this is not a survey or physical-scale validation.")
        self._refresh()

    def _diagnostic_points_for(self, camera_key: str) -> list[dict[str, Any]]:
        rows = []
        for original in self.points:
            point = copy.deepcopy(original)
            valid_camera = (
                point.get(camera_key) is not None
                and point.get(f"{camera_key}_source_sha256") == self.video_metadata[camera_key]["sha256"]
                and point.get(f"{camera_key}_coordinate_space") == self.coordinate_space
            )
            if not valid_camera:
                point[camera_key] = None
            valid_layout = (
                point.get("layout") is not None
                and point.get("layout_verified") is True
                and point.get("layout_source_sha256") == self.map_metadata["sha256"]
            )
            if not valid_layout:
                point["layout"] = None
            if point.get("role") == "heldout" and point.get("independent_of_prior_fit") is not True:
                point["role"] = "pending_independent_review"
            rows.append(point)
        return rows

    def _refresh_views(self) -> None:
        for key, view in self.views.items():
            view.set_points(self.points, self.selected_id)

    def _refresh(self) -> None:
        self._refresh_views()
        self.table.setRowCount(len(self.points))
        for index, point in enumerate(self.points):
            legacy = point.get("legacy_camera_coordinates", {})
            current_camera_coords = all(
                point.get(key) is not None
                and point.get(f"{key}_source_sha256") == self.video_metadata[key]["sha256"]
                and point.get(f"{key}_coordinate_space") == self.coordinate_space
                for key in ("cam02", "cam05")
            )
            coordinate_status = (
                "LIVE-scale set; legacy kept" if legacy and current_camera_coords and self.coordinate_space == LIVE_NATIVE_SPACE
                else "reselect LIVE-scale camera pixels" if legacy and self.coordinate_space == LIVE_NATIVE_SPACE
                else "rectified set; legacy kept" if legacy and current_camera_coords
                else "reselect rectified cameras; confirm map" if legacy
                else "complete in current files" if current_camera_coords
                and point.get("layout_verified")
                else "needs camera/map selection"
            )
            values = [point.get("point_id"), point.get("role", "fit"),
                      *(point.get("cam02") or [None, None]), point.get("cam02_frame_index"),
                      *(point.get("cam05") or [None, None]), point.get("cam05_frame_index"),
                      *(point.get("layout") or [None, None]), coordinate_status]
            for column, value in enumerate(values):
                text = "—" if value is None else (f"{float(value):.3f}" if isinstance(value, float) else str(value))
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                legacy_camera = legacy.get("cam02" if column in (2, 3, 4) else "cam05")
                if column in (2, 3, 4, 5, 6, 7) and legacy_camera:
                    coordinate_note = ("Old rectified/source pixel retained for audit only; it is not usable in this LIVE scale-only canvas. "
                                      if self.coordinate_space == LIVE_NATIVE_SPACE else
                                      "Previous camera pixel retained for audit only; source hash/coordinate space does not match this canvas. ")
                    item.setToolTip(
                        coordinate_note +
                        f"Legacy xy={legacy_camera.get('xy')}; old frame={legacy_camera.get('frame_index')}; old SHA-256={legacy_camera.get('source_sha256')}. Reselect here."
                    )
                if point.get("point_id") == "P03":
                    item.setToolTip((item.toolTip() + " " if item.toolTip() else "") +
                                    "P03 highlighted for review due to large prior LOO residuals in both cameras; not declared incorrect.")
                self.table.setItem(index, column, item)
            if point.get("role") == "heldout":
                for column in range(self.table.columnCount()):
                    self.table.item(index, column).setBackground(HELDOUT_COLOR)
        self.table.setColumnWidth(0, 62)
        self.table.setColumnWidth(1, 72)
        for column in (2, 3, 5, 6, 8, 9):
            self.table.setColumnWidth(column, 112)
        self.table.setColumnWidth(4, 60)
        self.table.setColumnWidth(7, 60)
        self.table.setColumnWidth(10, 235)
        self._render_diagnostics()

    def _render_diagnostics(self) -> None:
        source_shapes = {key: (self.coordinate_dimensions[key][1], self.coordinate_dimensions[key][0])
                         for key in self.video_metadata}
        layout_shape = (self.map_metadata["height"], self.map_metadata["width"])
        reports = [diagnose_homography(self._diagnostic_points_for(key), key, source_shape_hw=source_shapes[key],
                                       layout_shape_hw=layout_shape) for key in ("cam02", "cam05")]
        lines = ["FLOOR-MAP CALIBRATION STATUS: UNVERIFIED / EXPERIMENTAL (pixel units only)"]
        for key, report in zip(("CAM-02", "CAM-05"), reports, strict=True):
            loo = report["leave_one_out_rms_layout_px"]
            train = report["training_rms_layout_px"]
            held = report.get("heldout_rms_layout_px")
            lines.append(
                f"{key} image→layout: fit={report['fit_point_count']}, held-out={report['heldout_point_count']}, "
                f"train RMS={_fmt(train)} px, LOO RMS={_fmt(loo)} px, held-out RMS={_fmt(held)} px; "
                f"{report['fit_status']}"
            )
            if report.get("out_of_bounds_point_ids"):
                lines.append(f"  out of bounds: {', '.join(report['out_of_bounds_point_ids'])}")
            if report.get("duplicate_point_pairs_within_1px"):
                pairs = report["duplicate_point_pairs_within_1px"]
                lines.append("  possible duplicate source/map points: " + ", ".join(
                    f"{item['first']}/{item['second']}" for item in pairs
                ))
        pending = sum(1 for point in self.points if point.get("role") == "heldout" and point.get("independent_of_prior_fit") is not True)
        missing = {key: sum(
            point.get(key) is None
            or point.get(f"{key}_source_sha256") != self.video_metadata[key]["sha256"]
            or point.get(f"{key}_coordinate_space") != self.coordinate_space
            for point in self.points
        ) for key in ("cam02", "cam05")}
        lines.append(
            f"Camera clicks missing/mismatched in this coordinate space: CAM-02={missing['cam02']}, "
            f"CAM-05={missing['cam05']}; legacy camera pixels remain retained for audit. "
            f"Held-out candidates lacking independence attestation: {pending}."
        )
        lines.append("Prior AMC points cannot be converted to independent holdouts. Training/LOO/held-out values are pixel diagnostics, not physical validation or meters.")
        self.diagnostics.setText("\n".join(lines))

    def save_version(self) -> None:
        self.output_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.output_dir.parent, 0o700)
        self.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.output_dir, 0o700)
        existing = list(self.output_dir.glob(f"{self.output_stem}-v[0-9][0-9][0-9][0-9]-*.json"))
        versions = []
        for path in existing:
            match = re.match(rf"{re.escape(self.output_stem)}-v(\d{{4}})-", path.name)
            if match:
                versions.append(int(match.group(1)))
        version = max(versions, default=0) + 1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = self.output_dir / f"{self.output_stem}-v{version:04d}-{stamp}.json"
        source_shapes = {key: (self.coordinate_dimensions[key][1], self.coordinate_dimensions[key][0])
                         for key in self.video_metadata}
        diagnostic = {
            key: diagnose_homography(self._diagnostic_points_for(key), key,
                                     source_shape_hw=source_shapes[key],
                                     layout_shape_hw=(self.map_metadata["height"], self.map_metadata["width"]))
            for key in ("cam02", "cam05")
        }
        fit_ready = all(
            diagnostic[key]["matrix_row_major"] is not None
            and not diagnostic[key]["out_of_bounds_point_ids"]
            and not diagnostic[key]["duplicate_point_pairs_within_1px"]
            for key in ("cam02", "cam05")
        )
        holdout_ready = all(diagnostic[key]["heldout_point_count"] >= 2 for key in ("cam02", "cam05"))
        candidate_ready = fit_ready and holdout_ready
        candidate_status = (
            "fit_and_two_independent_heldout_landmarks_per_camera_available_accuracy_still_unverified"
            if candidate_ready else
            "fit_candidate_available_independent_heldout_points_missing" if fit_ready else
            "incomplete_draft_no_usable_fit_for_both_cameras"
        )
        source_to_coordinate = {}
        for key, meta in self.video_metadata.items():
            coordinate_width, coordinate_height = self.coordinate_dimensions[key]
            source_to_coordinate[key] = {
                "source_width": meta["width"], "source_height": meta["height"],
                "coordinate_width": coordinate_width, "coordinate_height": coordinate_height,
                "scale_x": coordinate_width / meta["width"],
                "scale_y": coordinate_height / meta["height"],
                "operation": "OpenCV cv2.resize INTER_AREA full-frame; no crop/pad/undistort/warp"
                if self.coordinate_space == LIVE_NATIVE_SPACE else "identity decoded-raster coordinates",
            }
        payload = {
            "schema_version": 2,
            "project": "Entrance Room",
            "camera_mapping": {"cam02": "CAM-02", "cam05": "CAM-05"},
            "coordinate_space": self.coordinate_space,
            "camera_coordinate_dimensions": {key: list(value) for key, value in self.coordinate_dimensions.items()},
            "source_to_coordinate_transform": source_to_coordinate,
            "coordinate_convention": "decoded raster pixel x-right/y-down; camera pixels are tied to exact per-camera file SHA-256 and coordinate-space label; editor display scaling preserves aspect and is not part of saved coordinates",
            "source_files": {key: self.video_metadata[key] for key in ("cam02", "cam05")},
            "layout": self.map_metadata,
            "reference_alignment": {"path": str(self.reference_alignment_path.resolve()),
                                    "sha256": self.alignment_hash,
                                    "point_count": self.reference_point_count},
            "points": self.points,
            "diagnostics": diagnostic,
            "candidate_status": candidate_status,
            "limitations": [
                "No camera distortion/remap is applied by this editor.",
                "Rectified mode uses exact rectified MV3DT inputs; live-native mode uses the verified scale-only live resize and no AMC remap.",
                "No physical map scale or surveyed floor coordinates are inferred.",
                "Frame indices are per-file only; equal indices do not prove synchronized exposure.",
                "Saved old AMC points were consumed in a prior fit and are not independent validation.",
                "Held-out errors are independent only if those physical floor targets were separately measured and excluded from fitting.",
                "Prior AMC-used point IDs P01–P08 cannot be relabeled as independent held-out validation points.",
                "The operator independence attestation is annotation metadata, not proof of a survey or metric accuracy.",
            ],
        }
        # Exclusive creation: never replace an earlier editor version.
        try:
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            self.status.setText("Version collision; no file overwritten. Press save again to allocate a new version.")
            return
        self.status.setText(f"Saved new immutable staging candidate: {target}; original AMC files were not changed.")

    def closeEvent(self, event) -> None:  # noqa: N802
        for capture in self.captures.values():
            capture.release()
        super().closeEvent(event)


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    default_project = Path("/home/apsidal/auto-magic-calib/projects/project_20261019_101845_7932")
    default_dataset = root / "calibration_recordings/entrance/entrance-20261009T130526Z"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cam02", type=Path, default=default_dataset / "camera_1.mp4")
    parser.add_argument("--cam05", type=Path, default=default_dataset / "camera_2.mp4")
    parser.add_argument("--layout", type=Path, default=default_project / "manual_adjustment/layout.png")
    parser.add_argument("--reference-alignment", type=Path,
                        default=root / ".runtime/entrance-calibration-editor/versions/alignment-v0011-20261021T111654Z.json")
    parser.add_argument("--coordinate-space", choices=("project_input_20fps_1920x1080",
                                                        "rectified_mv3dt_30fps_1920x1080",
                                                        LIVE_NATIVE_SPACE),
                        default=LIVE_NATIVE_SPACE)
    parser.add_argument("--output-dir", type=Path,
                        default=root / ".runtime/entrance-calibration-editor/live-native-versions")
    args = parser.parse_args(argv)
    paths = (args.cam02, args.cam05, args.layout, args.reference_alignment)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error("missing local input(s): " + ", ".join(missing))
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = CalibrationEditor(cam02=args.cam02, cam05=args.cam05, layout=args.layout,
                               reference_alignment=args.reference_alignment,
                               output_dir=args.output_dir,
                               coordinate_space=args.coordinate_space)
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
