from __future__ import annotations

from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QApplication

from services.frontend.app.camera_wall import (
    CameraTile,
    bbox_coordinate_dimensions,
    snapshot_bbox_coordinate_dimensions,
    snapshot_bbox_rows,
)


def _app():
    return QApplication.instance() or QApplication([])


def _tile():
    _app()
    tile = CameraTile("CAM-01", "http://127.0.0.1:8101")
    tile.tracked_object_feed_supported = True
    tile.current_source_frame_num = 100
    tile.current_pts_ns = 123456789
    tile.tracked_objects = [{
        "camera_id": "CAM-01", "frame": 100, "pts_ns": 123456789, "native_track_id": 42,
        "bbox": [0, 0, 2560, 1440],
    }]
    return tile


def test_tracker_box_paints_unknown_without_identity_and_never_duplicates():
    tile = _tile()
    tile.identity_rows = []
    pixmap = QPixmap(256, 144)
    pixmap.fill(QColor("black"))
    tile._draw_overlays(pixmap)
    assert pixmap.toImage().pixelColor(100, 0) == QColor(255, 210, 40)
    # The tracked-object feed owns box rendering; identity is text/color
    # enrichment and is not drawn as a second rectangle.
    assert tile._matching_identity_row(tile.tracked_objects[0]) is None


def test_identity_enriches_only_same_camera_native_track_and_nearby_frame():
    tile = _tile()
    tile.identity_rows = [{
        "camera_id": "CAM-01", "native_track_id": 42,
        "frame": 100, "application_id": "Person_01",
    }]
    pixmap = QPixmap(256, 144)
    pixmap.fill(QColor("black"))
    tile._draw_overlays(pixmap)
    assert pixmap.toImage().pixelColor(100, 0) == QColor(80, 220, 80)


def test_stale_or_wrong_camera_identity_cannot_attach_to_tracker_box():
    tile = _tile()
    tile.identity_rows = [
        {"camera_id": "CAM-04", "native_track_id": 42, "frame": 99, "application_id": "Person_01"},
        {"camera_id": "CAM-01", "native_track_id": 42, "frame": 103, "application_id": "Person_02"},
    ]
    assert tile._matching_identity_row(tile.tracked_objects[0]) is None


def test_tracker_box_requires_exact_current_pts():
    tile = _tile()
    tile.current_pts_ns = 123456790
    pixmap = QPixmap(256, 144)
    pixmap.fill(QColor("black"))
    tile._draw_overlays(pixmap)
    assert pixmap.toImage().pixelColor(100, 0) == QColor("black")


def test_identity_without_canonical_object_does_not_create_a_box():
    tile = CameraTile("CAM-01", "http://127.0.0.1:8101")
    tile.current_source_frame_num = 100
    tile.current_pts_ns = 123456789
    tile.set_identity_rows([{
        "camera_id": "CAM-01", "frame": 100, "native_track_id": 42,
        "bbox": [0, 0, 2560, 1440], "application_id": "Person_01",
    }])
    tile.tracked_objects = []
    pixmap = QPixmap(256, 144)
    pixmap.fill(QColor("black"))
    tile._draw_overlays(pixmap)
    assert pixmap.toImage().pixelColor(100, 0) == QColor("black")


def test_diagnostic_overlay_labels_same_frame_detector_and_active_track():
    tile = _tile()
    tile.box_diagnostics = True
    tile.tracked_objects = [{
        "camera_id": "CAM-01", "frame_num": 100, "pts_ns": 123456789,
        "native_track_id": 42, "bbox": [50, 60, 180, 230], "bbox_source": "TRACKER",
    }]
    tile.detector_proposals = [{
        "camera_id": "CAM-01", "frame_num": 100, "pts_ns": 123456789,
        "proposal_index": 0, "confidence": 0.81, "bbox": [10, 20, 200, 200],
        "source_frame_width": 2560, "source_frame_height": 1440,
    }]
    pixmap = QPixmap(256, 144)
    pixmap.fill(QColor("black"))
    tile._draw_overlays(pixmap)
    image = pixmap.toImage()
    assert image.pixelColor(2, 2) == QColor(255, 170, 35)
    assert image.pixelColor(6, 6) == QColor(70, 240, 100)


def test_pipeline_bbox_uses_mux_extent_not_native_source_extent():
    snapshot = {
        "bbox_coordinate_space": "pipeline",
        "bbox_coordinate_width": 2560,
        "bbox_coordinate_height": 1440,
        "pipeline_width": 2560,
        "pipeline_height": 1440,
        "source_frame_width": 3200,
        "source_frame_height": 1800,
    }
    assert snapshot_bbox_coordinate_dimensions(snapshot, (3200, 1800)) == (2560, 1440)


def test_pipeline_bbox_contract_fails_closed_when_dimensions_are_missing_or_conflict():
    source_only = {
        "bbox_coordinate_space": "pipeline",
        "source_frame_width": 3200,
        "source_frame_height": 1800,
    }
    mismatched = {
        "bbox_coordinate_space": "pipeline",
        "bbox_coordinate_width": 3200,
        "bbox_coordinate_height": 1800,
        "pipeline_width": 2560,
        "pipeline_height": 1440,
        "source_frame_width": 3200,
        "source_frame_height": 1800,
    }
    assert bbox_coordinate_dimensions(source_only, (3200, 1800)) is None
    assert snapshot_bbox_coordinate_dimensions(source_only, (3200, 1800)) is None
    assert snapshot_bbox_coordinate_dimensions(mismatched, (3200, 1800)) is None


def test_rows_with_bbox_space_different_from_frame_contract_are_withheld():
    snapshot = {
        "bbox_coordinate_space": "pipeline",
        "bbox_coordinate_width": 2560,
        "bbox_coordinate_height": 1440,
        "pipeline_width": 2560,
        "pipeline_height": 1440,
        "objects": [
            {"native_track_id": 1, "bbox_coordinate_space": "pipeline",
             "bbox_coordinate_width": 2560, "bbox_coordinate_height": 1440},
            {"native_track_id": 2, "bbox_coordinate_space": "pipeline",
             "bbox_coordinate_width": 3200, "bbox_coordinate_height": 1800},
            {"native_track_id": 3, "bbox_coordinate_space": "pipeline",
             "source_frame_width": 3200, "source_frame_height": 1800},
        ],
    }
    dimensions = snapshot_bbox_coordinate_dimensions(snapshot)
    assert dimensions == (2560, 1440)
    rows = snapshot_bbox_rows(snapshot, "objects", dimensions)
    assert [row["native_track_id"] for row in rows] == [1]


def test_cam02_mux_space_box_maps_to_matching_960x540_preview_pixels():
    tile = _tile()
    tile.tracked_objects = [{
        "camera_id": "CAM-01", "frame_num": 100, "pts_ns": 123456789,
        "native_track_id": 42, "bbox": [1262, 1062, 1600, 1386],
        "bbox_source": "TRACKER", "bbox_coordinate_space": "pipeline",
        "bbox_coordinate_width": 2560, "bbox_coordinate_height": 1440,
        "source_frame_width": 3200, "source_frame_height": 1800,
    }]
    pixmap = QPixmap(960, 540)
    pixmap.fill(QColor("black"))
    tile._draw_overlays(pixmap)
    image = pixmap.toImage()
    assert image.pixelColor(473, 398) == QColor(255, 210, 40)


def test_pipeline_box_missing_mux_extent_is_not_painted_at_native_scale():
    tile = _tile()
    tile.tracked_objects = [{
        "camera_id": "CAM-01", "frame_num": 100, "pts_ns": 123456789,
        "native_track_id": 42, "bbox": [1262, 1062, 1600, 1386],
        "bbox_source": "TRACKER", "bbox_coordinate_space": "pipeline",
        "source_frame_width": 3200, "source_frame_height": 1800,
    }]
    pixmap = QPixmap(960, 540)
    pixmap.fill(QColor("black"))
    tile._draw_overlays(pixmap)
    image = pixmap.toImage()
    # The wrong source-space scale would put the top-left edge near (378,318).
    assert image.pixelColor(378, 318) == QColor("black")
