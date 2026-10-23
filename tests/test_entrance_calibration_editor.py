from __future__ import annotations

import json
import os

import numpy as np
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from services.mv3dt_staging import calibration_editor
from services.mv3dt_staging.calibration_editor import (
    LIVE_NATIVE_SPACE,
    _eligible_new_holdout,
    _frame_in_coordinate_space,
    load_editor_points,
)
from services.mv3dt_staging.calibration_geometry import diagnose_homography, load_amc_alignment
from services.mv3dt_staging.qt_ui import MapCanvas


def _synthetic_points() -> list[dict]:
    homography = np.asarray([[1.12, 0.08, 31.0], [-0.04, 0.97, 14.0],
                             [0.00016, -0.00011, 1.0]], dtype=float)
    source = [(20, 20), (420, 30), (440, 280), (40, 300),
              (180, 70), (310, 210), (95, 185), (360, 115)]
    points = []
    for index, (x, y) in enumerate(source, start=1):
        result = homography @ np.asarray([x, y, 1.0])
        target = (result[:2] / result[2]).tolist()
        points.append({"point_id": f"P{index:02d}", "cam02": [float(x), float(y)],
                       "cam05": [float(x), float(y)], "layout": target,
                       "role": "heldout" if index > 6 else "fit"})
    return points


def test_saved_amc_triplets_load_as_legacy_fit_references(tmp_path) -> None:
    original = [[[10, 20], [30, 40], [50, 60]], [[70, 80], [90, 100], [110, 120]]]
    path = tmp_path / "alignment.json"
    path.write_text(json.dumps(original))
    points = load_amc_alignment(path)
    assert len(points) == 2
    assert points[0]["cam02"] == [10.0, 20.0]
    assert points[0]["cam05"] == [30.0, 40.0]
    assert points[0]["layout"] == [50.0, 60.0]
    assert points[0]["role"] == "fit"
    assert "prior_fit" in points[0]["provenance"]
    assert path.read_text() == json.dumps(original)


def test_rectified_editor_hides_legacy_camera_pixels_but_preserves_them(tmp_path) -> None:
    path = tmp_path / "alignment-v0010.json"
    payload = {
        "coordinate_space": "project_input_20fps_1920x1080",
        "source_files": {
            "cam02": {"path": "/private/cam02-source.mp4", "sha256": "source02"},
            "cam05": {"path": "/private/cam05-source.mp4", "sha256": "source05"},
        },
        "layout": {"path": "/private/layout.png", "sha256": "layout-hash"},
        "points": [{"point_id": "P03", "cam02": [100.0, 200.0], "cam05": [300.0, 400.0],
                    "layout": [50.0, 60.0], "role": "fit",
                    "provenance": "saved_amc_alignment_used_by_prior_fit",
                    "cam02_frame_index": 11, "cam05_frame_index": 12}],
    }
    path.write_text(json.dumps(payload))
    points = load_editor_points(
        path,
        video_metadata={
            "cam02": {"sha256": "rectified02"},
            "cam05": {"sha256": "rectified05"},
        },
        map_metadata={"sha256": "layout-hash"},
        coordinate_space="rectified_mv3dt_30fps_1920x1080",
    )
    point = points[0]
    assert point["cam02"] is None and point["cam05"] is None
    assert point["legacy_camera_coordinates"]["cam02"] == {
        "xy": [100.0, 200.0], "source_sha256": "source02",
        "source_path": "/private/cam02-source.mp4",
        "coordinate_space": "project_input_20fps_1920x1080", "frame_index": 11,
    }
    assert point["legacy_camera_coordinates"]["cam05"]["xy"] == [300.0, 400.0]
    assert point["layout"] == [50.0, 60.0]
    assert point["layout_verified"] is False
    assert point["role"] == "fit"
    assert point["provenance"] == "saved_amc_alignment_used_by_prior_fit"
    assert point["prior_alignment_reference"] is True


def test_previous_alignment_points_cannot_be_promoted_to_new_independent_holdouts(tmp_path) -> None:
    path = tmp_path / "alignment-v0011.json"
    path.write_text(json.dumps({"coordinate_space": "rectified",
                                "points": [{"point_id": "P01", "cam02": [1, 2],
                                            "cam05": [3, 4], "layout": [5, 6],
                                            "role": "fit"}]}))
    point = load_editor_points(path, video_metadata={"cam02": {"sha256": "raw02"},
                                                      "cam05": {"sha256": "raw05"}},
                               map_metadata={"sha256": "map"},
                               coordinate_space=LIVE_NATIVE_SPACE)[0]
    assert point["prior_alignment_reference"] is True
    assert not _eligible_new_holdout(point)
    assert _eligible_new_holdout({"provenance": "operator_selected_new_floor_landmark",
                                  "prior_alignment_reference": False})


def test_live_native_editor_resize_matches_live_full_fov_scale() -> None:
    import cv2

    from services.mv3dt_staging.live_tracking.native_frame_bridge import SOURCE_SIZE

    for key, camera_id in (("cam02", "CAM-02"), ("cam05", "CAM-05")):
        width, height = SOURCE_SIZE[camera_id]
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :width // 3, 0] = 255
        frame[height // 3:2 * height // 3, :, 1] = 127
        frame[-1, -1, 2] = 255
        actual = _frame_in_coordinate_space(frame, key, LIVE_NATIVE_SPACE)
        expected = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        assert actual.shape == (1080, 1920, 3)
        assert np.array_equal(actual, expected)


def test_live_native_resize_rejects_wrong_native_dimensions() -> None:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    try:
        _frame_in_coordinate_space(frame, "cam02", LIVE_NATIVE_SPACE)
    except ValueError as error:
        assert "expected native uint8 BGR 3200x1800" in str(error)
    else:
        raise AssertionError("wrong coordinate source dimensions were accepted")


def test_matching_rectified_hash_and_space_keeps_current_camera_point(tmp_path) -> None:
    path = tmp_path / "alignment.json"
    payload = {
        "coordinate_space": "rectified_mv3dt_30fps_1920x1080",
        "source_files": {"cam02": {"sha256": "rectified02"},
                         "cam05": {"sha256": "rectified05"}},
        "layout": {"sha256": "layout-hash"},
        "points": [{"point_id": "P09", "cam02": [101.0, 202.0], "cam05": [303.0, 404.0],
                    "layout": [50.0, 60.0], "role": "fit",
                    "provenance": "operator_selected_new_floor_landmark",
                    "cam02_coordinate_space": "rectified_mv3dt_30fps_1920x1080",
                    "cam05_coordinate_space": "rectified_mv3dt_30fps_1920x1080",
                    "layout_source_sha256": "layout-hash", "layout_verified": True}],
    }
    path.write_text(json.dumps(payload))
    points = load_editor_points(
        path,
        video_metadata={"cam02": {"sha256": "rectified02"},
                        "cam05": {"sha256": "rectified05"}},
        map_metadata={"sha256": "layout-hash"},
        coordinate_space="rectified_mv3dt_30fps_1920x1080",
    )
    assert points[0]["cam02"] == [101.0, 202.0]
    assert points[0]["cam05"] == [303.0, 404.0]
    assert points[0]["layout_verified"] is True
    assert "legacy_camera_coordinates" not in points[0]


def test_cli_forwards_selected_coordinate_space_to_editor(tmp_path, monkeypatch) -> None:
    files = [tmp_path / name for name in ("cam02.mp4", "cam05.mp4", "layout.png", "alignment.json")]
    for path in files:
        path.write_bytes(b"test")
    captured = {}

    class FakeApp:
        @staticmethod
        def instance():
            return FakeApp()

        def exec(self):
            return 0

    class FakeEditor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def showMaximized(self):
            pass

    monkeypatch.setattr(calibration_editor, "QApplication", FakeApp)
    monkeypatch.setattr(calibration_editor, "CalibrationEditor", FakeEditor)
    result = calibration_editor.main([
        "--cam02", str(files[0]), "--cam05", str(files[1]),
        "--layout", str(files[2]), "--reference-alignment", str(files[3]),
        "--coordinate-space", "rectified_mv3dt_30fps_1920x1080",
        "--output-dir", str(tmp_path / "versions"),
    ])
    assert result == 0
    assert captured["coordinate_space"] == "rectified_mv3dt_30fps_1920x1080"


def test_heldout_points_do_not_enter_fit_and_loo_is_separate() -> None:
    points = _synthetic_points()
    baseline = diagnose_homography(points, "cam02", source_shape_hw=(400, 500),
                                   layout_shape_hw=(500, 700))
    assert baseline["fit_point_count"] == 6
    assert baseline["heldout_point_count"] == 2
    assert baseline["training_rms_layout_px"] is not None
    assert baseline["leave_one_out_rms_layout_px"] is not None
    assert baseline["heldout_rms_layout_px"] is not None
    assert baseline["heldout_rms_layout_px"] < 1e-4
    points[6]["layout"] = [670.0, 20.0]
    changed_holdout = diagnose_homography(points, "cam02", source_shape_hw=(400, 500),
                                          layout_shape_hw=(500, 700))
    assert changed_holdout["matrix_row_major"] == baseline["matrix_row_major"]
    assert changed_holdout["heldout_rms_layout_px"] > 10.0


def test_degenerate_correspondences_fail_closed() -> None:
    points = [{"point_id": f"P{index:02d}", "cam02": [x, 0.0], "cam05": [x, 0.0],
               "layout": [x, 0.0], "role": "fit"}
              for index, x in enumerate((0.0, 100.0, 200.0, 300.0), start=1)]
    result = diagnose_homography(points, "cam02", source_shape_hw=(100, 100),
                                 layout_shape_hw=(100, 100))
    assert result["matrix_row_major"] is None
    assert "BLOCKED" in result["fit_status"]
    assert result["leave_one_out_rms_layout_px"] is None


def test_exactly_four_fit_points_have_no_leave_one_out_model() -> None:
    points = _synthetic_points()[:4]
    result = diagnose_homography(points, "cam02", source_shape_hw=(400, 500),
                                 layout_shape_hw=(500, 700))
    assert result["matrix_row_major"] is not None
    assert result["fit_point_count"] == 4
    assert result["leave_one_out_errors"] == []
    assert result["leave_one_out_rms_layout_px"] is None


def test_short_trails_are_camera_scoped_bounded_and_reset_on_seek(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    map_path = tmp_path / "map.png"
    image = QImage(120, 90, QImage.Format.Format_RGB32)
    image.fill(QColor("#ddd"))
    assert image.save(str(map_path))
    canvas = MapCanvas(map_path)
    canvas.set_transform([1, 0, 0, 0, -1, 0, 0, 0, 1])
    rows = [
        {"camera_id": camera, "mv3dt_global_id": "7", "raw_global_id": "7",
         "frame_id": frame, "world_position_unverified_units": [float(frame), -5.0]}
        for camera in ("CAM-02", "CAM-05") for frame in (10, 11)
    ]
    canvas.set_observations([row for row in rows if row["frame_id"] == 10], current_frame=10,
                            separate_by_camera=True)
    canvas.set_observations([row for row in rows if row["frame_id"] == 11], current_frame=11,
                            separate_by_camera=True)
    assert set(canvas.trails) == {("CAM-02", "7"), ("CAM-05", "7")}
    assert len(canvas.trails[("CAM-02", "7")]) == 2
    canvas.set_observations(rows[-2:], current_frame=11, separate_by_camera=True)
    assert len(canvas.trails[("CAM-02", "7")]) == 2  # same frame query is replacement, not duplicate

    later = [{**row, "frame_id": 250, "world_position_unverified_units": [250.0, -5.0]}
             for row in rows[:2]]
    canvas.set_observations(later, current_frame=250, separate_by_camera=True)
    assert [frame for frame, _xy in canvas.trails[("CAM-02", "7")]] == [250]
    earlier = [{**row, "frame_id": 20, "world_position_unverified_units": [20.0, -5.0]}
               for row in rows[:2]]
    canvas.set_observations(earlier, current_frame=20, separate_by_camera=True)
    assert [frame for frame, _xy in canvas.trails[("CAM-02", "7")]] == [20]
    canvas.close()
    app.processEvents()
