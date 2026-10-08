from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication

from services.frontend.app.camera_wall import CameraTile
from services.frontend.app.room_pair_panel import RoomPairPanel


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def candidate(camera="CAM-01", native=10, person="Person_01"):
    return {
        "camera_id": camera,
        "native_track_id": native,
        "application_id": person,
        "global_person_id": person,
        "frame": 50,
        "timestamp": "2026-09-24T12:00:00Z",
        "bbox": [100.0, 100.0, 300.0, 500.0],
        "world": [1.0, 2.0],
        "confidence": 0.9,
    }


def test_ui_acceptance_controls_are_opt_in_and_replay_is_rejected(app, tmp_path, monkeypatch):
    monkeypatch.setenv("AI_SURVEILLANCE_ROOT", str(tmp_path))
    panel = RoomPairPanel()
    panel.update_snapshot({"source_mode": "live", "status": "running", "session_id": "live-test-1", "people": []})
    assert not panel.acceptance.enabled
    assert panel.acceptance_controls.isHidden()

    panel.acceptance_toggle.setChecked(True)
    assert panel.acceptance.enabled
    assert not panel.acceptance_controls.isHidden()
    panel.subtest_choice.setCurrentIndex(1)
    panel.start_subtest()
    assert panel.acceptance.subtest == "B"
    assert not panel.begin_exit_button.isEnabled()
    assert "select a visible bbox" in panel.acceptance_status.text().lower()
    assert panel.acceptance.expected_runtime_session == "live-test-1"

    panel.update_acceptance_candidates({
        "acceptance_only": True,
        "source_mode": "replay",
        "coordinate_dimensions_ready": True,
        "session_id": "replay-test",
        "candidates": [candidate()],
    })
    assert panel._acceptance_candidates == []
    assert "requires source mode LIVE" in panel.acceptance_status.text()
    panel.deleteLater()


def test_subtest_b_exit_button_waits_for_stable_fresh_observations(app, tmp_path, monkeypatch):
    monkeypatch.setenv("AI_SURVEILLANCE_ROOT", str(tmp_path))
    panel = RoomPairPanel()
    panel.acceptance.pre_exit_stability_seconds = 0
    panel.acceptance.pre_exit_min_observations = 2
    panel.update_snapshot({"source_mode": "live", "status": "running", "session_id": "live-b-ready", "people": []})
    panel.acceptance_toggle.setChecked(True)
    panel.subtest_choice.setCurrentIndex(1)
    panel.start_subtest()
    assert not panel.begin_exit_button.isHidden()
    assert not panel.begin_exit_button.isEnabled()

    first = candidate("CAM-01", 10, "Person_01")
    panel.handle_acceptance_bbox_selected("CAM-01", first, QImage(20, 30, QImage.Format.Format_RGB32))
    assert not panel.begin_exit_button.isEnabled()
    second = dict(first, frame=51, timestamp="2026-09-24T12:00:00.050Z")
    panel.update_acceptance_candidates({
        "acceptance_only": True,
        "source_mode": "live",
        "coordinate_dimensions_ready": True,
        "session_id": "live-b-ready",
        "candidates": [second],
    })
    assert panel.begin_exit_button.isEnabled()
    panel.deleteLater()


def test_ui_operator_assignment_stores_review_crop_only_in_acceptance_artifacts(app, tmp_path, monkeypatch):
    monkeypatch.setenv("AI_SURVEILLANCE_ROOT", str(tmp_path))
    panel = RoomPairPanel()
    panel.update_snapshot({"source_mode": "live", "status": "running", "session_id": "live-test-2", "people": []})
    panel.acceptance_toggle.setChecked(True)
    panel.start_subtest()
    row = candidate()
    panel.update_acceptance_candidates({
        "acceptance_only": True,
        "source_mode": "live",
        "coordinate_dimensions_ready": True,
        "session_id": "live-test-2",
        "candidates": [row],
    })
    crop = QImage(20, 30, QImage.Format.Format_RGB32)
    panel.handle_acceptance_bbox_selected("CAM-01", row, crop)

    record = panel.acceptance.testers["Tester_A"]
    assert record["canonical_person_id"] == "Person_01"
    assert record["initial_camera"] == "CAM-01"
    assert record["initial_native_track_id"] == 10
    assert record["reference_crop"] is not None
    assert str(tmp_path) in record["reference_crop"]
    assert "Tester_A" not in row
    assert list((tmp_path / ".runtime/mv3dt/dev-room-cam01-cam04/acceptance/reference-crops").glob("*.png"))
    panel.deleteLater()


def test_changed_person_id_requires_acceptance_only_visual_confirmation(app, tmp_path, monkeypatch):
    monkeypatch.setenv("AI_SURVEILLANCE_ROOT", str(tmp_path))
    panel = RoomPairPanel()
    panel.update_snapshot({"source_mode": "live", "status": "running", "session_id": "live-id-review", "people": []})
    panel.acceptance_toggle.setChecked(True)
    panel.subtest_choice.setCurrentIndex(1)
    panel.start_subtest()
    first = candidate("CAM-01", 10, "Person_01")
    panel.update_acceptance_candidates({
        "acceptance_only": True,
        "source_mode": "live",
        "coordinate_dimensions_ready": True,
        "session_id": "live-id-review",
        "candidates": [first],
    })
    crop = QImage(20, 30, QImage.Format.Format_RGB32)
    panel.handle_acceptance_bbox_selected("CAM-01", first, crop)

    displayed = []
    monkeypatch.setattr(
        panel,
        "_confirm_same_physical_tester",
        lambda *args: displayed.append((args[1], args[2], not args[-1].isNull())) or False,
    )
    changed = candidate("CAM-04", 22, "Person_02")
    changed.update({"frame": 51, "timestamp": "2026-09-25T12:00:01Z"})
    panel.handle_acceptance_bbox_selected("CAM-04", changed, crop)

    assert displayed == [("Person_01", "Person_02", True)]
    assert panel.acceptance.testers["Tester_A"]["canonical_person_id"] == "Person_01"
    assert panel.acceptance.failure is None
    candidate_crops = list((tmp_path / ".runtime/mv3dt/dev-room-cam01-cam04/acceptance/candidate-crops").glob("*.png"))
    assert candidate_crops
    assert "Recorded as a different person" in panel.acceptance_status.text()
    panel.deleteLater()


def test_confirmed_same_person_with_new_canonical_id_is_an_acceptance_failure(app, tmp_path, monkeypatch):
    monkeypatch.setenv("AI_SURVEILLANCE_ROOT", str(tmp_path))
    panel = RoomPairPanel()
    panel.update_snapshot({"source_mode": "live", "status": "running", "session_id": "live-id-review-yes", "people": []})
    panel.acceptance_toggle.setChecked(True)
    panel.subtest_choice.setCurrentIndex(1)
    panel.start_subtest()
    panel.handle_acceptance_bbox_selected(
        "CAM-01", candidate("CAM-01", 10, "Person_01"), QImage(20, 30, QImage.Format.Format_RGB32)
    )
    monkeypatch.setattr(panel, "_confirm_same_physical_tester", lambda *args: True)
    changed = candidate("CAM-04", 22, "Person_02")
    changed.update({"frame": 52, "timestamp": "2026-09-25T12:00:02Z"})
    panel.handle_acceptance_bbox_selected("CAM-04", changed, QImage(20, 30, QImage.Format.Format_RGB32))

    assert panel.acceptance.failure["kind"] == "identity_change"
    assert panel.acceptance.failure["expected_person_id"] == "Person_01"
    assert panel.acceptance.failure["assigned_person_id"] == "Person_02"
    assert panel.acceptance.testers["Tester_A"]["canonical_person_id"] == "Person_01"
    panel.deleteLater()


def test_subtest_a_ui_waits_for_both_operator_labels_and_false_merge_confirmation(app, tmp_path, monkeypatch):
    monkeypatch.setenv("AI_SURVEILLANCE_ROOT", str(tmp_path))
    panel = RoomPairPanel()
    panel.update_snapshot({"source_mode": "live", "status": "running", "session_id": "live-a-gate", "people": []})
    panel.acceptance_toggle.setChecked(True)
    assert "WAITING FOR" in panel.acceptance_status.text()
    assert "[ ] Tester_A selected" in panel.acceptance_checklist.text()
    assert "[ ] Tester_B selected" in panel.acceptance_checklist.text()

    panel.start_subtest()
    assert panel.acceptance.started_monotonic is None
    first = candidate("CAM-01", 10, "Person_01")
    second = candidate("CAM-04", 20, "Person_01")
    panel.tester_choice.setCurrentText("Tester_A")
    panel.handle_acceptance_bbox_selected("CAM-01", first, QImage(20, 30, QImage.Format.Format_RGB32))
    panel.tester_choice.setCurrentText("Tester_B")
    panel.handle_acceptance_bbox_selected("CAM-04", second, QImage(20, 30, QImage.Format.Format_RGB32))

    assert panel.acceptance.started_monotonic is None
    assert panel.acceptance.failure is None  # same ID waits for explicit human ground-truth confirmation
    assert not panel.distinct_people_confirm.isHidden()
    panel.distinct_people_confirm.setChecked(True)
    panel.start_subtest()

    assert panel.acceptance.failure["kind"] == "FAIL_FALSE_MERGE_AT_START"
    assert panel.acceptance.started_monotonic is None
    assert "FAIL_FALSE_MERGE_AT_START" in panel.acceptance_status.text()
    panel.deleteLater()


def test_subtest_a_ui_starts_timer_only_after_confirmation_with_distinct_ids(app, tmp_path, monkeypatch):
    monkeypatch.setenv("AI_SURVEILLANCE_ROOT", str(tmp_path))
    panel = RoomPairPanel()
    panel.update_snapshot({"source_mode": "live", "status": "running", "session_id": "live-a-ready", "people": []})
    panel.acceptance_toggle.setChecked(True)
    panel.start_subtest()
    panel.tester_choice.setCurrentText("Tester_A")
    panel.handle_acceptance_bbox_selected(
        "CAM-01", candidate("CAM-01", 10, "Person_01"), QImage(20, 30, QImage.Format.Format_RGB32)
    )
    panel.tester_choice.setCurrentText("Tester_B")
    panel.handle_acceptance_bbox_selected(
        "CAM-01", candidate("CAM-01", 11, "Person_02"), QImage(20, 30, QImage.Format.Format_RGB32)
    )
    assert panel.acceptance.started_monotonic is None
    panel.distinct_people_confirm.setChecked(True)
    panel.start_subtest()
    assert panel.acceptance.started_monotonic is not None
    assert panel.acceptance.failure is None
    panel.deleteLater()


def test_camera_bbox_hit_test_uses_verified_source_dimensions(app, monkeypatch):
    from services.frontend.app import camera_wall

    monkeypatch.setenv("FRONTEND_USE_V11_SHARED_MEMORY", "1")
    monkeypatch.setattr(camera_wall, "overlay_source_dimensions",
                        lambda camera, mode="live", snapshot=None: (1920, 1080))
    tile = CameraTile("CAM-04", "http://127.0.0.1:8101")
    tile.set_acceptance_mode(True)
    tile.video.resize(320, 180)
    tile.current_image = QImage(640, 360, QImage.Format.Format_RGB32)
    tile.displayed_pixmap_size = QSize(320, 180)
    row = candidate("CAM-04", 22, "Person_02")
    row.update({"bbox": [400.0, 400.0, 700.0, 700.0], "source_frame_width": 3200, "source_frame_height": 1800})
    tile.set_acceptance_candidates([row])
    selected = []
    tile.acceptance_bbox_selected.connect(lambda camera, picked, crop: selected.append((camera, picked, crop.size())))

    # At display pixel (50, 50), verified source coordinates are (500, 500).
    # The environment fallback would map this to (300, 300), outside the box.
    tile._on_video_clicked(50, 50)
    assert len(selected) == 1
    assert selected[0][0] == "CAM-04"
    assert selected[0][1]["native_track_id"] == 22
    assert selected[0][2] == QSize(60, 60)
    tile.close_reader()
    tile.deleteLater()


def test_fullscreen_uses_full_shared_preview_without_stretching(app, monkeypatch, tmp_path):
    from services.frontend.app import camera_wall
    from types import SimpleNamespace
    from PySide6.QtGui import QColor

    monkeypatch.setenv("FRONTEND_USE_V11_SHARED_MEMORY", "1")
    monkeypatch.setattr(camera_wall, "overlay_source_dimensions",
                        lambda camera, mode="live", snapshot=None: (3200, 1800))
    monkeypatch.setattr(camera_wall, "FULLSCREEN_PREVIEW_PATH", str(tmp_path / "fullscreen-preview.bin"))
    monkeypatch.setattr(camera_wall, "FULLSCREEN_REQUEST_PATH", tmp_path / "fullscreen-request")

    class FakeFullscreenReader:
        def __init__(self, _path):
            self.closed = False

        def read_latest(self, **_kwargs):
            pts = 123_000_000
            snapshot = {"camera_id": "CAM-04", "pts_ns": pts,
                "source_frame_num": 7, "frame_num": 7,
                "source_frame_width": 3200, "source_frame_height": 1800,
                "pipeline_width": 2560, "pipeline_height": 1440,
                "bbox_coordinate_space": "pipeline",
                "bbox_coordinate_width": 2560, "bbox_coordinate_height": 1440,
                "objects": [{"camera_id": "CAM-04", "frame_num": 7, "pts_ns": pts,
                    "native_track_id": 77, "bbox": [1280, 720, 2560, 1440],
                    "bbox_source": "TRACKER"}]}
            return SimpleNamespace(sequence=1, width=1920, height=1080, stride=1920 * 4,
                payload=bytes(1920 * 1080 * 4), tracker_snapshot=snapshot,
                source_frame_num=7, tracker_frame_num=7, pts_ns=pts,
                timestamp_ns=time.monotonic_ns(), decoder_reference_ns=0, decoder_out_ns=0,
                preview_prepare_start_ns=0, preview_prepare_complete_ns=0, tracker_ready_ns=0)

        def close(self):
            self.closed = True

    monkeypatch.setattr(camera_wall, "PreviewFrameReader", FakeFullscreenReader)
    tile = CameraTile("CAM-04", "http://127.0.0.1:8101")
    tile.video.resize(800, 600)
    tile.show()
    app.processEvents()
    tile._draw_frame(QImage(960, 540, QImage.Format.Format_RGB32))

    # The wall keeps the tile frame at its own resolution and avoids an
    # unnecessary full-resolution tile QPixmap. Fullscreen reads its separate
    # source buffer, not an upscale of this tile image.
    assert tile.current_image.size() == QSize(960, 540)
    assert tile.current_frame_pixmap is None
    assert tile.displayed_pixmap_size.width() / tile.displayed_pixmap_size.height() == pytest.approx(16 / 9)

    tile._open_fullscreen()
    app.processEvents()
    # The offscreen Qt platform has no physical display mode; give its
    # fullscreen label a representative 16:9 presentation surface.
    tile.fullscreen_video.resize(1280, 720)
    tile._refresh_fullscreen_preview()
    app.processEvents()
    displayed = tile.fullscreen_video.pixmap()
    assert displayed is not None
    assert displayed.width() / displayed.height() == pytest.approx(16 / 9, abs=0.02)
    assert tile.fullscreen_current_image.size() == QSize(1920, 1080)
    assert tile.current_image.size() == QSize(960, 540)
    assert tile.fullscreen_tracked_objects[0]["bbox_coordinate_width"] == 2560
    assert tile.fullscreen_tracked_objects[0]["bbox_coordinate_height"] == 1440
    # A mux-space lower-right-half bbox maps to the corresponding lower-right
    # half of the true 1920x1080 fullscreen image, not native 3200x1800 space.
    assert displayed.toImage().pixelColor(640, 360) == QColor(255, 210, 40)
    tile.close_reader()
    tile.deleteLater()


def test_shared_preview_stale_frame_is_dropped_not_kept_visible(app, monkeypatch):
    from services.frontend.app import camera_wall

    monkeypatch.setenv("FRONTEND_USE_V11_SHARED_MEMORY", "1")
    class EmptyPreviewReader:
        def __init__(self, _path):
            pass

        def read_latest(self, **_kwargs):
            return None

        def close(self):
            pass

    monkeypatch.setattr(camera_wall, "PreviewFrameReader", EmptyPreviewReader)
    tile = CameraTile("CAM-01", "http://127.0.0.1:8101")
    tile.video.setPixmap(QPixmap(320, 180))
    tile.last_version = 1
    tile.last_frame_mono = time.monotonic() - 0.30
    tile.refresh()

    assert tile.video.pixmap().isNull()
    assert tile.current_image is None
    assert tile.status.text().startswith("STALE")
    tile.close_reader()
    tile.deleteLater()


@pytest.mark.parametrize(
    ("camera", "source_width", "source_height"),
    [("CAM-02", 3200, 1800), ("CAM-03", 3200, 1800),
     ("CAM-05", 2560, 1440), ("CAM-06", 2560, 1440)],
)
def test_tracker_overlay_uses_dimensions_joined_from_current_preview_snapshot(
        app, monkeypatch, camera, source_width, source_height):
    from types import SimpleNamespace

    from services.frontend.app import camera_wall

    monkeypatch.setenv("FRONTEND_USE_V11_SHARED_MEMORY", "1")
    pts = 987_654_321
    object_row = {
        "camera_id": camera, "frame_num": 7, "pts_ns": pts,
        "native_track_id": 31, "bbox": [1280, 720, 2560, 1440],
        "bbox_coordinate_space": "pipeline", "bbox_coordinate_width": 2560,
        "bbox_coordinate_height": 1440,
        "bbox_source": "TRACKER", "detector_confidence": 0.9,
        "tracker_confidence": 0.8,
    }
    snapshot = {
        "schema_version": 1, "camera_id": camera, "generation": "test-session",
        "frame_num": 7, "source_frame_num": 7, "pts_ns": pts,
        "timestamp_monotonic_ns": time.monotonic_ns(), "timestamp_epoch_ns": 0,
        "source_frame_width": source_width, "source_frame_height": source_height,
        "pipeline_width": 2560, "pipeline_height": 1440,
        "bbox_coordinate_width": 2560, "bbox_coordinate_height": 1440,
        "objects": [object_row], "detector_proposals": [],
    }
    frame = SimpleNamespace(
        sequence=1, width=960, height=540, stride=960 * 4,
        payload=bytes(960 * 540 * 4), tracker_snapshot=snapshot,
        pts_ns=pts, source_frame_num=7, tracker_frame_num=7,
        timestamp_ns=time.monotonic_ns(), decoder_reference_ns=1,
        decoder_out_ns=2, preview_prepare_start_ns=3,
        preview_prepare_complete_ns=4, tracker_ready_ns=5, fps=20.0,
    )

    class OneFrameReader:
        def read_latest(self, **_kwargs):
            return frame

        def close(self):
            pass

    tile = CameraTile(camera, "http://127.0.0.1:8101")
    tile.preview_reader.close()
    tile.preview_reader = OneFrameReader()
    monkeypatch.setattr(camera_wall, "_write_timing_row", lambda *_args: None)
    tile._consume_latest_preview()

    assert tile.current_source_dimensions == (source_width, source_height)
    assert tile.tracked_objects[0]["source_frame_width"] == source_width
    assert tile.tracked_objects[0]["source_frame_height"] == source_height
    assert tile.tracked_objects[0]["bbox_coordinate_width"] == 2560
    assert tile.tracked_objects[0]["bbox_coordinate_height"] == 1440
    assert camera_wall.scale_overlay_bbox(
        tile.tracked_objects[0]["bbox"], 960, 540,
        tile.tracked_objects[0]["bbox_coordinate_width"],
        tile.tracked_objects[0]["bbox_coordinate_height"],
    ) == (480, 270, 960, 540)
    assert camera_wall.scale_overlay_bbox(
        tile.tracked_objects[0]["bbox"], 1920, 1080,
        tile.tracked_objects[0]["bbox_coordinate_width"],
        tile.tracked_objects[0]["bbox_coordinate_height"],
    ) == (960, 540, 1920, 1080)
    tile.close_reader()
    tile.deleteLater()


def test_preview_stage_timing_and_one_shot_startup_capture(app, monkeypatch, tmp_path):
    from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameWriter
    from services.frontend.app.camera_wall import PreviewFrameReader
    from PySide6.QtTest import QTest

    preview_path = tmp_path / "preview.bin"
    capture_dir = tmp_path / "capture"
    timing_path = tmp_path / "paint.jsonl"
    monkeypatch.setenv("V11_UI_PREVIEW_PATH_CAM01", str(preview_path))
    monkeypatch.setenv("MV3DT_UI_STARTUP_CAPTURE_DIR", str(capture_dir))
    monkeypatch.setenv("MV3DT_PREVIEW_LATENCY_LOG", str(timing_path))
    writer = PreviewFrameWriter(str(preview_path), width=16, height=8)
    tile = CameraTile("CAM-01", "http://invalid")
    try:
        sample_frame = bytes([17, 17, 17, 255]) * (16 * 8)
        writer.publish(sample_frame, timestamp_ns=time.monotonic_ns(),
                       decoder_reference_ns=time.monotonic_ns() - 3_000_000,
                       decoder_out_ns=time.monotonic_ns() - 2_000_000,
                       pts_ns=1234, source_frame_num=1)
        observed_after_sequence = []
        original_read_latest = PreviewFrameReader.read_latest

        def spy_read_latest(reader, *, max_age_sec=1.20, after_sequence=None, metadata_only=False):
            observed_after_sequence.append(after_sequence)
            return original_read_latest(reader, max_age_sec=max_age_sec,
                                        after_sequence=after_sequence, metadata_only=metadata_only)

        monkeypatch.setattr(PreviewFrameReader, "read_latest", spy_read_latest)
        tile.show()
        tile.refresh()
        # The latest committed frame is consumed in this event; the UI no
        # longer schedules a second worker-to-GUI handoff for each camera.
        deadline = time.monotonic() + 2.0
        while tile.last_version != 1 and time.monotonic() < deadline:
            QTest.qWait(5)
        assert tile.last_version == 1
        displayed = tile.video.grab().toImage()
        app.processEvents()
        assert displayed.pixelColor(displayed.width() // 2, displayed.height() // 2).red() == 17
        assert tile.video.pixmap().isNull()
        row = tile.video.preview_timing_context
        assert row["t7a_reader_poll_begin_monotonic_ns"] <= row["t7b_reader_return_monotonic_ns"]
        assert row["t7c_qimage_create_begin_monotonic_ns"] <= row["t7c_qimage_created_monotonic_ns"]
        # The current tile renderer paints the shared-memory QImage directly;
        # no intermediate scaled QImage or QPixmap is created.
        assert row["tile_transformation"] == "QPainter.SmoothPixmapTransform/direct-QImage"
        assert row["t7c_qimage_created_monotonic_ns"] <= row["t7d_qimage_draw_begin_monotonic_ns"]
        assert row["t7d_qimage_draw_begin_monotonic_ns"] <= row["t7e_scale_completed_monotonic_ns"]
        assert row["t7e_scale_completed_monotonic_ns"] <= row["t7f_overlay_completed_monotonic_ns"]
        from services.frontend.app.camera_wall import _flush_timing_handles
        _flush_timing_handles()
        paint = __import__("json").loads(timing_path.read_text().splitlines()[-1])
        assert paint["t8_ui_paint_monotonic_ns"] >= paint["t7f_overlay_completed_monotonic_ns"]
        assert (row["preview_width"], row["preview_height"], row["preview_stride"]) == (16, 8, 64)
        for stage in ("shared-memory", "qimage", "displayed"):
            metadata = __import__("json").loads((capture_dir / f"cam-01-{stage}.json").read_text())
            raw = (capture_dir / f"cam-01-{stage}.raw").read_bytes()
            assert metadata["sequence"] == 1 and metadata["source_frame_num"] == 1
            assert metadata["pts_ns"] == 1234 and metadata["ready"] is True
            assert metadata["payload_bytes"] == len(raw) > 0
        assert (capture_dir / "cam-01-shared-memory.raw").read_bytes() == sample_frame
        assert timing_path.exists()
        tile.refresh()
        deadline = time.monotonic() + 2.0
        while (not observed_after_sequence or observed_after_sequence[-1] != 1) and time.monotonic() < deadline:
            QTest.qWait(5)
        assert observed_after_sequence[0] == 0
        assert observed_after_sequence[-1] == 1
        assert tile.last_version == 1
    finally:
        tile.close_reader()
        tile.deleteLater()
        writer.close()
