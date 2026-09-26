from __future__ import annotations

import os
import time

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
    monkeypatch.setattr(camera_wall, "overlay_source_dimensions", lambda camera, mode="live": (1920, 1080))
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


def test_fullscreen_uses_full_shared_preview_without_stretching(app, monkeypatch):
    from services.frontend.app import camera_wall

    monkeypatch.setenv("FRONTEND_USE_V11_SHARED_MEMORY", "1")
    monkeypatch.setattr(camera_wall, "overlay_source_dimensions", lambda camera, mode="live": (3200, 1800))
    tile = CameraTile("CAM-04", "http://127.0.0.1:8101")
    tile.video.resize(800, 600)
    tile._draw_frame(QImage(1920, 1080, QImage.Format.Format_RGB32))

    assert tile.current_frame_pixmap.size() == QSize(1920, 1080)
    assert tile.displayed_pixmap_size.width() / tile.displayed_pixmap_size.height() == pytest.approx(16 / 9)

    tile._open_fullscreen()
    app.processEvents()
    displayed = tile.fullscreen_video.pixmap()
    assert displayed is not None
    assert displayed.width() / displayed.height() == pytest.approx(16 / 9, abs=0.02)
    assert tile.current_frame_pixmap.width() == 1920  # no enlargement of the shared-memory intermediate
    tile.close_reader()
    tile.deleteLater()


def test_shared_preview_stale_frame_is_dropped_not_kept_visible(app, monkeypatch):
    monkeypatch.setenv("FRONTEND_USE_V11_SHARED_MEMORY", "1")
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
