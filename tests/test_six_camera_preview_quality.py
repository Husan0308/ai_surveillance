from __future__ import annotations

from pathlib import Path
import json

from scripts.build_six_camera_v13_wall_candidate import CANDIDATE


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "services/camera_v11/native/six_camera_v13"


def test_preview_preparation_is_parallel_dual_resolution_and_exact_frame_join():
    detection = (NATIVE / "detection.hpp").read_text()
    main = (NATIVE / "main.cpp").read_text()

    assert "F5R_PREVIEW_W = 960, F5R_PREVIEW_H = 540" in detection
    assert "F5R_FULLSCREEN_W = 1920, F5R_FULLSCREEN_H = 1080" in detection
    assert '"interpolation-method", 4' in detection
    assert "f5r_make_preview_branch(sources[i], false)" in detection
    assert "f5r_make_preview_branch(sources[i], true)" in detection
    assert 'make("valve", make_name("valve").c_str())' in detection
    assert "f5r_fullscreen_selected.load()" in detection
    assert "f5r_tracker_frame_ready" in detection
    assert "frame.frame_num != tracker.source_frame_num || frame.pts != tracker.pts" in detection
    assert "F5R_PREVIEW_METADATA_OFFSET" in detection
    assert "version = 5" in detection
    assert "prepare_start_ns" in detection
    assert "tracker->tee" in detection or "link(tracker,tracked_tee)" in detection
    assert "nvstreamdemux->per-camera" not in detection
    assert "system_convert" not in detection
    assert "f5r_make_preview_branch(sources[i], false)" in detection
    assert "V11_MUX_WIDTH" in main and "V11_MUX_HEIGHT" in main
    assert "v11_mux_width = 2560" in main and "v11_mux_height = 1440" in main
    if (CANDIDATE / "build-manifest.json").is_file():
        profile = json.loads((CANDIDATE / "build-manifest.json").read_text())["preview_output"]
        assert profile["tile"] == {"width": 960, "height": 540, "format": "BGRx"}
        assert profile["fullscreen"]["width"] == 1920
        assert profile["fullscreen"]["camera_branches_open_simultaneously"] == 1
        assert "source-frame sequence+PTS join" in profile["preparation"]


def test_ui_fullscreen_reads_a_dedicated_1920_buffer_not_the_tile_image():
    wall = (ROOT / "services/frontend/app/camera_wall.py").read_text()

    assert wall.count("Qt.TransformationMode.SmoothTransformation") >= 1
    assert wall.count("Qt.AspectRatioMode.KeepAspectRatio") >= 1
    assert "QPainter.RenderHint.SmoothPixmapTransform" in wall
    assert "pixmap.scaled(" in wall
    assert "PreviewFrameReader(FULLSCREEN_PREVIEW_PATH)" in wall
    assert "self.fullscreen_reader.read_latest" in wall
    assert "frame.width != 1920 or frame.height != 1080" in wall
    assert "QPixmap.fromImage(self.fullscreen_current_image)" in wall
    assert "QPixmap.fromImage(self.current_image)" not in wall
    assert "painter.drawImage(QRectF(offset_x, offset_y, display_width, display_height), image)" in wall
    assert wall.count("if pixmap.size() != target:") == 1
    draw_frame = wall.split("    def _draw_frame(", 1)[1].split("    def _capture_startup_stage(", 1)[0]
    assert "QPixmap.fromImage" not in draw_frame
    assert "self.current_frame_pixmap" in wall
    assert "fullscreen_image" in wall


def test_capture_hooks_are_opt_in_and_record_matching_source_frame():
    detection = (NATIVE / "detection.hpp").read_text()
    wall = (ROOT / "services/frontend/app/camera_wall.py").read_text()

    assert "V11_PREVIEW_CAPTURE_SOURCE_FRAME" in detection
    assert "V11_PREVIEW_CAPTURE_DIR" in detection
    assert "decoded-source" in detection
    assert "preview-input" in detection
    assert "MV3DT_UI_CAPTURE_FRAME_NUM" in wall
    assert '"fullscreen"' in wall


def test_frame_timer_returns_to_proven_default_interval():
    config = (ROOT / "services/frontend/app/config.py").read_text()
    main = (ROOT / "services/frontend/app/main.py").read_text()
    assert 'os.getenv("FRONTEND_FRAME_REFRESH_INTERVAL_MS", "16")' in config
    assert 'os.getenv("FRONTEND_FRAME_REFRESH_INTERVAL_MS", "4")' not in config
    assert 'os.getenv("FRONTEND_NOTIFY_WATCHDOG_INTERVAL_MS", "100")' in config
    assert "effective_frame_refresh_interval_ms(" in main
    assert "self.camera_wall.preview_notify_notifier is not None" in main
    assert "tile.refresh()" in (ROOT / "services/frontend/app/camera_wall.py").read_text()


def test_notification_watchdog_does_not_delay_event_driven_refresh():
    from services.frontend.app.config import effective_frame_refresh_interval_ms

    assert effective_frame_refresh_interval_ms(16, True, 100) == 100
    assert effective_frame_refresh_interval_ms(200, True, 100) == 200
    assert effective_frame_refresh_interval_ms(16, False, 100) == 16
    assert effective_frame_refresh_interval_ms(4, True, 100) == 100


def test_new_preview_frames_wake_only_the_affected_ui_tile():
    wall = (ROOT / "services/frontend/app/camera_wall.py").read_text()
    ipc = (ROOT / "services/camera_v11/ui_preview_ipc_v1.py").read_text()
    assert "QSocketNotifier" in wall
    assert "PreviewFrameNotifyReceiver" in wall
    assert "events = receiver.drain()" in wall
    assert 'if kind == "tile":' in wall and "tile.refresh()" in wall
    assert 'elif kind == "fullscreen":' in wall and "tile._refresh_fullscreen_preview" in wall
    assert "t7_notify_received_monotonic_ns" in wall
    assert "socket.AF_UNIX, socket.SOCK_DGRAM | socket.SOCK_NONBLOCK" in ipc
    assert "DEFAULT_NOTIFY_PATH" in ipc


def test_visible_tile_uses_one_immediate_paint_request_per_new_frame():
    wall = (ROOT / "services/frontend/app/camera_wall.py").read_text()
    update = wall.split("    def _update_tile_pixmap(", 1)[1].split(
        "    def _update_fullscreen_pixmap(", 1
    )[0]

    assert "if self.video.isVisible():\n            self.video.repaint()\n        else:\n            self.video.update()" in update
    assert update.count("self.video.update()") == 1
    assert update.count("self.video.repaint()") == 1
    assert "self.video.update()\n        if self.video.isVisible():\n            self.video.repaint()" not in update


def test_decoder_timestamp_caches_evict_in_insertion_order():
    main = (NATIVE / "main.cpp").read_text()
    assert "decoder_input_order.emplace_back(pts, sequence)" in main
    assert "decoded_clock_order.emplace_back(pts, stamp.sequence)" in main
    assert "decoder_input_order.pop_front()" in main
    assert "decoded_clock_order.pop_front()" in main
    assert "decoder_input_clock_ns.erase(ctx->decoder_input_clock_ns.begin())" not in main
    assert "decoded_clock_ns.erase(owner->decoded_clock_ns.begin())" not in main
