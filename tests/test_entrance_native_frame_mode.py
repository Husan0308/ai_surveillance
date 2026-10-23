from collections import OrderedDict
import queue
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from services.mv3dt_staging.live_tracking.live_bridge import (
    AUTHORIZED_RTSP_MAX_SECONDS,
    RTSP_CAPTURE_STOP_SECONDS,
    CaptureWorker,
    LiveController,
    live_gpu_guard_reason,
    rtsp_close_within_authorized_cap,
    rtsp_deadline_ns,
)
from services.mv3dt_staging.live_tracking.native_frame_bridge import (
    NativeFullFovScale,
    detector_box_to_source,
)
from services.camera_v11.source_ownership import SourceOwnership


def _capture_worker(fake_capture, *, deadline_reached=False):
    from services.ml_service.app.config import DeepStreamConfig

    stop_event = threading.Event()
    if not deadline_reached:
        stop_event.set()  # Exercise startup/cleanup without reading or opening a real stream.
    controller = SimpleNamespace(
        stop_event=stop_event,
        error_queue=queue.Queue(),
        register_rtsp_attempt=lambda _worker: None,
        rtsp_deadline_reached=lambda: deadline_reached,
        stats={},
        args=SimpleNamespace(native_frame=True),
        queue_frame=lambda *_args, **_kwargs: pytest.fail("test capture must not queue frames"),
    )
    camera = SimpleNamespace(
        camera_id="CAM-02", uri="rtsp://secret-user:secret-password@camera.invalid/stream",
        username="secret-user", password="secret-password",
        decoder_low_latency_mode=False, decoder_extra_surfaces=None,
    )
    settings = SimpleNamespace(deepstream=DeepStreamConfig(
        gpu_id=0, rtsp_transport="tcp", latency_ms=100, drop_on_latency=True,
        decoder_extra_surfaces=2, udp_buffer_size=0, postdecode_queue_buffers=2,
        capture_timeout_ms=1200, startup_grace_sec=1.0, reconnect_delay_sec=1.0,
        reconnect_delay_max_sec=5.0, startup_stagger_sec=0.0,
        display_width=1920, display_height=1080, display_fps=20, jpeg_quality=70,
    ))
    worker = CaptureWorker(controller, 0, camera, settings, lambda frame: frame)
    worker._create_capture = lambda _config: fake_capture
    return worker, controller


@pytest.mark.parametrize(
    ("camera", "shape", "scale"),
    [
        ("CAM-02", (1800, 3200, 3), 0.60),
        ("CAM-05", (1440, 2560, 3), 0.75),
    ],
)
def test_raw_full_fov_scale_is_strict_and_preserves_canvas(camera, shape, scale):
    x_gradient = np.broadcast_to(
        np.linspace(0, 255, shape[1], dtype=np.uint8)[None, :, None],
        shape,
    ).copy()
    result = NativeFullFovScale(camera)(x_gradient)
    assert result.shape == (1080, 1920, 3)
    assert result.dtype == np.uint8
    assert result.flags.c_contiguous
    assert result[540, 0, 0] < 3
    assert result[540, -1, 0] > 252
    assert scale == 1920 / shape[1]


def test_native_scale_rejects_unknown_geometry_instead_of_silently_resizing():
    with pytest.raises(ValueError, match="expected native uint8 BGR"):
        NativeFullFovScale("CAM-02")(np.zeros((1080, 1920, 3), dtype=np.uint8))


def test_detector_boxes_map_back_to_source_pixels_without_mutating_raw_metadata():
    original = {"left": 600.0, "top": 540.0, "width": 300.0, "height": 180.0,
                "class_id": 0, "tracker_id": 17}
    cam02 = detector_box_to_source(original, "CAM-02")
    cam05 = detector_box_to_source(original, "CAM-05")
    assert cam02["left"] == 1000.0 and cam02["top"] == 900.0
    assert cam02["width"] == 500.0 and cam02["height"] == 300.0
    assert cam05["left"] == 800.0 and cam05["top"] == 720.0
    assert cam05["width"] == 400.0 and cam05["height"] == 240.0
    assert original["left"] == 600.0 and original["tracker_id"] == 17


def test_live_gpu_guard_stops_on_sustained_overload_or_lost_telemetry():
    assert live_gpu_guard_reason([
        {"status": "AVAILABLE", "gpu_utilization_percent": 20},
        {"status": "AVAILABLE", "gpu_utilization_percent": 97},
        {"status": "AVAILABLE", "gpu_utilization_percent": 96},
    ])
    assert live_gpu_guard_reason([
        {"status": "AVAILABLE", "gpu_utilization_percent": 20},
        {"status": "UNAVAILABLE"},
        {"status": "UNAVAILABLE"},
    ])
    assert live_gpu_guard_reason([
        {"status": "AVAILABLE", "gpu_utilization_percent": 20},
        {"status": "AVAILABLE", "gpu_utilization_percent": 80},
        {"status": "AVAILABLE", "gpu_utilization_percent": 90},
    ]) is None


def test_rtsp_capture_deadline_is_anchored_to_first_connection_attempt():
    first_attempt_ns = 12_345_000_000
    deadline_ns = rtsp_deadline_ns(first_attempt_ns)
    assert deadline_ns - first_attempt_ns == round(RTSP_CAPTURE_STOP_SECONDS * 1e9)
    assert deadline_ns < first_attempt_ns + round(AUTHORIZED_RTSP_MAX_SECONDS * 1e9)


def test_rtsp_hard_stop_reserves_teardown_time_inside_authorized_cap():
    assert RTSP_CAPTURE_STOP_SECONDS == 57.0
    assert AUTHORIZED_RTSP_MAX_SECONDS - RTSP_CAPTURE_STOP_SECONDS == 3.0
    first_attempt_ns = 10_000_000_000
    assert rtsp_close_within_authorized_cap(first_attempt_ns, first_attempt_ns + 59_999_999_999)
    assert not rtsp_close_within_authorized_cap(first_attempt_ns, first_attempt_ns + 60_000_000_001)
    assert not rtsp_close_within_authorized_cap(first_attempt_ns, first_attempt_ns - 1)


def test_rtsp_deadline_rejects_invalid_cap_inputs():
    import pytest

    with pytest.raises(ValueError, match="invalid RTSP"):
        rtsp_deadline_ns(1_000_000_000, 60.0)


def test_rtsp_watchdog_stops_capture_without_waiting_for_the_ui_timer():
    class WatchdogProbe:
        def __init__(self):
            self.stop_event = threading.Event()
            self._rtsp_deadline_ready = threading.Event()
            self._rtsp_deadline_ready.set()
            self.rtsp_deadline_ns = time.monotonic_ns() + 25_000_000
            self.rtsp_cap_triggered_ns = None
            self.rtsp_cap_triggered_utc = None
            self.rtsp_stop_reason = ""
            self.run_phase = "LIVE"
            self.stop_called = threading.Event()

        def stop(self):
            self.stop_called.set()

    probe = WatchdogProbe()
    worker = threading.Thread(target=LiveController._watch_rtsp_deadline, args=(probe,))
    worker.start()
    worker.join(timeout=0.5)
    assert not worker.is_alive()
    assert probe.stop_event.is_set()
    assert probe.stop_called.is_set()
    assert probe.rtsp_cap_triggered_ns >= probe.rtsp_deadline_ns


def test_capture_close_is_called_and_reported_on_normal_shutdown():
    class FakeCapture:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    capture = FakeCapture()
    worker, _controller = _capture_worker(capture)
    worker.start()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert capture.close_calls == 1
    assert worker.capture_close_attempted is True
    assert worker.capture_close_succeeded is True
    assert worker.capture_close_error_type is None
    assert worker.close_started_ns > 0
    assert worker.close_return_ns >= worker.close_started_ns


def test_capture_close_exception_is_visible_without_leaking_uri_credentials():
    class FakeCapture:
        def close(self):
            raise RuntimeError("failure at rtsp://secret-user:secret-password@camera.invalid/stream")

    worker, controller = _capture_worker(FakeCapture())
    worker.start()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert worker.state == "ERROR"
    assert worker.capture_close_attempted is True
    assert worker.capture_close_succeeded is False
    assert worker.capture_close_error_type == "RuntimeError"
    assert "secret-password" not in worker.error
    assert "camera.invalid" not in worker.error
    assert "RTSP capture close failed (RuntimeError)" in controller.error_queue.get_nowait()


def test_capture_close_block_keeps_worker_alive_until_close_returns():
    close_started = threading.Event()
    allow_close_return = threading.Event()

    class FakeCapture:
        def close(self):
            close_started.set()
            assert allow_close_return.wait(timeout=1)

    worker, _controller = _capture_worker(FakeCapture())
    worker.start()
    assert close_started.wait(timeout=1)
    assert worker.is_alive()
    assert worker.close_started_ns > 0
    assert worker.close_return_ns == 0
    assert worker.capture_close_succeeded is None

    allow_close_return.set()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert worker.capture_close_succeeded is True
    assert worker.close_return_ns >= worker.close_started_ns


def test_deadline_expiry_skips_late_read_and_still_closes_capture():
    class FakeCapture:
        def __init__(self):
            self.read_calls = 0
            self.close_calls = 0

        def read(self):
            self.read_calls += 1
            pytest.fail("deadline-expired worker must not read another frame")

        def close(self):
            self.close_calls += 1

    capture = FakeCapture()
    worker, _controller = _capture_worker(capture, deadline_reached=True)
    worker.start()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert capture.read_calls == 0
    assert capture.close_calls == 1
    assert worker.capture_close_succeeded is True


def test_camera_locks_are_retained_until_late_capture_worker_terminates(tmp_path):
    lock_dir = tmp_path / "locks"
    output = tmp_path / "run"
    output.mkdir()
    controller = LiveController(SimpleNamespace(detector="yolo26m"), output)

    class LateWorker:
        name = "blocked-capture-close"

        def __init__(self):
            self.alive = True
            self.join_calls = 0
            self.terminated = threading.Event()

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            self.join_calls += 1
            # The first call is the bounded stop join; subsequent reaper joins wait.
            if self.join_calls > 1:
                self.terminated.wait(timeout=timeout)

    late_worker = LateWorker()
    controller.capture_workers = [late_worker]
    controller.pipe = SimpleNamespace(
        end_input=lambda: None,
        stop=lambda timeout_sec: None,
        process=SimpleNamespace(returncode=0),
    )
    controller.process_events = lambda: []
    controller.ownership = SourceOwnership(["CAM-02", "CAM-05"], lock_dir)
    controller.ownership.__enter__()

    controller.stop()
    assert controller._stopped is True
    assert controller.ownership is not None
    with pytest.raises(RuntimeError, match="another camera owner"):
        with SourceOwnership(["CAM-02"], lock_dir):
            pytest.fail("camera lock must remain held while capture worker is alive")

    late_worker.alive = False
    late_worker.terminated.set()
    end = time.monotonic() + 1.0
    while controller.ownership is not None and time.monotonic() < end:
        time.sleep(0.01)
    assert controller.ownership is None
    assert controller.ownership_released_ns is not None
    with SourceOwnership(["CAM-02", "CAM-05"], lock_dir):
        pass


def test_late_tracker_metadata_is_never_painted_on_a_different_frame():
    controller = LiveController.__new__(LiveController)
    controller.frame_lock = threading.Lock()
    frame = np.full((3, 4, 3), 17, dtype=np.uint8)
    controller.frame_cache = {0: OrderedDict({8: (frame, 800, 799, 880)}), 1: OrderedDict()}
    controller.latest_meta = {
        0: {"source_id": 0, "input_frame_index": 8, "source_pts_ns": 800, "objects": []},
        1: None,
    }

    payload = controller.frame_payload(0)
    assert payload is not None
    assert payload[0] is frame
    assert payload[1]["input_frame_index"] == 8
    assert controller.frame_payload(1) is None

    # A late result for an evicted frame must be dropped, not paired with the
    # current image or another slot's pixels.
    controller.latest_meta[0] = {"source_id": 0, "input_frame_index": 7,
                                 "source_pts_ns": 700, "objects": []}
    assert controller.frame_payload(0) is None
