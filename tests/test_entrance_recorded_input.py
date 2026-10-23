from __future__ import annotations

import ast
import inspect
import threading
import time
import queue
from types import SimpleNamespace

import numpy as np
import pytest

from services.mv3dt_staging.live_tracking import recorded_input
from services.mv3dt_staging.live_tracking import live_bridge as live_bridge_module
from services.mv3dt_staging.live_tracking.live_bridge import (
    RecordedCaptureWorker,
    source_panel_mode_label,
)


def test_recorded_view_labels_never_claim_physical_live_source():
    assert source_panel_mode_label(native_frame=True, recorded_mode=True).startswith("RECORDED")
    assert source_panel_mode_label(native_frame=False, recorded_mode=True).startswith("RECORDED")
    assert source_panel_mode_label(native_frame=True, recorded_mode=False).startswith("LIVE")


def test_video_pane_initializes_recorded_label_and_exact_frame_cursor():
    # VideoPane is intentionally local to run_live; inspect the parsed method
    # contract so the first paint and initial map-screenshot check are safe.
    tree = ast.parse(inspect.getsource(live_bridge_module.run_live))
    pane = next(node for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name == "VideoPane")
    init = next(node for node in pane.body
                if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    assigned = {
        node.targets[0].attr
        for node in ast.walk(init)
        if isinstance(node, ast.Assign) and node.targets
        and isinstance(node.targets[0], ast.Attribute)
        and isinstance(node.targets[0].value, ast.Name)
        and node.targets[0].value.id == "self"
    }
    assert {"recorded_mode", "_shown_index"} <= assigned
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "VideoPane"]
    assert len(calls) == 2
    assert all(any(keyword.arg == "recorded_mode" for keyword in call.keywords)
               for call in calls)


@pytest.mark.parametrize("camera_id", ["CAM-02", "CAM-05"])
def test_original_recording_hash_metadata_and_first_decoded_pts(camera_id):
    source_info = recorded_input.SOURCE_BY_CAMERA[camera_id]
    source = recorded_input.inspect_recording(camera_id, source_info["default_path"])
    assert source["sha256"] == source_info["sha256"]
    assert (source["width"], source["height"]) == (
        source_info["width"], source_info["height"]
    )
    assert source["frame_count"] == source_info["frames"]
    assert source["source_id"] == (0 if camera_id == "CAM-02" else 1)
    assert source["pts_ns_by_decoded_frame"][0] >= 0
    assert all(a < b for a, b in zip(
        source["pts_ns_by_decoded_frame"], source["pts_ns_by_decoded_frame"][1:]
    ))

    capture = recorded_input.GstRecordedFileCapture(source)
    try:
        frame = capture.pull(timeout_ms=5000)
        assert frame is not None
        assert frame.frame_index == 0
        assert frame.source_pts_ns == source["pts_ns_by_decoded_frame"][0]
        assert frame.bgr.shape == (source_info["height"], source_info["width"], 3)
        assert frame.bgr.dtype == np.uint8
        assert frame.bgr.flags.c_contiguous
        assert frame.bgr.flags.owndata
        assert frame.bgr.base is None
    finally:
        closed = capture.close()
    assert closed["close_state"] == "NULL"
    assert capture.decoded_frames == 1


def test_recorded_playback_maps_source_pts_delta_to_monotonic_deadline():
    assert recorded_input.playback_deadline_ns(5_000_000_000, 2_000_000_000, 2_050_000_000) == 5_050_000_000
    assert recorded_input.playback_deadline_ns(
        5_000_000_000, 2_000_000_000, 3_000_000_000, playback_speed=2.0
    ) == 5_500_000_000
    with pytest.raises(ValueError, match="ordered nonnegative"):
        recorded_input.playback_deadline_ns(5, 20, 19)
    with pytest.raises(ValueError, match="finite and positive"):
        recorded_input.playback_deadline_ns(0, 0, 1, playback_speed=float("nan"))


def test_recorded_worker_preserves_camera_frame_and_source_pts_crosswalk(monkeypatch):
    frames = [
        recorded_input.RecordedFrame(i, 1_000_000_000 + i * 50_000_000,
                                     900_000_000 + i * 50_000_000, 50_000_000,
                                     np.zeros((4, 8, 3), dtype=np.uint8))
        for i in range(3)
    ]

    class FakeCapture:
        def __init__(self, source, *, appsink_buffers):
            assert source["camera_id"] == "CAM-02"
            assert appsink_buffers == 1
            self.rows = iter(frames)
            self.eos_seen = False
            self.closed = False

        def pull(self, timeout_ms=500):
            try:
                return next(self.rows)
            except StopIteration:
                self.eos_seen = True
                return None

        def close(self):
            self.closed = True
            return {"close_state": "NULL", "eos_seen": self.eos_seen}

    monkeypatch.setattr(recorded_input, "GstRecordedFileCapture", FakeCapture)
    source = {
        "camera_id": "CAM-02", "source_id": 0, "path": "/private/mock.mp4",
        "sha256": "a" * 64,
        "pts_ns_by_decoded_frame": [row.source_pts_ns for row in frames],
    }
    queued = []
    recorded = []
    controller = SimpleNamespace(
        stop_event=threading.Event(), args=SimpleNamespace(native_frame=True),
        stats={0: {"recorded_gst_pull_ns": [], "recorded_playback_lateness_ns": [],
                   "frame_preprocess_ns": []}},
        error_queue=queue.Queue(),
        queue_frame=lambda *args, **kwargs: queued.append((args, kwargs)),
        record_recorded_frame=lambda *args: recorded.append(args),
    )
    worker = RecordedCaptureWorker(
        controller, 0, source, lambda image: image,
        start_frame=0, duration_sec=0.10,
    )
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert worker.error == ""
    assert worker.state == "BOUNDED_COMPLETE"
    assert worker.capture_close_succeeded is True
    assert worker.source_close_state == "NULL"
    assert [row[0][1] for row in queued] == [0, 1]
    assert [row[0][3] for row in queued] == [frames[0].source_pts_ns, frames[1].source_pts_ns]
    assert all(row[1]["display_image"] is frames[index].bgr
               for index, row in enumerate(queued))
    assert [row[2].frame_index for row in recorded] == [0, 1]
    assert all(row[2].source_pts_ns == queued[index][0][3]
               for index, row in enumerate(recorded))
