import hashlib
import json
import os
import threading
import time

import cv2
import numpy as np
import pytest

from services.mv3dt_staging.live_tracking.live_bridge import validate_live_duration
from services.mv3dt_staging.live_tracking.visual_evidence import FrameEvidenceCapture


def _frame_metadata(*, source_id=1, frame_index=4, source_pts_ns=2_000_000_000,
                    source_dts_ns=-1, arrival_mono_ns=3_000_000_000):
    return {
        "type": "TRACK", "source_id": source_id, "input_frame_index": frame_index,
        "source_pts_ns": source_pts_ns, "source_dts_ns": source_dts_ns,
        "arrival_mono_ns": arrival_mono_ns, "frame_num": frame_index + 10,
        "buf_pts": 9_000_000_000 + frame_index, "batch_id": source_id,
        "stage_mono_ns": 4_000_000_000,
        "objects": [{
            "class_id": 0, "confidence": 0.91, "tracker_id": 47,
            "left": 240.0, "top": 270.0, "width": 192.0, "height": 135.0,
        }],
    }


def test_visual_evidence_preserves_exact_raw_frame_and_local_id_overlay(tmp_path):
    # CAM-05 native shape. The marker pattern makes accidental cross-frame joins visible.
    frame = np.empty((1440, 2560, 3), dtype=np.uint8)
    frame[:, :] = (11, 23, 37)
    frame[100:140, 200:260] = (80, 120, 160)
    frame.setflags(write=False)
    evidence_dir = tmp_path / "visual-evidence"
    recorder = FrameEvidenceCapture(evidence_dir, sample_every_frames=4,
                                    max_samples_per_camera=10, queue_size=2)
    assert recorder.submit(1, 4, frame, _frame_metadata(),
                           cached_source_pts_ns=2_000_000_000,
                           cached_source_dts_ns=-1,
                           cached_arrival_mono_ns=3_000_000_000)
    summary = recorder.close()

    assert summary["camera_counts"]["CAM-05"]["written"] == 1
    assert summary["camera_counts"]["CAM-02"]["written"] == 0
    raw_path = evidence_dir / "CAM-05/raw/frame_000004.bgr"
    annotated_path = evidence_dir / "CAM-05/annotated/frame_000004.jpg"
    raw = np.fromfile(raw_path, dtype=np.uint8).reshape(frame.shape)
    annotated = cv2.imdecode(np.fromfile(annotated_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(raw, frame)  # lossless, same original source BGR pixels
    assert annotated.shape == frame.shape
    # 1920x1080 detector box mapped back at 4/3 to CAM-05 native coordinates.
    assert np.linalg.norm(annotated[450, 320].astype(int) - raw[450, 320].astype(int)) > 100

    record = json.loads((evidence_dir / "CAM-05/frames.jsonl").read_text().strip())
    assert record["camera_id"] == "CAM-05"
    assert record["source_id"] == 1
    assert record["input_frame_index"] == 4
    assert record["source_pts_ns"] == 2_000_000_000
    assert record["deepstream_frame_num"] == 14
    assert record["tracker_objects_native"][0]["tracker_id"] == 47
    assert record["tracker_objects_native"][0]["native_box"]["left"] == 320.0
    assert record["raw_frame_sha256"] == hashlib.sha256(frame.tobytes()).hexdigest()
    assert record["raw_bgr_sha256"] == hashlib.sha256(frame.tobytes()).hexdigest()
    assert os.stat(raw_path).st_mode & 0o777 == 0o600
    assert os.stat(annotated_path).st_mode & 0o777 == 0o600
    assert os.stat(evidence_dir).st_mode & 0o777 == 0o700
    assert summary["latency_ms"]["CAM-05"]["write_latency_ns"]["count"] == 1
    assert summary["camera_counts"]["CAM-05"]["bytes_written"] > frame.nbytes


def test_visual_evidence_sampling_and_camera_frame_crosswalk_fail_closed(tmp_path):
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    recorder = FrameEvidenceCapture(tmp_path / "visual-evidence", sample_every_frames=4,
                                    max_samples_per_camera=2, queue_size=2)
    assert recorder.submit(1, 5, frame, _frame_metadata(frame_index=5),
                           cached_source_pts_ns=2_000_000_000,
                           cached_source_dts_ns=-1,
                           cached_arrival_mono_ns=3_000_000_000) is False
    with pytest.raises(ValueError, match="frame index"):
        recorder.submit(1, 8, frame, _frame_metadata(frame_index=7),
                        cached_source_pts_ns=2_000_000_000,
                        cached_source_dts_ns=-1,
                        cached_arrival_mono_ns=3_000_000_000)
    with pytest.raises(ValueError, match="source PTS"):
        recorder.submit(1, 8, frame, _frame_metadata(frame_index=8, source_pts_ns=-1),
                        cached_source_pts_ns=-1,
                        cached_source_dts_ns=-1,
                        cached_arrival_mono_ns=3_000_000_000)
    recorder.close()
    assert list((tmp_path / "visual-evidence/CAM-05/raw").iterdir()) == []


def test_live_duration_can_be_capped_to_42_seconds_with_45_second_teardown_budget():
    validate_live_duration(capture_stop_seconds=42, authorized_max_seconds=45,
                           smoke_seconds=15, review_seconds=20)
    with pytest.raises(ValueError, match="teardown"):
        validate_live_duration(capture_stop_seconds=43, authorized_max_seconds=45,
                               smoke_seconds=15, review_seconds=20)
    with pytest.raises(ValueError, match="at most 60s"):
        validate_live_duration(capture_stop_seconds=60, authorized_max_seconds=61,
                               smoke_seconds=15, review_seconds=20)


def test_visual_evidence_writer_keeps_up_at_bounded_live_sample_rate(tmp_path):
    recorder = FrameEvidenceCapture(tmp_path / "visual-evidence", sample_every_frames=1,
                                    max_samples_per_camera=8, queue_size=16)
    frames = {
        0: np.full((1800, 3200, 3), (13, 29, 41), dtype=np.uint8),
        1: np.full((1440, 2560, 3), (17, 31, 43), dtype=np.uint8),
    }
    next_pts = {0: 10_000_000_000, 1: 20_000_000_000}
    for frame_index in range(6):
        for source_id in (0, 1):
            arrival = 30_000_000_000 + frame_index * 100_000_000 + source_id
            pts = next_pts[source_id] + frame_index * 50_000_000
            metadata = {
                "source_id": source_id, "input_frame_index": frame_index,
                "source_pts_ns": pts, "source_dts_ns": -1, "arrival_mono_ns": arrival,
                "objects": [],
            }
            assert recorder.submit(source_id, frame_index, frames[source_id], metadata,
                                   cached_source_pts_ns=pts, cached_source_dts_ns=-1,
                                   cached_arrival_mono_ns=arrival)
            time.sleep(0.075)
    summary = recorder.close(timeout_sec=10)
    assert summary["camera_counts"]["CAM-02"]["written"] == 6
    assert summary["camera_counts"]["CAM-05"]["written"] == 6
    assert summary["camera_counts"]["CAM-02"]["queue_dropped"] == 0
    assert summary["camera_counts"]["CAM-05"]["queue_dropped"] == 0


def test_identity_change_can_force_sparse_exact_frame_sample(tmp_path):
    recorder = FrameEvidenceCapture(tmp_path / "visual-evidence", sample_every_frames=10,
                                    max_samples_per_camera=4, queue_size=2)
    frame = np.zeros((1800, 3200, 3), dtype=np.uint8)
    frame.setflags(write=False)
    meta = _frame_metadata(source_id=0, frame_index=7, source_pts_ns=4_000_000_000,
                           source_dts_ns=-1, arrival_mono_ns=6_000_000_000)
    assert recorder.submit(0, 7, frame, meta, cached_source_pts_ns=4_000_000_000,
                           cached_source_dts_ns=-1, cached_arrival_mono_ns=6_000_000_000,
                           force=True)
    summary = recorder.close()
    assert summary["camera_counts"]["CAM-02"]["written"] == 1
    record = json.loads((tmp_path / "visual-evidence/CAM-02/frames.jsonl").read_text().strip())
    assert record["input_frame_index"] == 7
    assert record["sample_reason"] == "identity_set_change"


def test_full_writer_queue_drops_evidence_explicitly_without_blocking_caller(tmp_path):
    recorder = FrameEvidenceCapture(tmp_path / "visual-evidence", sample_every_frames=1,
                                    max_samples_per_camera=8, queue_size=1)
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    frame.setflags(write=False)
    original_write = recorder._write_item
    first_item_started = threading.Event()
    release_writer = threading.Event()

    def slow_first_write(item, manifest):
        if item["frame_index"] == 0:
            first_item_started.set()
            assert release_writer.wait(timeout=5)
        original_write(item, manifest)

    recorder._write_item = slow_first_write

    def submit(index):
        pts = 2_000_000_000 + index * 50_000_000
        arrival = 3_000_000_000 + index * 50_000_000
        meta = _frame_metadata(frame_index=index, source_pts_ns=pts,
                               source_dts_ns=-1, arrival_mono_ns=arrival)
        return recorder.submit(1, index, frame, meta, cached_source_pts_ns=pts,
                               cached_source_dts_ns=-1, cached_arrival_mono_ns=arrival)

    assert submit(0)
    assert first_item_started.wait(timeout=5)
    assert submit(1)  # occupies the one waiting slot while the writer is held
    start_ns = time.monotonic_ns()
    assert submit(2) is False
    caller_latency_ms = (time.monotonic_ns() - start_ns) / 1e6
    release_writer.set()
    summary = recorder.close(timeout_sec=10)
    counts = summary["camera_counts"]["CAM-05"]
    assert counts["queue_dropped"] == 1
    assert counts["submitted"] == 2
    assert counts["written"] == 2
    assert caller_latency_ms < 50.0


def test_async_disk_failure_is_reported_without_blocking_submitter(tmp_path):
    recorder = FrameEvidenceCapture(tmp_path / "visual-evidence", sample_every_frames=1,
                                    max_samples_per_camera=8, queue_size=2)
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    frame.setflags(write=False)

    def fail_write(_item, _manifest):
        raise OSError("synthetic disk failure")

    recorder._write_item = fail_write
    meta = _frame_metadata(frame_index=4)
    assert recorder.submit(1, 4, frame, meta, cached_source_pts_ns=2_000_000_000,
                           cached_source_dts_ns=-1, cached_arrival_mono_ns=3_000_000_000)
    deadline = time.monotonic() + 3
    while recorder.error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert recorder.error is not None
    start_ns = time.monotonic_ns()
    assert recorder.submit(1, 5, frame, _frame_metadata(frame_index=5),
                           cached_source_pts_ns=2_000_000_000,
                           cached_source_dts_ns=-1,
                           cached_arrival_mono_ns=3_000_000_000) is False
    assert (time.monotonic_ns() - start_ns) / 1e6 < 50.0
    with pytest.raises(RuntimeError, match="synthetic disk failure"):
        recorder.close()
    counts = recorder.summary()["camera_counts"]["CAM-05"]
    assert counts["write_failed"] == 1
    assert counts["writer_error_skipped"] == 1


def test_writer_supports_devroom_frame_geometry_but_rejects_unvalidated_bbox_mapping(tmp_path):
    frame = np.full((48, 64, 3), 83, dtype=np.uint8)
    frame.setflags(write=False)
    raw_hash = hashlib.sha256(frame.tobytes()).hexdigest()
    output = tmp_path / "dev-room-evidence"
    recorder = FrameEvidenceCapture(
        output, sample_every_frames=1, max_samples_per_camera=2, queue_size=2,
        camera_by_source_id={0: "CAM-01", 3: "CAM-04"},
        source_size_by_camera={"CAM-01": (64, 48), "CAM-04": (64, 48)},
    )
    metadata = {
        "source_id": 3, "input_frame_index": 9, "source_pts_ns": 900_000_000,
        "source_dts_ns": -1, "arrival_mono_ns": 1234, "frame_num": -1,
        "buf_pts": -1, "batch_id": 1, "stage_mono_ns": -1, "objects": [],
    }
    assert recorder.submit(3, 9, frame, metadata,
                           cached_source_pts_ns=900_000_000,
                           cached_source_dts_ns=-1,
                           cached_arrival_mono_ns=1234)
    summary = recorder.close()
    row = json.loads((output / "CAM-04/frames.jsonl").read_text().strip())
    assert row["camera_id"] == "CAM-04" and row["source_id"] == 3
    assert row["input_frame_index"] == 9 and row["source_pts_ns"] == 900_000_000
    assert row["raw_bgr_sha256"] == raw_hash
    assert row["tracker_objects_raw_1920x1080"] == []
    assert summary["camera_counts"]["CAM-04"]["written"] == 1
    with pytest.raises(ValueError, match="bbox mapping is not validated"):
        from services.mv3dt_staging.live_tracking.visual_evidence import annotate_native_frame
        annotate_native_frame(
            frame, "CAM-04", [{"class_id": 0, "left": 1, "top": 1,
                               "width": 4, "height": 8, "tracker_id": 2}],
            source_size=(64, 48),
        )
