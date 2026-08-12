"""Automated tests for walking detection blink fix and visual track retention."""
import time
import numpy as np

from services.ml_service.tracking.camera_tracker import CameraTracker
from services.ml_service.tracking.schemas import TrackState
from services.ml_service.tracking.disappearance_logger import disappearance_auditor
from services.ml_service.events.frame_metadata import merge_visual_identity_results
from services.frontend.ui import CameraState, RealtimeTrack, unique_overlay_payloads

class MockDetection:
    def __init__(self, bbox, confidence, detection_source="FULL_FRAME", detection_id=None):
        self.bbox_xyxy = tuple(float(v) for v in bbox)
        self.confidence = float(confidence)
        self.detection_source = detection_source
        self.detection_id = detection_id

class MockResult:
    def __init__(self, camera_id, frame_id, capture_timestamp, detections, source_width=1920, source_height=1080):
        self.camera_id = camera_id
        self.frame_id = frame_id
        self.capture_timestamp = capture_timestamp
        self.capture_monotonic = capture_timestamp
        self.receive_timestamp = capture_timestamp
        self.detections = detections
        self.source_width = source_width
        self.source_height = source_height

def test_walking_blink_prevention_over_1500ms_gap():
    tracker = CameraTracker("CAM-04", {"track_high_thresh": 0.22, "track_low_thresh": 0.05, "max_lost_time_ms": 1800, "prediction_horizon_ms": 1800})
    t0 = time.monotonic()
    
    # 1. Initial 3 detections to confirm track
    for f in range(1, 4):
        res = MockResult("CAM-04", f, t0 + f * 0.1, [MockDetection((100, 100, 200, 400), 0.75)])
        tracker.update(res)
    
    active = [t for t in tracker.tracks if t.state == TrackState.CONFIRMED]
    assert len(active) == 1
    track_id = active[0].track_id

    # 2. Simulate 12 scheduler omission cycles (1200ms gap)
    t_gap = t0 + 0.4
    for step in range(12):
        now_mono = t_gap + step * 0.1
        res = tracker.predict_visual(10 + step, now_mono, now_mono, now_mono)
        assert len(res.tracks) == 1
        assert res.tracks[0].track_id == track_id
        assert res.tracks[0].observation_type == "predicted"

    # 3. Reacquire with low-confidence detection
    res_reacquire = MockResult("CAM-04", 30, t0 + 1.7, [MockDetection((120, 100, 220, 400), 0.15)])
    track_res = tracker.update(res_reacquire)
    
    active_after = [t for t in tracker.tracks if t.state in (TrackState.CONFIRMED, TrackState.LOST)]
    assert len(active_after) == 1
    assert active_after[0].track_id == track_id
    assert active_after[0].misses == 0

def test_metadata_merger_retains_predictions_during_gaps():
    tracker = CameraTracker("CAM-05", {"max_lost_time_ms": 1800, "prediction_horizon_ms": 1800})
    t0 = time.monotonic()
    for f in range(1, 4):
        tracker.update(MockResult("CAM-05", f, t0 + f * 0.1, [MockDetection((300, 200, 400, 500), 0.8)]))

    # Generate prediction result at 800ms age (misses > 5)
    pred_res = tracker.predict_visual(20, t0 + 0.9, t0 + 0.9, t0 + 0.9)
    
    class MockTrackingResults:
        results = [pred_res]

    cache = {}
    # First pass populates cache
    live_mock = [pred_res]
    merge_visual_identity_results(MockTrackingResults(), live_mock, cache)
    
    # Second pass with no live detections
    merged = merge_visual_identity_results(MockTrackingResults(), [], cache)
    assert len(merged) == 1
    assert len(merged[0].tracks) == 1
    assert merged[0].tracks[0].observation_type == "predicted"

def test_frontend_set_metadata_does_not_block_reacquired_tracks():
    cam = CameraState("CAM-04", "Test Cam", "Room A")
    
    # Message 1 with track
    msg1 = {
        "camera_id": "CAM-04", "frame_id": 100, "timestamp": 1000.0,
        "tracks": [{"local_track_id": "CAM-04:TRACK-00001", "bbox": [100, 100, 200, 400], "confidence": 0.8, "observation_type": "detected"}]
    }
    assert cam.set_metadata(msg1)
    assert len(cam.tracks) == 1

    # Message 2 with no tracks (temporary drop)
    msg2 = {"camera_id": "CAM-04", "frame_id": 101, "timestamp": 1000.1, "tracks": []}
    assert cam.set_metadata(msg2)
    assert len(cam.tracks) == 0

    # Message 3 reacquired track on same generation
    msg3 = {
        "camera_id": "CAM-04", "frame_id": 102, "timestamp": 1000.2,
        "tracks": [{"local_track_id": "CAM-04:TRACK-00001", "bbox": [110, 100, 210, 400], "confidence": 0.7, "observation_type": "detected"}]
    }
    assert cam.set_metadata(msg3)
    assert len(cam.tracks) == 1
    assert cam.tracks[0].track_id == "CAM-04:TRACK-00001"

def test_disappearance_logging_instrumentation():
    auditor_summary_before = disappearance_auditor.summary()
    tracker = CameraTracker("CAM-04", {"max_lost_time_ms": 200, "prediction_horizon_ms": 200, "min_expiry_misses": 1})
    t0 = time.monotonic()
    tracker.update(MockResult("CAM-04", 1, t0, [MockDetection((100, 100, 200, 400), 0.8)]))
    
    # Force expiry after 300ms
    t_expire = t0 + 0.4
    for f in range(2, 5):
        tracker.update(MockResult("CAM-04", f, t_expire, []))

    summary = disappearance_auditor.summary()
    assert summary["total_disappearances"] > auditor_summary_before["total_disappearances"]
    assert "RETENTION_EXPIRED" in summary["counts_by_reason"]

def test_metadata_versioning_prevents_frontend_rewind():
    # Test that the single canonical publisher properly assigns strictly increasing
    # metadata versions, ensuring prediction threads cannot overwrite newer YOLO state
    cam = CameraState("CAM-06", "Test Cam", "Room C")
    
    # 1. Prediction publishes frame 105 (older capture_timestamp, but predicted later)
    # Actually prediction extrapolates from old state.
    msg1 = {
        "camera_id": "CAM-06", "frame_id": 105, "timestamp": 1000.5,
        "metadata_version": 1,
        "tracks": [{"local_track_id": "CAM-06:TRACK-1", "bbox": [10, 10, 20, 20], "confidence": 0.8, "observation_type": "predicted"}]
    }
    assert cam.set_metadata(msg1)
    
    # 2. YOLO finishes processing frame 102 and publisher sends it with newer metadata_version
    msg2 = {
        "camera_id": "CAM-06", "frame_id": 102, "timestamp": 1000.2,
        "metadata_version": 2,  # Monotonic version ensures it is applied despite older frame_id
        "tracks": [{"local_track_id": "CAM-06:TRACK-1", "bbox": [15, 15, 25, 25], "confidence": 0.9, "observation_type": "detected"}]
    }
    assert cam.set_metadata(msg2)
    assert cam.tracks[0].observation_type == "detected"
    assert cam.tracks[0]._bbox == (15.0, 15.0, 25.0, 25.0)
    
    # 3. An out-of-order stale message from a hypothetical race condition should be rejected
    msg3 = {
        "camera_id": "CAM-06", "frame_id": 106, "timestamp": 1000.6,
        "metadata_version": 1,  # Older version!
        "tracks": [{"local_track_id": "CAM-06:TRACK-1", "bbox": [0, 0, 0, 0], "confidence": 0.1, "observation_type": "predicted"}]
    }
    assert not cam.set_metadata(msg3)
    assert cam.tracks[0].observation_type == "detected"
