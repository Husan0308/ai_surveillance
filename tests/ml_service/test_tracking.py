import time
import unittest
import numpy as np
from services.ml_service.detection.schemas import Detection, CameraDetectionResult, DetectionBatchResult
from services.ml_service.tracking.appearance import AppearanceExtractor
from services.ml_service.tracking.tracker_manager import TrackerManager
from services.ml_service.tracking.schemas import TrackState

def camera(camera_id, frame_id, boxes=(), timestamp=None):
    stamp = timestamp or time.time()
    detections = tuple(Detection(tuple(box), confidence) for box, confidence in boxes)
    return CameraDetectionResult(camera_id, frame_id, stamp, stamp, detections)

CONFIG = {"tracking": {"min_confirmed_hits": 2, "track_high_thresh": .2,
    "track_low_thresh": .05, "new_track_thresh": .2, "match_thresh": .2,
    "max_lost_frames": 2, "max_lost_time_ms": 10000, "reconnect_grace_period_ms": 1000}}

class FakeAppearanceModel:
    def __init__(self): self.calls = 0; self.batch_sizes = []
    def extract_batch(self, crops):
        self.calls += 1; self.batch_sizes.append(len(crops))
        return [np.array([float(crop[0, 0, 0]), 1], np.float32) for crop in crops], {"gpu_ms": .5}

class TrackingTests(unittest.TestCase):
    def test_consecutive_and_short_miss_keep_id(self):
        manager = TrackerManager(CONFIG)
        first = manager.update(camera("CAM-01", 1, [((10, 10, 30, 50), .9)]))
        track_id = first.tracks[0].track_id
        second = manager.update(camera("CAM-01", 2, [((11, 10, 31, 50), .9)]))
        self.assertEqual(second.tracks[0].track_id, track_id)
        manager.update(camera("CAM-01", 3, []))
        recovered = manager.update(camera("CAM-01", 4, [((13, 10, 33, 50), .9)]))
        self.assertEqual(recovered.tracks[0].track_id, track_id)
        self.assertEqual(recovered.tracks[0].state, TrackState.CONFIRMED)

    def test_long_disappearance_removes_track(self):
        manager = TrackerManager(CONFIG); manager.update(camera("CAM-01", 1, [((0, 0, 20, 40), .9)]))
        manager.update(camera("CAM-01", 2, [])); manager.update(camera("CAM-01", 3, []))
        result = manager.update(camera("CAM-01", 4, []))
        self.assertEqual(result.tracks, ()); self.assertEqual(manager.metrics.snapshot()["cameras"]["CAM-01"]["removed_tracks"], 1)

    def test_camera_isolation_and_failure(self):
        manager = TrackerManager(CONFIG)
        a = manager.update(camera("CAM-01", 1, [((0, 0, 20, 40), .9)])); b = manager.update(camera("CAM-02", 1, [((0, 0, 20, 40), .9)]))
        self.assertTrue(a.tracks[0].track_id.startswith("CAM-01:")); self.assertTrue(b.tracks[0].track_id.startswith("CAM-02:"))
        manager.reset_camera("CAM-02")
        a2 = manager.update(camera("CAM-01", 2, [((1, 0, 21, 40), .9)]))
        self.assertEqual(a2.tracks[0].track_id, a.tracks[0].track_id)

    def test_empty_dynamic_crossing_overlap_and_unique_ids(self):
        manager = TrackerManager(CONFIG)
        self.assertEqual(manager.update(camera("CAM-01", 1, [])).tracks, ())
        boxes = [((0, 0, 20, 40), .9), ((30, 0, 50, 40), .9), ((10, 5, 28, 45), .8)]
        first = manager.update(camera("CAM-01", 2, boxes)); ids = [t.track_id for t in first.tracks]
        self.assertEqual(len(ids), len(set(ids)))
        crossed = manager.update(camera("CAM-01", 3, [((2, 0, 22, 40), .9), ((28, 0, 48, 40), .9), ((11, 5, 29, 45), .8)]))
        self.assertEqual(len(crossed.tracks), 3)

    def test_output_identity_preserved(self):
        result = TrackerManager(CONFIG).update(camera("CAM-07", 99, [((0, 0, 10, 20), .9)]))
        self.assertEqual((result.camera_id, result.frame_id), ("CAM-07", 99))

    def test_reconnect_grace(self):
        manager = TrackerManager(CONFIG); manager.update(camera("CAM-01", 1, [((0, 0, 10, 20), .9)]))
        manager.camera_disconnected("CAM-01", 10); self.assertTrue(manager.camera_reconnected("CAM-01", 10.5))
        self.assertEqual(len(manager._tracker("CAM-01").tracks), 1)
        manager.camera_disconnected("CAM-01", 20); self.assertFalse(manager.camera_reconnected("CAM-01", 22))
        self.assertEqual(len(manager._tracker("CAM-01").tracks), 0)

    def test_appearance_disabled_and_batched_when_enabled(self):
        off = TrackerManager(CONFIG)
        batch = DetectionBatchResult(1, time.time(), time.time(), (camera("CAM-01", 3, [((0, 0, 10, 20), .9)]), camera("CAM-02", 3, [((0, 0, 10, 20), .9)])))
        off.update_batch(batch, {"CAM-01": np.zeros((30, 20, 3), np.uint8)})
        model = FakeAppearanceModel(); cfg = {"tracking": {**CONFIG["tracking"], "appearance_enabled": True, "appearance_interval_frames": 3}}
        on = TrackerManager(cfg, AppearanceExtractor(model, "cpu", 16))
        frames = {cid: np.ones((30, 20, 3), np.uint8) for cid in ("CAM-01", "CAM-02")}
        on.update_batch(batch, frames)
        self.assertEqual(model.calls, 1); self.assertEqual(model.batch_sizes, [2])

    def test_missing_track_embedding_triggers_reid_off_interval(self):
        model=FakeAppearanceModel();cfg={"tracking":{**CONFIG["tracking"],"appearance_enabled":True,"appearance_interval_frames":30}}
        manager=TrackerManager(cfg,AppearanceExtractor(model,"cpu",16));frame=np.ones((60,40,3),np.uint8)
        batch=DetectionBatchResult(1,time.time(),time.time(),(camera("CAM-01",1,[((0,0,20,40),.9)]),))
        manager.update_batch(batch,{"CAM-01":frame})
        self.assertEqual(model.calls,1);self.assertGreater(manager.reid_batch_size,0)

if __name__ == "__main__": unittest.main()
