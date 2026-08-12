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
    "max_lost_frames": 2, "max_lost_time_ms": 10000, "reconnect_grace_period_ms": 1000,
    "new_track_min_width":1,"new_track_min_height":1,"new_track_min_area":1}}

class FakeAppearanceModel:
    def __init__(self): self.calls = 0; self.batch_sizes = []
    def extract_batch(self, crops):
        self.calls += 1; self.batch_sizes.append(len(crops))
        return [np.array([float(crop[0, 0, 0]), 1], np.float32) for crop in crops], {"gpu_ms": .5}

class TrackingTests(unittest.TestCase):
    def test_visual_prediction_survives_five_gaps_and_reacquires_same_id(self):
        manager=TrackerManager(CONFIG);first=manager.update(camera("CAM-01",1,[((10,10,30,50),.9)]),now_monotonic=100.0)
        confirmed=manager.update(camera("CAM-01",2,[((12,10,32,50),.9)]),now_monotonic=100.1);track_id=confirmed.tracks[0].track_id
        predicted=[]
        for index in range(3,8):
            manager.update(camera("CAM-01",index,[]),now_monotonic=100.0+index*.1)
            visual=manager.predict_visual("CAM-01",index,100.0+index*.1,100.0+index*.1,100.0+index*.1)
            self.assertEqual(len(visual.tracks),1);self.assertEqual(visual.tracks[0].track_id,track_id);self.assertEqual(visual.tracks[0].observation_type,"predicted");predicted.append(visual.tracks[0].bbox)
        recovered=manager.update(camera("CAM-01",8,[((20,10,40,50),.9)]),now_monotonic=100.65)
        self.assertEqual(len(recovered.tracks),1);self.assertEqual(recovered.tracks[0].track_id,track_id)
        self.assertEqual(manager.metrics.snapshot()["cameras"]["CAM-01"]["new_tracks"],1)

    def test_strong_iou_tombstone_recovers_before_new_track(self):
        cfg={"tracking":{**CONFIG["tracking"],"max_lost_time_ms":500,"tombstone_recovery_ms":3000,"tombstone_recovery_iou":.75,"min_expiry_misses":1}};manager=TrackerManager(cfg)
        manager.update(camera("CAM-02",1,[((10,10,40,70),.9)]),now_monotonic=10.0)
        old=manager.update(camera("CAM-02",2,[((12,10,42,70),.9)]),now_monotonic=10.1).tracks[0].track_id
        manager.update(camera("CAM-02",3,[]),now_monotonic=10.7)
        recovered=manager.update(camera("CAM-02",4,[((13,10,43,70),.9)]),now_monotonic=10.8)
        self.assertEqual(recovered.tracks[0].track_id,old);metrics=manager.metrics.snapshot()["cameras"]["CAM-02"]
        self.assertEqual(metrics["new_tracks"],1);self.assertEqual(metrics["tombstone_recoveries"],1)

    def test_extended_tombstone_requires_stricter_overlap(self):
        cfg={"tracking":{**CONFIG["tracking"],"max_lost_time_ms":500,"tombstone_recovery_ms":1000,"tombstone_extended_ms":5000,"tombstone_recovery_iou":.70,"tombstone_extended_iou":.90,"min_expiry_misses":1}}
        strong=TrackerManager(cfg);strong.update(camera("CAM-02",1,[((10,10,40,70),.9)]),now_monotonic=10.0);old=strong.update(camera("CAM-02",2,[((10,10,40,70),.9)]),now_monotonic=10.1).tracks[0].track_id;strong.update(camera("CAM-02",3,[]),now_monotonic=10.7)
        self.assertEqual(strong.update(camera("CAM-02",4,[((11,10,41,70),.9)]),now_monotonic=12.5).tracks[0].track_id,old)
        weak=TrackerManager(cfg);weak.update(camera("CAM-02",1,[((10,10,40,70),.9)]),now_monotonic=20.0);old=weak.update(camera("CAM-02",2,[((10,10,40,70),.9)]),now_monotonic=20.1).tracks[0].track_id;weak.update(camera("CAM-02",3,[]),now_monotonic=20.7)
        self.assertNotEqual(weak.update(camera("CAM-02",4,[((18,10,48,70),.9)]),now_monotonic=22.5).tracks[0].track_id,old)

    def test_suspected_split_is_deferred_for_one_recovery_cycle(self):
        manager=TrackerManager(CONFIG);tracker=manager._tracker("CAM-05")
        manager.update(camera("CAM-05",1,[((10,10,40,70),.9)]),now_monotonic=10.0)
        manager.update(camera("CAM-05",2,[((10,10,40,70),.9)]),now_monotonic=10.1)
        # Reproduce the admission edge independently of model appearance values.
        tracker.match=.99;tracker.relaxed_match=.95
        manager.update(camera("CAM-05",3,[((11,10,41,70),.9)]),now_monotonic=10.2)
        metrics=manager.metrics.snapshot()["cameras"]["CAM-05"]
        self.assertEqual(len(tracker.tracks),1)
        self.assertEqual(metrics["deferred_new_track_admissions"],1)
        self.assertEqual(metrics["local_track_fragments"],0)

    def test_new_track_after_timeout_emits_bounded_fragment_audit(self):
        cfg={"tracking":{**CONFIG["tracking"],"max_lost_time_ms":500,"tombstone_recovery_ms":0,"min_expiry_misses":1}};manager=TrackerManager(cfg)
        manager.update(camera("CAM-03",1,[((10,10,30,50),.9)]),now_monotonic=10.0)
        old=manager.update(camera("CAM-03",2,[((12,10,32,50),.9)]),now_monotonic=10.1).tracks[0].track_id
        manager.update(camera("CAM-03",3,[]),now_monotonic=10.7)
        new=manager.update(camera("CAM-03",4,[((14,10,34,50),.9)]),now_monotonic=10.8).tracks[0].track_id
        event=manager.metrics.snapshot()["cameras"]["CAM-03"]["fragment_events"][-1]
        self.assertEqual((event["old_local_track"],event["new_local_track"]),(old,new));self.assertEqual(event["reason"],"new_track_after_lost_timeout")
        self.assertIn("normalized_center_distance",event);self.assertIn("old_predicted_bbox",event)

    def test_prediction_covers_measured_latency_gaps_then_expires(self):
        for gap_ms in (300,500,700,900):
            manager=TrackerManager({"tracking":{**CONFIG["tracking"],"prediction_horizon_ms":1000,"max_lost_time_ms":1800}})
            manager.update(camera("CAM-01",1,[((10,10,30,50),.9)]),now_monotonic=10.0);manager.update(camera("CAM-01",2,[((11,10,31,50),.9)]),now_monotonic=10.1)
            predicted=manager.predict_visual("CAM-01",3,10.1+gap_ms/1000,10.1+gap_ms/1000,10.1+gap_ms/1000)
            self.assertEqual(len(predicted.tracks),1,msg=f"gap={gap_ms}ms")
        self.assertEqual(manager.predict_visual("CAM-01",4,12.0,12.0,12.0).tracks,())
        manager.update(camera("CAM-01",5,[]),now_monotonic=12.0);manager.update(camera("CAM-01",6,[]),now_monotonic=12.1)
        removed=manager.update(camera("CAM-01",7,[]),now_monotonic=12.2);self.assertEqual(removed.tracks,())
        self.assertEqual(manager.predict_visual("CAM-01",6,12.1,12.1,12.1).tracks,())


    def test_overlapping_duplicate_detections_cannot_create_duplicate_tracks(self):
        manager=TrackerManager(CONFIG)
        result=manager.update(camera("CAM-04",1,[((10,10,50,90),.9),((12,11,52,91),.85)]),now_monotonic=10.0)
        self.assertEqual(len(result.tracks),1)
        metrics=manager.metrics.snapshot()["cameras"]["CAM-04"]
        self.assertEqual(metrics["new_tracks"],1);self.assertEqual(metrics["duplicate_new_track_suppressed"],1)

    def test_required_irregular_observations_and_source_starvation_have_no_visual_gap(self):
        cfg={"tracking":{**CONFIG["tracking"],"prediction_horizon_ms":1000,"max_lost_time_ms":1800}}
        manager=TrackerManager(cfg);base=100.0;track_id=None
        observations=((0,10),(500,12),(1200,15),(1800,18),(2600,22))
        for frame_id,(offset_ms,x1) in enumerate(observations,1):
            now=base+offset_ms/1000
            result=manager.update(camera("CAM-01",frame_id,[((x1,10,x1+20,50),.9)]),now_monotonic=now)
            if frame_id>=2:
                track_id=track_id or result.tracks[0].track_id;self.assertEqual(result.tracks[0].track_id,track_id)
            if frame_id>1:
                midpoint=now+.4;visual=manager.predict_visual("CAM-01",100+frame_id,midpoint,midpoint,midpoint)
                self.assertEqual([track.track_id for track in visual.tracks],[track_id])
        starvation_end=base+3.6
        visual=manager.predict_visual("CAM-01",200,starvation_end,starvation_end,starvation_end)
        self.assertEqual([track.track_id for track in visual.tracks],[track_id])
        recovered=manager.update(camera("CAM-01",201,[((24,10,44,50),.9)]),now_monotonic=starvation_end)
        self.assertEqual(recovered.tracks[0].track_id,track_id)
        manager.update(camera("CAM-01",202,[]),now_monotonic=starvation_end+1.81);manager.update(camera("CAM-01",203,[]),now_monotonic=starvation_end+1.91)
        departed=manager.update(camera("CAM-01",204,[]),now_monotonic=starvation_end+2.01);self.assertEqual(departed.tracks,())
        metrics=manager.metrics.snapshot()["cameras"]["CAM-01"]
        self.assertEqual((metrics["new_tracks"],metrics["visual_gap_count"]),(1,0));self.assertEqual(metrics["visual_track_removed"],1)

    def test_visual_velocity_is_damped_then_falls_back_to_conservative_motion(self):
        manager=TrackerManager(CONFIG);manager.update(camera("CAM-01",1,[((10,10,30,50),.9)]),now_monotonic=10.0);manager.update(camera("CAM-01",2,[((20,10,40,50),.9)]),now_monotonic=10.1)
        track=manager._tracker("CAM-01").tracks[0];early=track.predict_visual(10.2,200,650);late=track.predict_visual(10.9,200,650);legacy=track.predict(10.9)
        self.assertGreater(early[0],track.bbox[0]);self.assertAlmostEqual(late[0],legacy[0],places=4)

    def test_time_based_lifecycle_is_update_rate_independent(self):
        for hz in (2,5,10):
            cfg={"tracking":{**CONFIG["tracking"],"max_lost_time_ms":1000}};manager=TrackerManager(cfg);manager.update(camera("CAM-01",1,[((10,10,30,50),.9)]),now_monotonic=20.0)
            step=1.0/hz;frame_id=2;now=20.0
            while now+step<21.0:now+=step;manager.update(camera("CAM-01",frame_id,[]),now_monotonic=now);frame_id+=1
            self.assertEqual(manager.update(camera("CAM-01",frame_id,[]),now_monotonic=21.01).tracks,())

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

    def test_long_disappearance_removes_track_by_monotonic_time(self):
        manager = TrackerManager(CONFIG); manager.update(camera("CAM-01", 1, [((0, 0, 20, 40), .9)]),now_monotonic=10.0)
        manager.update(camera("CAM-01", 2, []),now_monotonic=10.5)
        result = manager.update(camera("CAM-01", 4, []),now_monotonic=20.1)
        self.assertEqual(result.tracks, ()); self.assertEqual(manager.metrics.snapshot()["cameras"]["CAM-01"]["removed_tracks"], 1)

    def test_repeated_low_confidence_object_never_creates_or_confirms_track(self):
        manager=TrackerManager(CONFIG)
        for frame_id in range(1,11):
            result=manager.update(camera("CAM-01",frame_id,[((10,10,30,50),.10)]))
            self.assertEqual(result.tracks,())
        self.assertEqual(manager.metrics.snapshot()["cameras"]["CAM-01"]["tracks_confirmed"],0)

    def test_tentative_requires_repeated_high_confidence_evidence(self):
        manager=TrackerManager(CONFIG)
        first=manager.update(camera("CAM-01",1,[((10,10,30,50),.9)]));self.assertEqual(first.tracks[0].state,TrackState.TENTATIVE)
        low=manager.update(camera("CAM-01",2,[((10,10,30,50),.10)]));self.assertEqual(low.tracks[0].state,TrackState.TENTATIVE)
        confirmed=manager.update(camera("CAM-01",3,[((10,10,30,50),.9)]));self.assertEqual(confirmed.tracks[0].state,TrackState.CONFIRMED)

    def test_low_confidence_recovers_already_confirmed_track(self):
        manager=TrackerManager(CONFIG)
        manager.update(camera("CAM-01",1,[((10,10,30,50),.9)]));confirmed=manager.update(camera("CAM-01",2,[((11,10,31,50),.9)]));track_id=confirmed.tracks[0].track_id
        manager.update(camera("CAM-01",3,[]));recovered=manager.update(camera("CAM-01",4,[((12,10,32,50),.10)]))
        self.assertEqual(recovered.tracks[0].track_id,track_id);self.assertEqual(recovered.tracks[0].state,TrackState.CONFIRMED)

    def test_tiny_high_confidence_box_cannot_start_track(self):
        cfg={"tracking":{**CONFIG["tracking"],"new_track_min_width":12,"new_track_min_height":36,"new_track_min_area":432}}
        result=TrackerManager(cfg).update(camera("CAM-01",1,[((10,10,18,40),.9)]))
        self.assertEqual(result.tracks,())

    def test_partial_batch_does_not_advance_absent_camera(self):
        manager=TrackerManager(CONFIG);manager.update(camera("CAM-02",1,[((10,10,30,50),.9)]),now_monotonic=10.0);confirmed=manager.update(camera("CAM-02",2,[((11,10,31,50),.9)]),now_monotonic=10.1);track_id=confirmed.tracks[0].track_id
        partial=DetectionBatchResult(3,time.time(),time.time(),(camera("CAM-01",3,[((0,0,20,40),.9)]),));manager.update_batch(partial)
        track=manager._tracker("CAM-02").tracks[0];self.assertEqual(track.track_id,track_id);self.assertEqual(track.misses,0);self.assertEqual(track.last_frame_id,2)
        visual=manager.predict_visual("CAM-02",3,10.2,10.2,10.2);self.assertEqual(visual.tracks[0].track_id,track_id)

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

    def test_invalid_async_reid_evidence_is_rejected_without_breaking_recovery(self):
        manager=TrackerManager(CONFIG)
        manager.update(camera("CAM-01",1,[((10,10,30,50),.9)]),now_monotonic=10.0)
        confirmed=manager.update(camera("CAM-01",2,[((11,10,31,50),.9)]),now_monotonic=10.1);track_id=confirmed.tracks[0].track_id
        self.assertFalse(manager.set_embedding("CAM-01",track_id,None))
        self.assertFalse(manager.set_embedding("CAM-01",track_id,np.array(np.nan,np.float32)))
        track=manager._tracker("CAM-01").tracks[0];track.appearance_embedding=np.array(np.nan,np.float32)
        recovered=manager.update(camera("CAM-01",3,[((12,10,32,50),.10)]),embeddings=[np.array([1.0,0.0],np.float32)],now_monotonic=10.2)
        self.assertEqual(recovered.tracks[0].track_id,track_id)

if __name__ == "__main__": unittest.main()
