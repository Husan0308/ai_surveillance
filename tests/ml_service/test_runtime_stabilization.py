import time,unittest
from services.ml_service.detection.schemas import Detection,CameraDetectionResult
from services.ml_service.tracking.camera_tracker import CameraTracker
from services.ml_service.pipeline.gpu_coordinator import GPUInferenceCoordinator

def result(frame,boxes):
 stamp=time.time();return CameraDetectionResult("CAM-01",frame,stamp,stamp,tuple(Detection(box,confidence) for box,confidence in boxes))

class RuntimeStabilizationTests(unittest.TestCase):
 def test_duplicate_observation_cannot_spawn_fragment_beside_matched_track(self):
  tracker=CameraTracker("CAM-01",{"min_confirmed_hits":2,"track_high_thresh":.2,"track_low_thresh":.05,"new_track_thresh":.2,"match_thresh":.2,"new_track_min_width":1,"new_track_min_height":1,"new_track_min_area":1,"max_lost_time_ms":5000})
  tracker.update(result(1,[((10,10,50,90),.9)]),now_monotonic=10);confirmed=tracker.update(result(2,[((11,10,51,90),.9)]),now_monotonic=10.1);track_id=confirmed.tracks[0].track_id
  output=tracker.update(result(3,[((12,10,52,90),.9),((14,12,54,92),.8)]),now_monotonic=10.2)
  self.assertEqual([track.track_id for track in output.tracks],[track_id]);self.assertEqual(tracker.metrics.new_tracks,1);self.assertEqual(tracker.metrics.duplicate_new_track_suppressed,1)
 def test_scale_shifted_duplicate_is_suppressed_but_nearby_people_remain(self):
  tracker=CameraTracker("CAM-01",{"min_confirmed_hits":2,"track_high_thresh":.2,"track_low_thresh":.05,"new_track_thresh":.2,"match_thresh":.2,"new_track_min_width":1,"new_track_min_height":1,"new_track_min_area":1})
  tracker.update(result(1,[((10,10,50,90),.9),((58,10,98,90),.9)]),now_monotonic=10)
  tracker.update(result(2,[((11,10,51,90),.9),((59,10,99,90),.9)]),now_monotonic=10.1)
  output=tracker.update(result(3,[((12,10,52,90),.9),((14,12,58,94),.8),((60,10,100,90),.9)]),now_monotonic=10.2)
  self.assertEqual(len(output.tracks),2);self.assertEqual(tracker.metrics.new_tracks,2);self.assertEqual(tracker.metrics.duplicate_new_track_suppressed,1)
 def test_stale_predicted_fragment_is_retired_beside_real_match(self):
  tracker=CameraTracker("CAM-01",{"min_confirmed_hits":2,"track_high_thresh":.2,"track_low_thresh":.05,"new_track_thresh":.2,"match_thresh":.2,"new_track_min_width":1,"new_track_min_height":1,"new_track_min_area":1,"max_lost_time_ms":5000})
  tracker.update(result(1,[((10,10,50,90),.9),((100,10,140,90),.9)]),now_monotonic=10)
  tracker.update(result(2,[((11,10,51,90),.9),((101,10,141,90),.9)]),now_monotonic=10.1)
  stale=tracker.tracks[1];stale.bbox=(12,10,52,90);stale.motion.bbox=stale.bbox;stale.motion.state[:4]=stale.motion._measurement(stale.bbox)
  output=tracker.update(result(3,[((12,10,52,90),.9)]),now_monotonic=10.2)
  self.assertEqual(len(output.tracks),1);self.assertEqual(sum(track.state.value=="REMOVED" for track in tracker.tracks),1);self.assertEqual(tracker.metrics.removal_reason,"overlapping_stale_fragment_reconciled")
 def test_primary_waiter_has_priority_over_next_secondary(self):
  gate=GPUInferenceCoordinator();order=[]
  import threading
  entered=threading.Event();release=threading.Event()
  def secondary_one():
   with gate.secondary():order.append("s1");entered.set();release.wait(1)
  def primary():
   entered.wait(1)
   with gate.primary():order.append("p")
  def secondary_two():
   entered.wait(1)
   with gate.secondary():order.append("s2")
  threads=[threading.Thread(target=secondary_one),threading.Thread(target=primary),threading.Thread(target=secondary_two)]
  [thread.start() for thread in threads];entered.wait(1);time.sleep(.02);release.set();[thread.join(1) for thread in threads]
  self.assertEqual(order,["s1","p","s2"]);self.assertEqual(gate.snapshot()["max_active"],1)

 def test_velocity_uses_real_source_time_and_backtest_is_recorded(self):
  tracker=CameraTracker("CAM-01",{"min_confirmed_hits":1,"track_high_thresh":.2,"new_track_thresh":.2,"match_thresh":.1,"new_track_min_width":1,"new_track_min_height":1,"new_track_min_area":1})
  first=CameraDetectionResult("CAM-01",1,100.0,100.0,(Detection((0,0,20,40),.9),),10.0,640,360);second=CameraDetectionResult("CAM-01",2,100.2,100.2,(Detection((10,0,30,40),.9),),10.2,640,360)
  tracker.update(first);tracker.update(second);track=tracker.tracks[0]
  self.assertAlmostEqual(float(track.last_observed_velocity[0]),50.0,places=3);self.assertAlmostEqual(track.last_real_observation_monotonic,10.2)
  self.assertEqual(tracker.metrics.prediction_backtest_count,1);self.assertIn("150-300",tracker.metrics.prediction_horizon_buckets);self.assertGreater(tracker.metrics.prediction_center_error_norm_p50,0)

 def test_boundary_exit_hides_visual_but_retains_internal_track(self):
  tracker=CameraTracker("CAM-01",{"min_confirmed_hits":1,"track_high_thresh":.2,"new_track_thresh":.2,"match_thresh":.1,"new_track_min_width":1,"new_track_min_height":1,"new_track_min_area":1,"max_lost_time_ms":1800,"min_expiry_misses":1})
  observed=CameraDetectionResult("CAM-01",1,100.0,100.0,(Detection((590,100,638,220),.9),),10.0,640,360);tracker.update(observed)
  track=tracker.tracks[0];track.state=__import__("services.ml_service.tracking.schemas",fromlist=["TrackState"]).TrackState.CONFIRMED;track.visual_velocity[:2]=(160,0);track.motion.state[4:6]=(160,0)
  missed=CameraDetectionResult("CAM-01",2,100.1,100.1,(),10.1,640,360);tracker.update(missed)
  output=tracker.predict_visual(3,100.65,100.65,10.65);self.assertFalse(output.tracks[0].visual_visible)
  self.assertNotEqual(track.state.value,"REMOVED");self.assertEqual(tracker.metrics.boundary_exit_visual_hides_total,1)

 def test_near_border_inward_and_parallel_motion_do_not_hide(self):
  tracker=CameraTracker("CAM-01",{"min_confirmed_hits":1});track=__import__("services.ml_service.tracking.track",fromlist=["Track"]).Track("CAM-01",1,(0,100,40,220),.9,1,100,100,1,source_width=640,source_height=360);track.state=__import__("services.ml_service.tracking.schemas",fromlist=["TrackState"]).TrackState.LOST;track.misses=1;track.last_real_observation_monotonic=10
  track.visual_velocity[:2]=(100,0);self.assertEqual(track.evaluate_boundary_exit(10.8)[:2],(False,False));track.visual_velocity[:2]=(0,100);self.assertEqual(track.evaluate_boundary_exit(10.8)[:2],(False,False))

 def test_all_four_outward_boundaries_hide_from_unclipped_geometry(self):
  from services.ml_service.tracking.track import Track
  from services.ml_service.tracking.schemas import TrackState
  cases=(((0,100,40,220),(-160,0)),((600,100,640,220),(160,0)),((100,0,180,40),(0,-160)),((100,320,180,360),(0,160)))
  for local_id,(bbox,velocity) in enumerate(cases,1):
   with self.subTest(bbox=bbox,velocity=velocity):
    track=Track("CAM-01",local_id,bbox,.9,1,100,100,1,source_width=640,source_height=360);track.state=TrackState.LOST;track.misses=1;track.last_real_observation_monotonic=track.motion_monotonic=10.0;track.visual_velocity[:2]=velocity;track.motion.state[4:6]=velocity
    predicted=track.predict_visual(10.4,damping_start_ms=10_000,horizon_ms=10_001)
    self.assertTrue(predicted[0]<0 or predicted[1]<0 or predicted[2]>640 or predicted[3]>360)
    candidate,hidden,delay_ms,ratio=track.evaluate_boundary_exit(10.1)
    self.assertTrue(candidate);self.assertFalse(hidden);self.assertEqual(delay_ms,0);self.assertLess(ratio,1.0)
    candidate,hidden,delay_ms,ratio=track.evaluate_boundary_exit(10.4)
    self.assertTrue(candidate);self.assertTrue(hidden);self.assertGreaterEqual(delay_ms,250);self.assertLessEqual(delay_ms,600);self.assertLessEqual(ratio,.55)

 def test_boundary_exit_delay_starts_at_reliable_evidence_not_last_detection(self):
  from services.ml_service.tracking.track import Track
  from services.ml_service.tracking.schemas import TrackState
  track=Track("CAM-01",1,(600,100,640,220),.9,1,100,100,1,source_width=640,source_height=360);track.state=TrackState.LOST;track.misses=1;track.last_real_observation_monotonic=track.motion_monotonic=5.0;track.visual_velocity[:2]=(160,0);track.motion.state[4:6]=(160,0)
  self.assertEqual(track.evaluate_boundary_exit(10.0)[2],0.0)
  candidate,hidden,delay_ms,_=track.evaluate_boundary_exit(10.3)
  self.assertTrue(candidate);self.assertTrue(hidden);self.assertAlmostEqual(delay_ms,300.0)

 def test_visual_motion_reacts_to_start_stop_and_reversal(self):
  from services.ml_service.tracking.track import Track
  track=Track("CAM-01",1,(0,0,20,40),.9,1,100,100,1);track.motion_monotonic=track.last_detection_monotonic=track.last_real_observation_monotonic=10.0;track.real_observations.clear();track.real_observations.append((10.0,track.motion._measurement(track.bbox)))
  track.update((20,0,40,40),.9,2,100.2,now_monotonic=10.2);walking=track.velocity[0]
  self.assertGreater(walking,50)
  track.update((20,0,40,40),.9,3,100.4,now_monotonic=10.4);stopped=track.velocity[0]
  self.assertLess(abs(stopped),abs(walking)*.35)
  track.update((0,0,20,40),.9,4,100.6,now_monotonic=10.6)
  self.assertLess(track.velocity[0],0)

 def test_appearance_cannot_rescue_geometrically_impossible_match(self):
  import numpy as np
  tracker=CameraTracker("CAM-04",{"min_confirmed_hits":1,"track_high_thresh":.2,"track_low_thresh":.05,"new_track_thresh":.2,"match_thresh":.2,"new_track_min_width":1,"new_track_min_height":1,"new_track_min_area":1,"association_appearance_weight":.25})
  embedding=np.array([1.0,0.0],np.float32)
  tracker.update(result(1,[((0,0,20,40),.9)]),[embedding],10.0)
  confirmed=tracker.update(result(2,[((1,0,21,40),.9)]),[embedding],10.1);old_id=confirmed.tracks[0].track_id
  output=tracker.update(result(3,[((500,0,520,40),.9)]),[embedding],10.2)
  old=next(track for track in output.tracks if track.track_id==old_id)
  self.assertEqual(old.state.value,"LOST")
  rejected=[item for item in tracker.metrics.association_candidates if item["frame_id"]==3 and item["track_id"]==old_id]
  self.assertTrue(rejected);self.assertFalse(rejected[-1]["geometry_passed"]);self.assertFalse(rejected[-1]["selected"])

 def test_equal_crossing_candidates_abstain_instead_of_forcing_match(self):
  tracker=CameraTracker("CAM-04",{"min_confirmed_hits":1,"track_high_thresh":.2,"track_low_thresh":.05,"new_track_thresh":.2,"match_thresh":.15,"new_track_min_width":1,"new_track_min_height":1,"new_track_min_area":1,"association_ambiguity_margin":.08})
  tracker.update(result(1,[((0,0,20,40),.9),((40,0,60,40),.9)]),now_monotonic=10.0)
  tracker.update(result(2,[((0,0,20,40),.9),((40,0,60,40),.9)]),now_monotonic=10.1)
  output=tracker.update(result(3,[((20,0,40,40),.9)]),now_monotonic=10.2)
  old=[track for track in output.tracks if track.local_track_id in (1,2)]
  self.assertEqual([track.state.value for track in old],["LOST","LOST"])
  self.assertGreaterEqual(tracker.metrics.association_ambiguity_abstentions,1)
  self.assertTrue(any(item["abstained"] for item in tracker.metrics.association_candidates if item["frame_id"]==3))

if __name__=="__main__":unittest.main()
