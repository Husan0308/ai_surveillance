import time,unittest
from unittest.mock import patch
import numpy as np
from services.ml_service.cameras.display_manager import OnDemandDisplayManager,reuses_ai_reader
from services.ml_service.cameras.frame import FramePacket
from services.ml_service.control import MLRuntimeState,display_fps_cap,display_scale
class FakeReader:
 def __init__(self,config,buffer,factory,on_frame):self.config=config;self.started=False;self.stopped=False
 def start(self):self.started=True
 def stop(self):self.stopped=True
 def join(self,_timeout):return True
 def metrics(self):return {"online":self.started and not self.stopped}
class DisplaySourceTests(unittest.TestCase):
 def test_supported_display_cadence_candidates_load_without_side_effects(self):
  self.assertEqual([display_fps_cap({"fps_cap":value}) for value in (12,14,16)],[12.0,14.0,16.0])
 def test_grid_and_fullscreen_scaling_are_bounded(self):
  self.assertAlmostEqual(display_scale(3200,1800),.3);self.assertAlmostEqual(display_scale(3200,1800,True),.4);self.assertEqual(display_scale(640,360,True),1.0)
 def test_same_high_resolution_ai_source_is_reused_without_second_reader(self):
  config={"ai_source":"rtsp://host/main","display_source":"rtsp://host/main"};self.assertTrue(reuses_ai_reader(config));self.assertFalse(OnDemandDisplayManager(lambda _:None).start("CAM-01",config))

 def test_fullscreen_source_is_separate_and_does_not_enter_ai_path(self):
  config={"id":"CAM-06","ai_source":"rtsp://host/402","display_source":"rtsp://host/401","codec":"h265","display_codec":"h265"}
  with patch("services.ml_service.cameras.display_manager.CameraReader",FakeReader):
   manager=OnDemandDisplayManager(lambda _packet:None);self.assertTrue(manager.start("CAM-06",config));self.assertEqual(manager._reader.config["source"],config["display_source"]);self.assertEqual(config["ai_source"],"rtsp://host/402");manager.stop("CAM-06");self.assertIsNone(manager.camera_id)
 def test_shared_ai_source_fullscreen_never_suppresses_ai_video(self):
  runtime=MLRuntimeState();now=time.time();ai=FramePacket("CAM-01",1,now,now,np.zeros((1440,2560,3),np.uint8),2560,1440)
  runtime.set_high_quality("CAM-01",reuses_ai=True);runtime.frame(ai)
  self.assertEqual(runtime.frames["CAM-01"][0],1);self.assertTrue(runtime.high_quality_reuses_ai)
  runtime.clear_high_quality("CAM-01");self.assertFalse(runtime.high_quality_reuses_ai)

 def test_separate_display_reader_stress_is_bounded_and_balanced(self):
  config={"id":"CAM-06","ai_source":"rtsp://host/402","display_source":"rtsp://host/401","codec":"h265"}
  with patch("services.ml_service.cameras.display_manager.CameraReader",FakeReader):
   manager=OnDemandDisplayManager(lambda _packet:None)
   for _ in range(60):self.assertTrue(manager.start("CAM-06",config));self.assertEqual(manager.snapshot()["active_reader_count"],1);self.assertTrue(manager.stop("CAM-06"))
   metrics=manager.snapshot();self.assertEqual((metrics["active_reader_count"],metrics["starts"],metrics["stops"],metrics["failed_joins"]),(0,60,60,0))

 def test_runtime_accepts_only_selected_high_quality_frames(self):
  runtime=MLRuntimeState();now=time.time();ai=FramePacket("CAM-06",1,now,now,np.zeros((360,640,3),np.uint8),640,360);high=FramePacket("CAM-06",2,now,now,np.zeros((1440,2560,3),np.uint8),2560,1440)
  runtime.frame(ai);self.assertEqual(runtime.frames["CAM-06"][2].shape[:2],(360,640));runtime.set_high_quality("CAM-06");runtime.frame(ai);runtime.display_frame(high);self.assertEqual(runtime.frames["CAM-06"][2].shape[:2],(1440,2560));runtime.clear_high_quality("CAM-06");runtime.frame(ai);self.assertEqual(runtime.frames["CAM-06"][2].shape[:2],(360,640))

 def test_frame_rate_limiter_drops_excess_frames_and_publishes_latest(self):
  """The token-bucket in frame() limits MJPEG publish rate to DISPLAY_FPS cap."""
  import services.ml_service.control as ctrl_module
  old_fps=ctrl_module.DISPLAY_FPS
  try:
   ctrl_module.DISPLAY_FPS=2.0  # 2fps cap for deterministic test
   runtime=MLRuntimeState();now=time.time();frame=np.zeros((4,4,3),np.uint8)
   # Submit 5 frames in rapid succession (<<500ms apart) — only the first and
   # possibly second should pass the token bucket; the rest are dropped.
   accepted=[]
   for frame_id in range(1,6):
    runtime.frame(FramePacket("CAM-01",frame_id,now+frame_id*.001,now,frame,4,4))
    if runtime.frames.get("CAM-01",(-1,))[0]==frame_id:accepted.append(frame_id)
   self.assertLessEqual(len(accepted),3,"Token bucket should have rate-limited excess frames")
   stats=runtime.video_stats.get("CAM-01",{})
   self.assertGreaterEqual(stats.get("display_drops",0),2,"Expected at least 2 display drops")
   self.assertTrue(runtime.shutdown())
  finally:
   ctrl_module.DISPLAY_FPS=old_fps
if __name__=="__main__":unittest.main()
