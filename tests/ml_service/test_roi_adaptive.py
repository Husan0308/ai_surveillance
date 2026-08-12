import time,unittest
import numpy as np
from services.ml_service.cameras.frame import FramePacket
from services.ml_service.detection.roi import ROIRecoveryScheduler

ROI={"id":"far","enabled":True,"polygon":[[.5,0],[1,0],[1,1],[.5,1]]}
def packet(camera_id="CAM-05",stamp=None):
 stamp=time.time() if stamp is None else stamp;frame=np.zeros((100,200,3),np.uint8);return FramePacket(camera_id,1,stamp,stamp,frame,200,100)

class AdaptiveROITests(unittest.TestCase):
 def test_pressure_stretches_discovery_but_preserves_urgent(self):
  scheduler=ROIRecoveryScheduler(1500,750,500);scheduler.configure([{"id":"CAM-05","recovery_rois":[ROI]}]);scheduler.update_pressure(350,6,6,0,95);snapshot=scheduler.snapshot()
  self.assertTrue(snapshot["overloaded"]);self.assertEqual(snapshot["effective_discovery_interval_ms"],5000);self.assertEqual(snapshot["effective_urgent_interval_ms"],3000)
  self.assertIsNone(scheduler.select((packet(),),()))
  self.assertEqual(scheduler.snapshot()["roi_skipped_pressure"],1)
  scheduler.update_pressure(80,3,6,0,30);snapshot=scheduler.snapshot();self.assertFalse(snapshot["overloaded"]);self.assertEqual(snapshot["effective_discovery_interval_ms"],1500);self.assertEqual(snapshot["effective_urgent_interval_ms"],750)
 def test_oldest_due_roi_wins_and_requests_are_coalesced(self):
  scheduler=ROIRecoveryScheduler(500,500,500);scheduler.configure([{"id":"CAM-05","recovery_rois":[ROI]},{"id":"CAM-06","recovery_rois":[ROI]}]);now=time.time();a,b=packet("CAM-05",now),packet("CAM-06",now)
  first=scheduler.select((a,b),(),now);self.assertEqual(first[0].camera_id,"CAM-05");scheduler._last[("CAM-06","far")]=now-10;scheduler._last[("CAM-05","far")]=now-1
  second=scheduler.select((packet("CAM-05",now+.6),packet("CAM-06",now+.6)),(),now+.6);self.assertEqual(second[0].camera_id,"CAM-06");self.assertGreaterEqual(scheduler.snapshot()["roi_coalesced"],1);self.assertFalse(hasattr(scheduler,"queue"))
 def test_stale_latest_request_is_discarded(self):
  scheduler=ROIRecoveryScheduler(500,500,100);scheduler.configure([{"id":"CAM-05","recovery_rois":[ROI]}]);self.assertIsNone(scheduler.select((packet(stamp=time.time()-1),),()));self.assertEqual(scheduler.snapshot()["roi_stale_drops"],1)

if __name__=="__main__":unittest.main()
