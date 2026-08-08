import time,unittest
from services.frontend.video_renderer import MetadataBuffer
class OverlaySyncTests(unittest.TestCase):
 def test_exact_frame_and_bounded_buffer(self):
  buffer=MetadataBuffer(capacity=2,max_age_ms=100)
  for frame_id in (1,2,3):buffer.put({"camera_id":"CAM-01","frame_id":frame_id,"timestamp":time.time(),"tracks":[]})
  self.assertIsNone(buffer.match("CAM-01",1));self.assertIsNotNone(buffer.match("CAM-01",2))
 def test_stale_metadata_rejected(self):
  buffer=MetadataBuffer(max_age_ms=10);buffer.put({"camera_id":"CAM-01","frame_id":1,"timestamp":time.time()-1,"tracks":[]})
  self.assertIsNone(buffer.match("CAM-01",1))
