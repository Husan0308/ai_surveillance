import unittest
from pydantic import ValidationError
from services.api_service.schemas import CameraCreate,CameraUpdate
from services.api_service.services.domain import CameraService

class CameraSchemaTests(unittest.TestCase):
 def test_runtime_stream_and_decode_fields_are_preserved(self):
  value=CameraCreate(id="CAM-X",name="X",source="rtsp://camera/main",ai_source="rtsp://camera/ai",display_source="rtsp://camera/display",codec="h265",latency_ms=20,decoder_backend="nvv4l2decoder")
  self.assertEqual(value.model_dump()["ai_source"],"rtsp://camera/ai");self.assertEqual(value.model_dump()["display_source"],"rtsp://camera/display")
  self.assertEqual(CameraUpdate(codec="h264",ai_source="rtsp://camera/new-ai").ai_source,"rtsp://camera/new-ai")
 def test_codec_is_required_and_validated(self):
  with self.assertRaises(ValidationError):CameraCreate(id="CAM-X",name="X",source="rtsp://camera/live")
  with self.assertRaises(ValidationError):CameraCreate(id="CAM-X",name="X",source="rtsp://camera/live",codec="vp9")
 def test_credentials_are_masked_in_every_source_role(self):
  public=CameraService._public({"id":"C","username":"u","password":"secret","source":"rtsp://u:secret@host/main","rtsp_url":"rtsp://u:secret@host/main","ai_source":"rtsp://u:secret@host/sub","display_source":"rtsp://u:secret@host/main"})
  self.assertNotIn("username",public);self.assertNotIn("password",public)
  for key in ("source","rtsp_url","ai_source","display_source"):self.assertEqual(public[key].split("@")[0],"rtsp://u:***")

if __name__ == "__main__":unittest.main()
