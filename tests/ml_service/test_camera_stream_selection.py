import unittest
from services.ml_service.cameras.config import _normalize
from services.ml_service.cameras.manager import CameraManager

class CameraStreamSelectionTests(unittest.TestCase):
 def test_ml_selects_ai_source_and_legacy_source_is_compatible(self):
  defaults={}
  selected=_normalize({"id":"C","source":"rtsp://host/main","ai_source":"rtsp://host/sub","display_source":"rtsp://host/main","enabled":True},defaults)
  self.assertEqual(selected["source"],"rtsp://host/sub")
  self.assertEqual(_normalize({"id":"L","source":"rtsp://host/legacy","enabled":True},defaults)["source"],"rtsp://host/legacy")
 def test_display_source_never_enters_reader_and_does_not_replace_reader(self):
  manager=CameraManager();base={"id":"C","source":"rtsp://host/main","ai_source":"rtsp://host/sub","display_source":"rtsp://host/display-1","enabled":True}
  manager.configure([base]);first=manager.buffers()["C"]
  self.assertEqual(manager._configs["C"]["source"],"rtsp://host/sub");self.assertNotIn("display_source",manager._configs["C"]);self.assertEqual(len(manager._readers),1)
  manager.configure([{**base,"display_source":"rtsp://host/display-2"}]);self.assertIs(manager.buffers()["C"],first);self.assertEqual(len(manager._readers),1)
  manager.shutdown()
 def test_changing_ai_source_replaces_only_affected_reader(self):
  manager=CameraManager();a={"id":"A","source":"rtsp://h/a-main","ai_source":"rtsp://h/a1","enabled":True};b={"id":"B","source":"rtsp://h/b-main","ai_source":"rtsp://h/b1","enabled":True}
  manager.configure([a,b]);old=manager.buffers();manager.configure([{**a,"ai_source":"rtsp://h/a2"},b]);new=manager.buffers()
  self.assertIsNot(old["A"],new["A"]);self.assertIs(old["B"],new["B"]);self.assertEqual(len(manager._readers),2);manager.shutdown()

if __name__=="__main__":unittest.main()
