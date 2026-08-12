import tempfile,time,unittest
from pathlib import Path
import cv2,numpy as np
from shared.topology import compile_topology,TopologyValidationError
from shared.enrollment_paths import stage_files,validate_staged_paths
from shared.event_taxonomy import classify,is_persistent
from services.ml_service.events.api_publisher import APIEventPublisher
from services.ml_service.control import MLRuntimeState
from services.ml_service.cameras.frame import FramePacket
from unittest.mock import patch
from services.ml_service.snapshots import UnknownSnapshotManager

class TopologyHardeningTests(unittest.TestCase):
 def test_unverified_empty_is_safe(self):
  result=compile_topology({"verified":False,"rooms":{},"overlaps":[],"adjacency":{},"travel_time":{}},["A","B"]);self.assertFalse(result["verified"]);self.assertEqual(result["camera_rooms"],{})
 def test_verified_requires_membership_and_symmetric_adjacency(self):
  with self.assertRaises(TopologyValidationError):compile_topology({"verified":True,"rooms":{"r":{"cameras":["A"]}}},["A","B"])
  with self.assertRaises(TopologyValidationError):compile_topology({"rooms":{"r1":["A"],"r2":["B"]},"adjacency":{"r1":["r2"]}},["A","B"])
 def test_duplicate_and_unknown_relations_rejected(self):
  with self.assertRaises(TopologyValidationError):compile_topology({"overlaps":[["A","B"],["B","A"]]},["A","B"])
  with self.assertRaises(TopologyValidationError):compile_topology({"overlaps":[["A","X"]]},["A","B"])

class EnrollmentSecurityTests(unittest.TestCase):
 def test_only_staged_decodable_files_are_accepted(self):
  with tempfile.TemporaryDirectory() as tmp:
   source=Path(tmp)/"face.jpg";cv2.imwrite(str(source),np.full((32,32,3),127,np.uint8));staged=stage_files([str(source)])
   self.assertEqual(validate_staged_paths(staged),staged)
   with self.assertRaises((OSError,ValueError)):validate_staged_paths(["/etc/passwd"])

class SnapshotTests(unittest.TestCase):
 def test_quality_replacement_and_identity_bound(self):
  with tempfile.TemporaryDirectory() as tmp:
   manager=UnknownSnapshotManager(tmp,max_identities=1,retention_days=1,min_improvement=.1);image=np.random.default_rng(3).integers(0,255,(100,100,3),dtype=np.uint8)
   first=manager.consider("UNK-1","A",1,time.time(),image,(0,0,80,80),quality=.5);self.assertIsNotNone(first)
   self.assertIsNone(manager.consider("UNK-1","A",2,time.time(),image,(0,0,80,80),quality=.55))
   manager.consider("UNK-2","A",3,time.time()+1,image,(0,0,80,80),quality=.8);self.assertLessEqual(len(manager.index),1);manager.close()

class EventTaxonomyTests(unittest.TestCase):
 def test_realtime_and_legacy_are_not_new_business_writes(self):
  self.assertEqual(classify("person_detected"),"legacy");self.assertFalse(is_persistent({"type":"frame.metadata"}));self.assertTrue(is_persistent({"type":"camera.online"}))

class RealtimePolicyTests(unittest.TestCase):
 def test_publisher_routes_only_business_events_to_persistent_endpoint(self):
  publisher=APIEventPublisher("http://api")
  publisher.publish({"type":"frame.metadata","camera_id":"A"});publisher.publish({"type":"camera.online","camera_id":"A"})
  realtime=publisher.queue.get_nowait();business=publisher.queue.get_nowait()
  self.assertFalse(realtime[0]);self.assertEqual(realtime[1]["type"],"frame.metadata");self.assertTrue(business[0])
 def test_display_handoff_samples_twenty_fps_near_eighteen_without_ai(self):
  runtime=MLRuntimeState();frame=np.zeros((4,4,3),np.uint8);times=[i*.05 for i in range(20)]
  with patch("services.ml_service.control.DISPLAY_FPS",18.0),patch("services.ml_service.control.time.monotonic",side_effect=times):
   for frame_id in range(20):runtime.frame(FramePacket("A",frame_id,frame_id*.05,frame_id*.05,frame,4,4))
  self.assertGreaterEqual(runtime.video_stats["A"]["display_frames"],17);self.assertLessEqual(runtime.video_stats["A"]["display_frames"],19);self.assertEqual(runtime.frames["A"][0],19)

if __name__=="__main__":unittest.main()
