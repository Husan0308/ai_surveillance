import asyncio,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock
import cv2,numpy as np
from services.api_service.database import SQLiteDatabase
from services.api_service.repositories.domain import EnrollmentRepository
from services.api_service.services.domain import CameraService,PersonService
from services.ml_service.face.gallery import KnownPersonGallery,SQLiteGalleryRepository
from services.ml_service.face.image_enrollment import validate_enrollment_image
from shared.topology import validate_topology,TopologyValidationError

class P3PersistenceTests(unittest.IsolatedAsyncioTestCase):
 async def asyncSetUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.db=SQLiteDatabase(Path(self.tmp.name)/"app.db");self.assertTrue(await self.db.connect())
 async def asyncTearDown(self):self.tmp.cleanup()
 async def test_enrollment_survives_gallery_restart(self):
  vector=np.zeros(512,np.float32);vector[3]=1
  await EnrollmentRepository(self.db).create({"id":"session-1","name":"Husan","status":"started"})
  await EnrollmentRepository(self.db).complete({"session_id":"session-1","person_id":"person-1","name":"Husan","dimension":512,"model_version":"buffalo_l:w600k_r50","embeddings":[{"embedding":vector.tolist(),"quality":.9,"source_metadata":{"filename":"face.jpg"}}]})
  first=KnownPersonGallery(SQLiteGalleryRepository(self.db.path));second=KnownPersonGallery(SQLiteGalleryRepository(self.db.path))
  self.assertEqual(first.enabled()[0].person_id,"person-1");self.assertTrue(np.allclose(first.enabled()[0].embeddings[0],second.enabled()[0].embeddings[0]))
 async def test_person_delete_removes_embeddings_and_notifies_ml(self):
  vector=np.zeros(512,np.float32);vector[0]=1;ml=Mock();ml.command=Mock(side_effect=lambda *_:asyncio.sleep(0))
  await EnrollmentRepository(self.db).create({"id":"s","status":"started"});await EnrollmentRepository(self.db).complete({"session_id":"s","person_id":"p","name":"P","embeddings":[{"embedding":vector.tolist()}]})
  await PersonService(self.db,ml).delete("p")
  self.assertEqual(KnownPersonGallery(SQLiteGalleryRepository(self.db.path)).enabled(),[]);ml.command.assert_called_once()
 async def test_camera_credentials_are_not_public(self):
  public=CameraService._public({"id":"C","username":"u","password":"secret","source":"rtsp://u:secret@host/live"})
  self.assertNotIn("password",public);self.assertNotIn("username",public);self.assertEqual(public["source"],"rtsp://u:***@host/live")

class P3TopologyTests(unittest.TestCase):
 def test_valid_canonical_topology(self):
  records={"A":{"camera_id":"A","room_id":"R","overlapping_camera_ids":["B"],"adjacent_camera_ids":[],"physically_separate_camera_ids":[]},"B":{"camera_id":"B","room_id":"R","overlapping_camera_ids":["A"],"adjacent_camera_ids":[],"physically_separate_camera_ids":[]}}
  self.assertEqual(len(validate_topology(["A","B"],records)),2)
 def test_rejects_self_unknown_and_asymmetric_overlap(self):
  with self.assertRaises(TopologyValidationError):validate_topology(["A"],{"A":{"camera_id":"A","room_id":"R","overlapping_camera_ids":["A"],"adjacent_camera_ids":[],"physically_separate_camera_ids":[]}})
  with self.assertRaises(TopologyValidationError):validate_topology(["A","B"],{"A":{"camera_id":"A","room_id":"R","overlapping_camera_ids":["B"],"adjacent_camera_ids":[],"physically_separate_camera_ids":[]},"B":{"camera_id":"B","room_id":"R","overlapping_camera_ids":[],"adjacent_camera_ids":[],"physically_separate_camera_ids":[]}})

class P3ImageValidationTests(unittest.TestCase):
 def test_corrupt_and_no_face_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   corrupt=Path(tmp)/"bad.jpg";corrupt.write_bytes(b"bad")
   self.assertEqual(validate_enrollment_image(corrupt,Mock(),Mock())["reason"],"invalid_or_corrupt_image")
   valid=Path(tmp)/"plain.jpg";cv2.imwrite(str(valid),np.full((100,100,3),127,np.uint8));engine=Mock();engine.detect.return_value=[]
   self.assertEqual(validate_enrollment_image(valid,engine,Mock())["reason"],"no_face")

if __name__=="__main__":unittest.main()
