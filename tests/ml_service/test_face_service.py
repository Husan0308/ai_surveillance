import time,unittest,numpy as np
from services.ml_service.face.gallery import KnownPersonGallery
from services.ml_service.face.matcher import KnownPersonMatcher
from services.ml_service.face.quality import FaceQualityScorer
from services.ml_service.face.schemas import FaceDetection,FaceDecision
from services.ml_service.face.enrollment import EnrollmentService
from services.ml_service.identity.identity_store import IdentityStore
from services.ml_service.identity.identity_resolver import IdentityResolver
from services.ml_service.identity.schemas import IdentityTrackObservation

class FaceTests(unittest.TestCase):
 def gallery(self):
  g=KnownPersonGallery();g.add("P1","Husan",[np.array([1,0]),np.array([.99,.01])]);g.add("P2","Ali",[np.array([0,1])]);return g
 def test_known_unknown_ambiguous(self):
  m=KnownPersonMatcher(self.gallery(),.6,.8,.05)
  self.assertEqual(m.match(np.array([1,0])).person_id,"P1");self.assertEqual(m.match(np.array([.3,.3])).decision,FaceDecision.AMBIGUOUS)
  self.assertEqual(m.match(np.array([-.5,-.5])).decision,FaceDecision.UNKNOWN)
 def test_low_quality_rejected(self):
  image=np.zeros((20,20,3),np.uint8);q=FaceQualityScorer(.7,30).score(image,FaceDetection((0,0,10,10),.9))
  self.assertFalse(q.accepted)
 def test_multi_evidence_preserves_global_and_propagates(self):
  store=IdentityStore();o=IdentityTrackObservation("C1",1,"T1",(0,0,1,1),.9,time.time(),np.array([1,0]),.9);identity=store.create(o)
  resolver=IdentityResolver(store,3,.95);match=KnownPersonMatcher(self.gallery(),.6,.95,.05).match(np.array([1,0]))
  for _ in range(3):result=resolver.resolve(identity.global_id,match,.8)
  self.assertEqual(result.global_id,identity.global_id);self.assertEqual(result.person_id,"P1")
  self.assertEqual(resolver.resolve(identity.global_id,match,.8).person_id,"P1")
 def test_conflict_flagged(self):
  store=IdentityStore();o=IdentityTrackObservation("C",1,"T",(0,0,1,1),.9,time.time());i=store.create(o);r=IdentityResolver(store,1,0)
  g=self.gallery();m=KnownPersonMatcher(g,.6,.7,.05);r.resolve(i.global_id,m.match(np.array([1,0])),1)
  self.assertTrue(r.resolve(i.global_id,m.match(np.array([0,1])),1).identity_conflict)
 def test_enrollment_quality_diversity_duplicate(self):
  g=KnownPersonGallery();events=[];e=EnrollmentService(g,3,2,.6,.05,.9,lambda name,payload:events.append(name));s=e.start("P1","H")
  self.assertFalse(e.add_sample(s.session_id,np.array([1,0]),.2));self.assertTrue(e.add_sample(s.session_id,np.array([1,0]),.9));self.assertFalse(e.add_sample(s.session_id,np.array([1,0]),.9));self.assertTrue(e.add_sample(s.session_id,np.array([.8,.4]),.9));self.assertTrue(e.finish(s.session_id)["ok"])
  s2=e.start("P2","Duplicate");e.add_sample(s2.session_id,np.array([1,0]),.9);e.add_sample(s2.session_id,np.array([.8,.4]),.9)
  self.assertEqual(e.finish(s2.session_id)["reason"],"potential_duplicate_person")

if __name__=="__main__":unittest.main()
