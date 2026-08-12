import time,unittest,numpy as np
from services.ml_service.identity.global_identity_manager import GlobalIdentityManager
from services.ml_service.identity.schemas import IdentityTrackObservation

CONFIG={"camera_rooms":{"CAM-01":"R1","CAM-06":"R1","CAM-02":"R2","CAM-09":"FAR"},
 "identity":{"strong_match_threshold":.85,"match_threshold":.82,"ambiguity_margin":.04,"required_merge_evidence":2,"max_embeddings_per_identity":3,"min_embedding_quality":.5,
 "topology":{"overlapping_camera_pairs":[["CAM-01","CAM-06"]],"relationships":{"R1:FAR":"impossible_transition"}}},
 "ai":{"reid":{"min_crop_width":20,"min_crop_height":45}}}
def obs(camera,track,embedding,frame=1,quality=.9,stamp=None,bbox=(0,0,40,120),embedding_frame_id=None,embedding_timestamp=None,source=(640,360)):
 return IdentityTrackObservation(camera,frame,track,bbox,.9,time.time() if stamp is None else stamp,None if embedding is None else np.asarray(embedding,np.float32),quality,embedding_frame_id,embedding_timestamp,source[0],source[1])
def evidence(manager,camera,track,vectors,start):
 result=None
 for index,vector in enumerate(vectors,2):result=manager.update(obs(camera,track,vector,index,stamp=start+index*.1)).tracks[0]
 return result

class GlobalIdentityTests(unittest.TestCase):
 def test_every_new_local_track_gets_unique_provisional_id(self):
  base=time.time();m=GlobalIdentityManager(CONFIG);a=m.update(obs("CAM-04","T1",[1,0],stamp=base)).tracks[0];b=m.update(obs("CAM-04","T2",[1,0],stamp=base+.1)).tracks[0]
  self.assertEqual((a.global_id,b.global_id),("UNK 1","UNK 2"));self.assertNotEqual(a.global_id,b.global_id)

 def test_active_global_and_local_counts_are_separate_and_auditable(self):
  base=time.time();m=GlobalIdentityManager(CONFIG);m.update(obs("CAM-04","T1",[1,0],stamp=base));m.update(obs("CAM-04","T2",[0,1],stamp=base+.1));metrics=m.metrics.snapshot()
  self.assertEqual(metrics["active_local_tracks"],2);self.assertEqual(metrics["global_identities_active"],2);self.assertEqual(metrics["active_local_to_global"],{"CAM-04/T1":"UNK 1","CAM-04/T2":"UNK 2"})

 def test_same_camera_simultaneous_tracks_never_merge(self):
  base=time.time();m=GlobalIdentityManager(CONFIG);a=m.update(obs("CAM-01","A",[1,0],stamp=base)).tracks[0];b=m.update(obs("CAM-01","B",[.999,.001],stamp=base+.1)).tracks[0]
  evidence(m,"CAM-01","B",([.998,.002],[.997,.003]),base+.1)
  self.assertNotEqual(m.store.binding("CAM-01","B"),a.global_id);self.assertGreaterEqual(m.metrics.snapshot()["global_merge_rejected_same_camera"],1)

 def test_unverified_simultaneous_cross_camera_never_merges(self):
  cfg={"identity":{"topology":{"verified":False},"strong_match_threshold":.85,"required_merge_evidence":2,"min_embedding_quality":.5},"ai":{"reid":{}}};base=time.time();m=GlobalIdentityManager(cfg)
  a=m.update(obs("CAM-01","A",[1,0],stamp=base)).tracks[0];b=m.update(obs("CAM-02","B",[.999,.001],stamp=base+.1)).tracks[0];evidence(m,"CAM-02","B",([.998,.002],[.997,.003]),base+.1)
  self.assertNotEqual(m.store.binding("CAM-02","B"),a.global_id);self.assertGreaterEqual(m.metrics.snapshot()["global_merge_rejected_active_conflict"],1)

 def test_verified_overlap_can_merge_with_two_independent_strong_results(self):
  base=time.time();m=GlobalIdentityManager(CONFIG);a=m.update(obs("CAM-01","A",[1,0],stamp=base)).tracks[0];provisional=m.update(obs("CAM-06","B",[.999,.001],stamp=base+.1)).tracks[0]
  merged=evidence(m,"CAM-06","B",([.998,.002],[.997,.003]),base+.1)
  self.assertEqual(merged.global_id,a.global_id);self.assertEqual(m.consume_remaps(),((provisional.global_id,a.global_id),));self.assertEqual(m.metrics.snapshot()["global_merge_accepted"],1)

 def test_same_camera_fragment_recovers_only_after_not_simultaneous(self):
  base=time.time();m=GlobalIdentityManager(CONFIG);a=m.update(obs("CAM-01","T5",[1,0],stamp=base)).tracks[0];provisional=m.update(obs("CAM-01","T9",[.999,.001],stamp=base+3)).tracks[0]
  merged=evidence(m,"CAM-01","T9",([.998,.002],[.997,.003]),base+3)
  self.assertNotEqual(provisional.global_id,a.global_id);self.assertEqual(merged.global_id,a.global_id);self.assertEqual(len(m.store.identities()),1)

 def test_unverified_sequential_cross_camera_reuse(self):
  cfg={"identity":{"topology":{"verified":False},"strong_match_threshold":.85,"required_merge_evidence":2,"min_embedding_quality":.5},"ai":{"reid":{}}};base=time.time();m=GlobalIdentityManager(cfg)
  a=m.update(obs("CAM-01","A",[1,0],stamp=base)).tracks[0];m.update(obs("CAM-03","A2",[.999,.001],stamp=base+3));merged=evidence(m,"CAM-03","A2",([.998,.002],[.997,.003]),base+3)
  self.assertEqual(merged.global_id,a.global_id)

 def test_ambiguous_top_two_does_not_merge_or_contaminate(self):
  base=time.time();m=GlobalIdentityManager(CONFIG);a=m.update(obs("CAM-01","A",[1,.1],stamp=base)).tracks[0];b=m.update(obs("CAM-02","B",[1,-.1],stamp=base)).tracks[0];m.update(obs("CAM-06","C",[1,0],stamp=base+3));before=(len(m.store.get(a.global_id).appearance_history),len(m.store.get(b.global_id).appearance_history))
  result=m.update(obs("CAM-06","C",[.999,.001],2,stamp=base+3.2)).tracks[0]
  self.assertNotIn(result.global_id,(a.global_id,b.global_id));self.assertEqual(before,(len(m.store.get(a.global_id).appearance_history),len(m.store.get(b.global_id).appearance_history)));self.assertGreaterEqual(m.metrics.snapshot()["global_merge_rejected_ambiguous"],1)

 def test_low_quality_embedding_cannot_merge_or_update_gallery(self):
  base=time.time();m=GlobalIdentityManager(CONFIG);a=m.update(obs("CAM-01","A",[1,0],stamp=base)).tracks[0];m.update(obs("CAM-02","B",[1,0],stamp=base+3,bbox=(0,0,8,20)));m.update(obs("CAM-02","B",[.999,.001],2,stamp=base+3.2,bbox=(0,0,8,20)))
  self.assertNotEqual(m.store.binding("CAM-02","B"),a.global_id);self.assertGreaterEqual(m.metrics.snapshot()["global_merge_rejected_low_quality"],1)

 def test_false_merge_cascade_does_not_contaminate_galleries(self):
  base=time.time();m=GlobalIdentityManager(CONFIG);vectors={"A":[1,0,0],"B":[.8,.6,0],"C":[.8,0,.6]};ids={}
  for index,(name,vector) in enumerate(vectors.items()):ids[name]=m.update(obs("CAM-01",name,vector,stamp=base+index*.1)).tracks[0].global_id
  for index,(name,vector) in enumerate(vectors.items()):m.update(obs("CAM-01",name,np.asarray(vector)+np.asarray([0,.001,.001]),2,stamp=base+1+index*.1))
  self.assertEqual(len(set(ids.values())),3)
  for name,global_id in ids.items():
   identity=m.store.get(global_id);self.assertTrue(identity.appearance_history)
   expected=np.asarray(vectors[name],np.float32);expected/=np.linalg.norm(expected)
   self.assertTrue(all(float(item[0]@expected)>.99 for item in identity.appearance_history))

 def test_distinct_crossing_people_do_not_swap(self):
  base=time.time();m=GlobalIdentityManager(CONFIG);a=m.update(obs("CAM-01","A",[1,0],stamp=base)).tracks[0];b=m.update(obs("CAM-01","B",[0,1],stamp=base+.1)).tracks[0]
  m.update(obs("CAM-06","A2",[.999,.001],stamp=base+3));a2=evidence(m,"CAM-06","A2",([.998,.002],[.997,.003]),base+3)
  m.update(obs("CAM-06","B2",[.001,.999],stamp=base+6));b2=evidence(m,"CAM-06","B2",([.002,.998],[.003,.997]),base+6)
  self.assertEqual(a2.global_id,a.global_id);self.assertEqual(b2.global_id,b.global_id);self.assertNotEqual(a2.global_id,b2.global_id)

 def test_gallery_history_is_bounded(self):
  base=time.time();m=GlobalIdentityManager(CONFIG);item=m.update(obs("CAM-01","A",[1,0],stamp=base)).tracks[0]
  for index in range(1,8):m.update(obs("CAM-01","A",[1,index*.001],index+1,stamp=base+index*.1))
  self.assertLessEqual(len(m.store.get(item.global_id).appearance_history),3)

 def test_human_readable_global_id_is_not_local_track(self):
  item=GlobalIdentityManager(CONFIG).update(obs("CAM-01","TRACK-99",[1,0])).tracks[0];self.assertEqual(item.global_id,"UNK 1");self.assertEqual(item.display_name,"UNK 1");self.assertNotEqual(item.global_id,item.local_track_id)

 def test_same_extraction_cannot_satisfy_independent_evidence_twice(self):
  base=time.time();m=GlobalIdentityManager(CONFIG);a=m.update(obs("CAM-01","A",[1,0],stamp=base,embedding_frame_id=1,embedding_timestamp=base)).tracks[0]
  provisional=m.update(obs("CAM-03","B",[.99,.01],stamp=base+3,embedding_frame_id=30,embedding_timestamp=base+3)).tracks[0]
  for frame,vector in ((31,[.999,.001]),(32,[.998,.002])):
   m.update(obs("CAM-03","B",vector,frame,stamp=base+3.2,embedding_frame_id=31,embedding_timestamp=base+3.1))
  self.assertEqual(m.store.binding("CAM-03","B"),provisional.global_id);self.assertNotEqual(provisional.global_id,a.global_id)
  self.assertEqual(m.metrics.snapshot()["decision_distributions"]["same_local_track"]["count"],1)

 def test_stale_reid_result_cannot_merge_or_pollute_gallery(self):
  base=time.time();m=GlobalIdentityManager(CONFIG);a=m.update(obs("CAM-01","A",[1,0],stamp=base,embedding_frame_id=1,embedding_timestamp=base)).tracks[0]
  b=m.update(obs("CAM-03","B",[.999,.001],stamp=base+5,embedding_frame_id=2,embedding_timestamp=base)).tracks[0]
  self.assertNotEqual(a.global_id,b.global_id);self.assertEqual(len(m.store.get(b.global_id).appearance_history),0);self.assertGreaterEqual(m.metrics.snapshot()["global_merge_rejected_low_quality"],0)

 def test_untrusted_gallery_is_filtered_before_top2_ranking(self):
  base=time.time();m=GlobalIdentityManager(CONFIG);a=m.update(obs("CAM-01","A",[1,0],stamp=base)).tracks[0];identity=m.store.get(a.global_id)
  identity.add_embedding([-1,0],.9,20);self.assertFalse(identity.audit_gallery(.65))
  b=m.update(obs("CAM-03","B",[.999,.001],stamp=base+3)).tracks[0];evidence(m,"CAM-03","B",([.998,.002],[.997,.003]),base+3)
  self.assertNotEqual(m.store.binding("CAM-03","B"),a.global_id);self.assertGreaterEqual(m.metrics.snapshot()["gallery_untrusted"],1)

 def test_spatially_plausible_same_camera_fragment_uses_context_margin(self):
  base=time.time();m=GlobalIdentityManager(CONFIG);a=m.update(obs("CAM-01","A",[1,0],stamp=base,bbox=(100,80,160,240))).tracks[0]
  m.update(obs("CAM-01","DISTRACTOR",[.9801,.1987],stamp=base+.1,bbox=(450,80,510,240)))
  provisional=m.update(obs("CAM-01","A2",[.9998,.02],stamp=base+3,bbox=(105,82,165,242))).tracks[0]
  merged=None
  for index,vector in enumerate(([.99955,.03],[.9992,.04]),2):
   merged=m.update(obs("CAM-01","A2",vector,index,stamp=base+3+index*.1,bbox=(106,82,166,242))).tracks[0]
  self.assertNotEqual(provisional.global_id,a.global_id);self.assertEqual(merged.global_id,a.global_id);self.assertEqual(m.metrics.snapshot()["global_reused_same_camera"],1)

 def test_same_close_scores_remain_ambiguous_cross_camera(self):
  base=time.time();m=GlobalIdentityManager(CONFIG);a=m.update(obs("CAM-01","A",[1,0],stamp=base)).tracks[0]
  m.update(obs("CAM-02","D",[.9801,.1987],stamp=base+.1));b=m.update(obs("CAM-03","B",[.9998,.02],stamp=base+3)).tracks[0]
  evidence(m,"CAM-03","B",([.99955,.03],[.9992,.04]),base+3)
  self.assertNotEqual(m.store.binding("CAM-03","B"),a.global_id);self.assertEqual(m.store.binding("CAM-03","B"),b.global_id)

 def test_camera_failure_isolated(self):
  m=GlobalIdentityManager(CONFIG);a=m.update(obs("CAM-01","A",[1,0])).tracks[0];b=m.update(obs("CAM-02","B",[0,1])).tracks[0];m.camera_failed("CAM-02");self.assertEqual(m.store.binding("CAM-01","A"),a.global_id);self.assertIsNotNone(m.store.get(b.global_id))

if __name__=="__main__":unittest.main()
