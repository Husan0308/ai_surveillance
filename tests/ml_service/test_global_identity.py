import time,unittest,numpy as np
from services.ml_service.identity.global_identity_manager import GlobalIdentityManager
from services.ml_service.identity.schemas import IdentityTrackObservation,IdentityStatus

CONFIG={"camera_rooms":{"CAM-01":"R1","CAM-06":"R1","CAM-02":"R2","CAM-09":"FAR"},
 "identity":{"strong_match_threshold":.84,"match_threshold":.74,"new_identity_threshold":.55,"ambiguity_margin":.04,
 "max_embeddings_per_identity":3,"min_embedding_quality":.5,"topology":{"relationships":{"R1:FAR":"impossible_transition"}}}}
def obs(camera,track,embedding,frame=1,quality=.9,stamp=None):
    return IdentityTrackObservation(camera,frame,track,(0,0,40,120),.9,stamp or time.time(),np.asarray(embedding,np.float32),quality)

class GlobalIdentityTests(unittest.TestCase):
    def test_same_track_binding_and_changed_local_track_recovery(self):
        m=GlobalIdentityManager(CONFIG);a=m.update(obs("CAM-01","T1",[1,0])).tracks[0]
        self.assertEqual(m.update(obs("CAM-01","T1",[.99,.01],2)).tracks[0].global_id,a.global_id)
        self.assertEqual(m.update(obs("CAM-01","T57",[.99,.01],3)).tracks[0].global_id,a.global_id)

    def test_overlapping_cameras_share_global_id(self):
        m=GlobalIdentityManager(CONFIG);a=m.update(obs("CAM-01","T1",[1,0])).tracks[0]
        b=m.update(obs("CAM-06","T2",[.99,.01])).tracks[0]
        self.assertEqual(a.global_id,b.global_id);self.assertEqual(len(m.store.get(a.global_id).active_tracks),2)

    def test_different_people_and_impossible_transition_do_not_merge(self):
        m=GlobalIdentityManager(CONFIG);a=m.update(obs("CAM-01","T1",[1,0])).tracks[0]
        b=m.update(obs("CAM-02","T2",[0,1])).tracks[0];c=m.update(obs("CAM-09","T3",[1,0])).tracks[0]
        self.assertGreaterEqual(m.metrics.snapshot()["rejected_impossible_merges"],1)
        self.assertNotEqual(a.global_id,b.global_id);self.assertNotEqual(a.global_id,c.global_id)

    def test_ambiguous_candidates_do_not_force_merge(self):
        m=GlobalIdentityManager(CONFIG)
        a=m.update(obs("CAM-01","A",[1,0])).tracks[0]
        seed=obs("CAM-02","B",[0,1]);second=m.store.create(seed);m.store.bind("CAM-02","B",second.global_id)
        m._touch(second,seed,.9)
        first=m.store.get(a.global_id);first.appearance_embedding=np.array([1,.1],np.float32)
        first.appearance_embedding/=np.linalg.norm(first.appearance_embedding)
        second.active_track_seen.clear()
        m.topology.overlapping.add(frozenset(("CAM-02","CAM-06")))
        second.appearance_embedding=np.array([1,-.1],np.float32);second.appearance_embedding/=np.linalg.norm(second.appearance_embedding)
        result=m.update(obs("CAM-06","C",[1,0])).tracks[0]
        self.assertIsNone(result.global_id);self.assertEqual(result.identity_status,IdentityStatus.AMBIGUOUS)
        count=m.metrics.snapshot()["ambiguous_matches"]
        for frame in range(2,20):m.update(obs("CAM-06","C",[1,0],frame,stamp=time.time()+frame*.01))
        self.assertEqual(m.metrics.snapshot()["ambiguous_matches"],count)

    def test_short_disappearance_recovers_and_history_is_bounded(self):
        m=GlobalIdentityManager(CONFIG);first=m.update(obs("CAM-01","T1",[1,0])).tracks[0]
        for index in range(2,8):m.update(obs("CAM-01","T1",[1,index*.01],index))
        self.assertLessEqual(len(m.store.get(first.global_id).appearance_history),3)
        recovered=m.update(obs("CAM-06","NEW",[1,.01],9,stamp=time.time()+2)).tracks[0]
        self.assertEqual(recovered.global_id,first.global_id)

    def test_poor_embedding_does_not_corrupt_gallery(self):
        m=GlobalIdentityManager(CONFIG);item=m.update(obs("CAM-01","T1",[1,0])).tracks[0]
        identity=m.store.get(item.global_id);before=identity.appearance_embedding.copy()
        m.update(obs("CAM-01","T1",[0,1],2,quality=.1))
        np.testing.assert_allclose(identity.appearance_embedding,before)

    def test_camera_failure_isolated_multiple_identities(self):
        m=GlobalIdentityManager(CONFIG);a=m.update(obs("CAM-01","A",[1,0])).tracks[0];b=m.update(obs("CAM-02","B",[0,1])).tracks[0]
        m.camera_failed("CAM-02")
        self.assertEqual(m.store.binding("CAM-01","A"),a.global_id);self.assertIsNotNone(m.store.get(b.global_id))

    def test_conflict_detection_and_no_duplicate_model(self):
        m=GlobalIdentityManager(CONFIG);a=m.update(obs("CAM-01","A",[1,0])).tracks[0]
        m.store.bind("CAM-09","FORCED",a.global_id);m.update(obs("CAM-09","FORCED",[1,0]))
        self.assertEqual(m.metrics.snapshot()["identity_conflicts"],1)

    def test_simultaneous_identities(self):
        m=GlobalIdentityManager(CONFIG);ids={m.update(obs("CAM-01",f"T{i}",vector)).tracks[0].global_id for i,vector in enumerate(([1,0,0],[0,1,0],[0,0,1]))}
        self.assertEqual(len(ids),3)

if __name__=="__main__":unittest.main()
