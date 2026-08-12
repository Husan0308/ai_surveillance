import time
import unittest
from pathlib import Path
import numpy as np

from shared.topology import compile_topology, load_topology
from services.ml_service.identity.global_identity_manager import GlobalIdentityManager
from services.ml_service.identity.schemas import IdentityTrackObservation
from services.ml_service.events.frame_metadata import frame_metadata_messages
from services.ml_service.cameras.frame import FramePacket

CAMERAS=[f"CAM-0{i}" for i in range(1,7)]
ROOMS={"CAM-01":"ROOM-1","CAM-04":"ROOM-1","CAM-02":"ROOM-2","CAM-05":"ROOM-2","CAM-03":"ROOM-3","CAM-06":"ROOM-3"}

def config():
    topology=compile_topology(load_topology(Path(__file__).parents[2]/"config/topology.yaml"),CAMERAS)
    return {"identity":{"topology":topology,"strong_match_threshold":.85,"match_threshold":.82,"ambiguity_margin":.04,"required_merge_evidence":2,"min_embedding_quality":.5},"ai":{"reid":{"min_crop_width":20,"min_crop_height":45}}}

def obs(camera,track,vector,frame,stamp):
    value=np.asarray(vector,np.float32);value/=np.linalg.norm(value)
    return IdentityTrackObservation(camera,frame,track,(10,10,70,170),.9,stamp,value,.9,frame,stamp,640,360)

def fuse_people(count,left="CAM-01",right="CAM-04"):
    manager=GlobalIdentityManager(config());base=time.time();vectors=np.eye(max(count,4),dtype=np.float32);left_ids=[]
    for index in range(count):left_ids.append(manager.update(obs(left,f"L{index}",vectors[index],1,base+index*.001)).tracks[0].global_id)
    for index in range(count):manager.update(obs(right,f"R{index}",vectors[index]+.001,2,base+.1+index*.001))
    for frame in (3,4):manager.update_batch(tuple(obs(right,f"R{index}",vectors[index]+frame*.0001,frame,base+frame*.2+index*.001) for index in range(count)))
    right_ids=[manager.store.binding(right,f"R{index}") for index in range(count)]
    return manager,left_ids,right_ids

class RoomIdentityAcceptanceTests(unittest.TestCase):
    def test_topology_is_exact_verified_and_survives_reload(self):
        first=compile_topology(load_topology(Path(__file__).parents[2]/"config/topology.yaml"),CAMERAS)
        second=compile_topology(load_topology(Path(__file__).parents[2]/"config/topology.yaml"),CAMERAS)
        self.assertTrue(first["verified"]);self.assertEqual(first,second);self.assertEqual(first["camera_rooms"],ROOMS)
        for a,b in (("CAM-01","CAM-04"),("CAM-02","CAM-05"),("CAM-03","CAM-06")):self.assertEqual(first["camera_relationships"][f"{a}:{b}"],"same_room")
        self.assertTrue(all(value in ("same_room","different_room") for value in first["camera_relationships"].values()))

    def test_one_person_two_cameras_is_one_canonical(self):
        manager,left,right=fuse_people(1);self.assertEqual(left,right);metrics=manager.metrics.snapshot();self.assertEqual(metrics["active_local_tracks"],2);self.assertEqual(metrics["active_canonical_people"],1)

    def test_two_people_are_one_to_one_not_many_to_one(self):
        manager,left,right=fuse_people(2);self.assertEqual(left,right);self.assertEqual(len(set(right)),2);self.assertEqual(manager.metrics.snapshot()["active_canonical_people"],2)

    def test_four_people_are_one_to_one_not_collapsed(self):
        manager,left,right=fuse_people(4);self.assertEqual(left,right);self.assertEqual(len(set(right)),4);self.assertEqual(manager.metrics.snapshot()["active_local_tracks"],8);self.assertEqual(manager.metrics.snapshot()["active_canonical_people"],4)

    def test_all_three_verified_room_paths_accept_strong_simultaneous_evidence(self):
        for left,right in (("CAM-01","CAM-04"),("CAM-02","CAM-05"),("CAM-03","CAM-06")):
            manager,left_ids,right_ids=fuse_people(1,left,right);self.assertEqual(left_ids,right_ids);self.assertNotIn("unverified_topology_simultaneous_active_conflict",manager.metrics.snapshot()["identity_decision_reasons"])

    def test_decision_observability_exposes_distinct_top_two_and_accepted_evidence(self):
        manager,left,right=fuse_people(2);decisions=manager.decision_snapshot();ranked=[item for item in decisions.values() if item["candidate_count"]>=2 and item["top1"] is not None]
        self.assertTrue(ranked);self.assertTrue(all(item["top2"] is not None and item["top2"]!=item["top1"] for item in ranked))
        accepted=[item for item in decisions.values() if item["decision"]=="ACCEPT"];self.assertTrue(accepted);self.assertTrue(all(item["independent_evidence_count"]==manager.required_merge_evidence for item in accepted))

    def test_alias_is_atomic_transitive_and_metadata_canonicalizes_delayed_id(self):
        manager,left,right=fuse_people(1);aliases=manager.store.aliases();self.assertEqual(len(aliases),1);old,canonical=next(iter(aliases.items()));self.assertEqual(manager.store.canonicalize(old),canonical);self.assertEqual(manager.store.get(old).global_id,canonical)
        stale=manager._result(obs("CAM-04","R0",[1,0,0,0],9,time.time()),manager.store.get(canonical),.9,"test")
        from dataclasses import replace
        stale_track=replace(stale.tracks[0],global_id=old,display_name=old,identity_version=0)
        message=frame_metadata_messages((FramePacket("CAM-04",9,time.time(),time.time(),None,640,360),),(replace(stale,tracks=(stale_track,),identity_version=0),),manager.canonicalize,manager.identity_version)[0]
        self.assertEqual(message["tracks"][0]["global_id"],canonical);self.assertEqual(message["tracks"][0]["display_name"],canonical);self.assertEqual(message["identity_version"],manager.identity_version)

    def test_same_camera_people_remain_distinct_and_gallery_safe(self):
        manager=GlobalIdentityManager(config());base=time.time();a=manager.update(obs("CAM-01","A",[1,0,0,0],1,base)).tracks[0];b=manager.update(obs("CAM-01","B",[.999,.001,0,0],1,base+.01)).tracks[0]
        manager.update_batch((obs("CAM-01","B",[.998,.002,0,0],2,base+.2),obs("CAM-01","B",[.997,.003,0,0],3,base+.4)))
        self.assertNotEqual(manager.store.binding("CAM-01","B"),a.global_id);self.assertEqual(manager.store.binding("CAM-01","B"),b.global_id);self.assertGreater(manager.metrics.snapshot()["gallery_contamination_guard"]+manager.metrics.snapshot()["global_merge_rejected_same_camera"],0)

if __name__=="__main__":unittest.main()
