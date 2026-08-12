import time,unittest
import numpy as np
from services.ml_service.identity.global_identity_manager import GlobalIdentityManager
from services.ml_service.identity.schemas import IdentityTrackObservation
from services.ml_service.identity.worker import IdentityAssociationWorker

CONFIG={"identity":{"topology":{"verified":False},"strong_match_threshold":.85,"ambiguity_margin":.04,"required_merge_evidence":2,"min_embedding_quality":.5},"ai":{"reid":{"min_crop_width":20,"min_crop_height":45}}}

def observation(camera,track,embedding=None,frame=1,stamp=None,embedding_frame=None):
    stamp=time.time() if stamp is None else stamp
    return IdentityTrackObservation(camera,frame,track,(10,10,60,150),.9,stamp,None if embedding is None else np.asarray(embedding,np.float32),.9,embedding_frame,stamp if embedding_frame is not None else None,640,360)

def wait_for(worker,count,timeout=2):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if worker.snapshot()["processed"]>=count:return
        time.sleep(.005)
    raise AssertionError(worker.snapshot())

class IdentityWorkerTests(unittest.TestCase):
    def test_fast_binding_is_immediate_and_duplicate_version_runs_once(self):
        manager=GlobalIdentityManager(CONFIG);worker=IdentityAssociationWorker(manager);worker.start()
        try:
            first=worker.observe(observation("CAM-01","T1")).tracks[0]
            for frame in range(2,30):self.assertEqual(worker.observe(observation("CAM-01","T1",frame=frame)).tracks[0].global_id,first.global_id)
            wait_for(worker,1);metrics=worker.snapshot()
            self.assertEqual(metrics["submitted"],1);self.assertEqual(metrics["processed"],1);self.assertLess(metrics["fast_lookup"]["p95"],3)
        finally:worker.shutdown()

    def test_async_sequential_cross_camera_merge_updates_binding(self):
        base=time.time();manager=GlobalIdentityManager(CONFIG);worker=IdentityAssociationWorker(manager);worker.start()
        try:
            a=worker.observe(observation("CAM-01","A",[1,0],1,base,1)).tracks[0];wait_for(worker,1)
            provisional=worker.observe(observation("CAM-03","B",[.999,.001],2,base+3,2)).tracks[0];wait_for(worker,2)
            worker.observe(observation("CAM-03","B",[.998,.002],3,base+3.2,3));wait_for(worker,3)
            worker.observe(observation("CAM-03","B",[.997,.003],4,base+3.4,4));wait_for(worker,4)
            canonical=worker.observe(observation("CAM-03","B",frame=5,stamp=base+3.5)).tracks[0]
            self.assertNotEqual(provisional.global_id,a.global_id);self.assertEqual(canonical.global_id,a.global_id)
        finally:worker.shutdown()

    def test_queue_is_bounded_and_stale_work_cannot_corrupt(self):
        class SlowManager:
            def __init__(self):self.calls=[];self.next=0
            def lookup_or_create(self,obs):
                self.next+=1
                from services.ml_service.identity.schemas import GlobalTrack,GlobalTrackResult,IdentityStatus
                return GlobalTrackResult(obs.camera_id,obs.frame_id,(GlobalTrack(obs.local_track_id,f"UNK {self.next}",obs.bbox,.9,.1,IdentityStatus.ACTIVE,"test"),)),True
            def update(self,obs):time.sleep(.08);self.calls.append((obs.camera_id,obs.local_track_id));return None
            def consume_remaps(self):return ()
        manager=SlowManager();worker=IdentityAssociationWorker(manager,queue_size=2,max_task_age_ms=20);worker.start()
        try:
            for index in range(8):worker.observe(observation("C",str(index),frame=index))
            time.sleep(.3);metrics=worker.snapshot()
            self.assertLessEqual(metrics["queue_max"],2);self.assertGreater(metrics["dropped"]+metrics["stale"],0);self.assertEqual(metrics["errors"],0)
        finally:worker.shutdown()

if __name__=="__main__":unittest.main()
