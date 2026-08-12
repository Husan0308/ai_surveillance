import time,unittest
import numpy as np
from services.ml_service.cameras.frame import FramePacket
from services.ml_service.detection.person_detector import PersonDetector
from services.ml_service.detection.roi import (ROIRecoveryScheduler,RecoveryROI,crop_rectangle,
 fuse_detections,map_crop_bbox,point_in_polygon,source_polygon)
from services.ml_service.detection.schemas import Detection
from services.ml_service.pipeline.batch import BatchOutput
from services.ml_service.tracking.camera_tracker import CameraTracker

CONFIG={"ai":{"max_frame_age_ms":200,"detector":{"imgsz":[100,100],"min_box_width":1,"min_box_height":1,"low_conf_size_threshold":0},"roi_recovery":{"discovery_interval_ms":100,"urgent_interval_ms":50,"max_task_age_ms":500}},"tracking":{"min_confirmed_hits":3,"new_track_thresh":.28,"track_high_thresh":.22,"track_low_thresh":.05}}
ROI={"id":"far","enabled":True,"polygon":[[.5,0],[1,0],[1,1],[.5,1]]}

class SequenceBackend:
    coordinates_original=True
    def __init__(self,*calls):self.outputs=list(calls);self.calls=0;self.model=object()
    def infer(self,prepared):
        value=self.outputs[min(self.calls,len(self.outputs)-1)];self.calls+=1
        return [np.asarray(value,np.float32) for _ in prepared.batch.frames],{"gpu_inference_ms":1.0}

def packet(frame_id=1,stamp=None):
    stamp=time.time() if stamp is None else stamp;frame=np.zeros((100,200,3),np.uint8);return FramePacket("CAM-05",frame_id,stamp,stamp,frame,200,100)
def batch(frame_id=1,stamp=None):return BatchOutput(frame_id,time.time(),(packet(frame_id,stamp),))

class ROIRecoveryTests(unittest.TestCase):
    def detector(self,backend):
        detector=PersonDetector(CONFIG,backend);detector.configure_rois([{"id":"CAM-05","recovery_rois":[ROI]}]);return detector
    def test_source_crop_mapping_and_polygon_containment(self):
        roi=RecoveryROI("far",True,((.5,0),(1,0),(1,1),(.5,1)))
        self.assertEqual(crop_rectangle(roi,200,100),(100,0,200,100));self.assertEqual(map_crop_bbox((10,20,50,80),(100,0)),(110,20,150,80));self.assertTrue(point_in_polygon((130,80),source_polygon(roi,200,100)));self.assertFalse(point_in_polygon((30,80),source_polygon(roi,200,100)))
    def test_people_outside_roi_keep_full_frame_detection(self):
        backend=SequenceBackend([[10,10,40,90,.8,0]],[]);result=self.detector(backend).process_batch(batch())
        self.assertTrue(any(item.bbox_xyxy[0]<50 for item in result.results[0].detections))
    def test_roi_recovers_main_miss_and_maps_to_source(self):
        backend=SequenceBackend([],[[10,10,50,80,.7,0]]);detector=self.detector(backend);result=detector.process_batch(batch());item=result.results[0].detections[0]
        self.assertEqual(item.bbox_xyxy,(110.0,10.0,150.0,80.0));self.assertEqual(item.detection_source,"ROI_RECOVERY");self.assertEqual(backend.calls,2);self.assertEqual(detector.roi_snapshot()["roi_recovered"],1)
    def test_full_and_roi_duplicate_are_fused_once(self):
        backend=SequenceBackend([[110,10,150,80,.5,0]],[[10,10,50,80,.8,0]]);detector=self.detector(backend)
        # A predicted missed track makes the scan urgent despite main coverage.
        class T:results=()
        detector.roi._hints["CAM-05"]=({"bbox":(110,10,150,80),"confirmed":True,"misses":1},)
        result=detector.process_batch(batch());self.assertEqual(len(result.results[0].detections),1);self.assertAlmostEqual(result.results[0].detections[0].confidence,.8,places=5);self.assertEqual(detector.roi_snapshot()["roi_duplicates_suppressed"],1)
    def test_discovery_can_create_only_through_existing_confirmation_policy(self):
        backend=SequenceBackend([],[[10,10,50,80,.8,0]],[],[[10,10,50,80,.8,0]],[],[[10,10,50,80,.8,0]]);detector=self.detector(backend);tracker=CameraTracker("CAM-05",CONFIG["tracking"])
        ids=[]
        for frame in range(1,4):
            detector.roi._last.clear();result=detector.process_batch(batch(frame));tracked=tracker.update(result.results[0]);ids.extend(item.track_id for item in tracked.tracks)
        self.assertEqual(len(set(ids)),1);self.assertTrue(tracked.tracks[0].confirmed)
    def test_roi_detection_restores_same_confirmed_track(self):
        tracker=CameraTracker("CAM-05",CONFIG["tracking"]);stamp=time.time();track_id=None
        from services.ml_service.detection.schemas import CameraDetectionResult
        for frame in range(1,4):track_id=tracker.update(CameraDetectionResult("CAM-05",frame,stamp,stamp,(Detection((110,10,150,80),.8),))).tracks[0].track_id
        recovered=tracker.update(CameraDetectionResult("CAM-05",4,stamp,stamp,(Detection((111,10,151,80),.18,detection_source="ROI_RECOVERY"),)))
        self.assertEqual(recovered.tracks[0].track_id,track_id);self.assertEqual(tracker.metrics.new_tracks,1);self.assertGreater(tracker.metrics.low_confidence_recovery_matches,0)
    def test_roi_recovery_preserves_global_id_and_produces_normal_overlay_metadata(self):
        from services.ml_service.detection.schemas import CameraDetectionResult
        from services.ml_service.events import frame_metadata_messages
        from services.ml_service.identity.global_identity_manager import GlobalIdentityManager
        from services.ml_service.identity.schemas import IdentityTrackObservation
        tracker=CameraTracker("CAM-05",CONFIG["tracking"]);stamp=time.time();tracked=None
        for frame in range(1,4):tracked=tracker.update(CameraDetectionResult("CAM-05",frame,stamp,stamp,(Detection((110,10,150,80),.8),)))
        local=tracked.tracks[0].track_id;manager=GlobalIdentityManager({"identity":{},"ai":{"reid":{}}});observation=lambda frame,bbox:IdentityTrackObservation("CAM-05",frame,local,bbox,.8,stamp,None,0.0,None,None,200,100);first=manager.update(observation(3,(110,10,150,80))).tracks[0]
        recovered=tracker.update(CameraDetectionResult("CAM-05",4,stamp,stamp,(Detection((111,10,151,80),.18,detection_source="ROI_RECOVERY"),)));second=manager.update(observation(4,recovered.tracks[0].bbox)).tracks[0]
        self.assertEqual((first.global_id,second.global_id),("UNK 1","UNK 1"));message=frame_metadata_messages((packet(4),),(manager._result(observation(4,recovered.tracks[0].bbox),manager.store.get(second.global_id),.8,"roi"),))[0];self.assertEqual(message["tracks"][0]["global_id"],"UNK 1");self.assertEqual(message["tracks"][0]["bbox"],list(recovered.tracks[0].bbox))

    def test_person_can_leave_roi_under_normal_tracking(self):
        tracker=CameraTracker("CAM-05",CONFIG["tracking"]);stamp=time.time();from services.ml_service.detection.schemas import CameraDetectionResult
        boxes=((110,10,150,80),(100,10,140,80),(80,10,120,80),(60,10,100,80))
        ids=[]
        for frame,box in enumerate(boxes,1):ids.append(tracker.update(CameraDetectionResult("CAM-05",frame,stamp,stamp,(Detection(box,.8),))).tracks[0].track_id)
        self.assertEqual(len(set(ids)),1)
    def test_stale_and_bounded_scheduler(self):
        scheduler=ROIRecoveryScheduler(100,50,100);scheduler.configure([{"id":"CAM-05","recovery_rois":[ROI]},{"id":"CAM-06","recovery_rois":[{"id":"a","polygon":[[0,0],[1,0],[1,1]]},{"id":"b","polygon":[[0,0],[1,0],[0,1]]}]}])
        stale=packet(stamp=time.time()-1);self.assertIsNone(scheduler.select((stale,),()));self.assertEqual(scheduler.snapshot()["roi_stale_drops"],1)
        fresh=packet();self.assertIsNotNone(scheduler.select((fresh,),()));self.assertEqual(scheduler.snapshot()["max_roi_per_batch"],0)
    def test_no_roi_keeps_exact_single_forward(self):
        backend=SequenceBackend([[10,10,40,90,.8,0]]);detector=PersonDetector(CONFIG,backend);detector.configure_rois([{"id":"CAM-05","recovery_rois":[]}]);detector.process_batch(batch());self.assertEqual(backend.calls,1);self.assertEqual(detector.roi_snapshot()["roi_inferences"],0)

class FusionTests(unittest.TestCase):
    def test_separate_people_are_not_suppressed(self):self.assertEqual(len(fuse_detections((Detection((0,0,20,50),.8),),(Detection((100,0,130,60),.7,detection_source="ROI_RECOVERY"),))),2)

if __name__=="__main__":unittest.main()
