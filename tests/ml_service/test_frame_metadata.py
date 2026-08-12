import unittest
from types import SimpleNamespace
from services.ml_service.cameras.frame import FramePacket
from services.ml_service.events import frame_metadata_messages,merge_visual_identity_results
from services.ml_service.identity.schemas import GlobalTrack,GlobalTrackResult,IdentityStatus
from services.ml_service.tracking.schemas import TrackState
from services.frontend.video_renderer import MetadataBuffer

class FrameMetadataTests(unittest.TestCase):
    def test_three_tracks_are_aggregated_into_one_frame_message(self):
        packet=FramePacket("CAM-01",42,100.25,100.30,None,1280,720)
        results=[]
        for index in range(3):
            track=GlobalTrack(f"T{index}",f"UNK-{index}",(index,1,index+10,20),.9,.8,IdentityStatus.ACTIVE,"test")
            results.append(GlobalTrackResult("CAM-01",42,(track,)))
        messages=frame_metadata_messages((packet,),results)
        self.assertEqual(len(messages),1);self.assertEqual(len(messages[0]["tracks"]),3)
        self.assertEqual(messages[0]["frame_id"],42);self.assertEqual(messages[0]["timestamp"],100.25)
        self.assertEqual(messages[0]["capture_timestamp"],100.25)
        self.assertEqual((messages[0]["frame_width"],messages[0]["frame_height"]),(1280,720))
        buffer=MetadataBuffer();buffer.put(messages[0]);self.assertEqual(len(buffer.match("CAM-01",42,100.25)["tracks"]),3)

    def test_independent_display_matching_never_uses_future_metadata(self):
        buffer=MetadataBuffer();buffer.put({"camera_id":"CAM-06","frame_id":10,"timestamp":10.0,"tracks":[]});buffer.put({"camera_id":"CAM-06","frame_id":11,"timestamp":11.0,"tracks":[]})
        self.assertEqual(buffer.match("CAM-06",1,10.4,True)["frame_id"],10);self.assertIsNone(buffer.match("CAM-06",1,9.9,True))

    def test_duplicate_local_track_is_emitted_once(self):
        packet=FramePacket("CAM-01",9,10.0,10.1,None,3200,1800)
        first=GlobalTrack("T1","UNK-1",(0,0,10,10),.5,.5,IdentityStatus.ACTIVE,"test")
        newest=GlobalTrack("T1","UNK-1",(1,1,20,20),.8,.8,IdentityStatus.ACTIVE,"test")
        messages=frame_metadata_messages((packet,),(GlobalTrackResult("CAM-01",9,(first,)),GlobalTrackResult("CAM-01",9,(newest,))))
        self.assertEqual(len(messages[0]["tracks"]),1);self.assertEqual(messages[0]["tracks"][0]["bbox"],[1,1,20,20])

    def test_confirmed_lost_track_gets_bounded_visual_persistence_and_tentative_is_real_only(self):
        cache={};identity=GlobalTrack("CAM-01:TRACK-00001","UNK-1",(10,10,30,50),.8,.7,IdentityStatus.ACTIVE,"test")
        live=(GlobalTrackResult("CAM-01",1,(identity,)),)
        confirmed=SimpleNamespace(camera_id="CAM-01",tracks=(SimpleNamespace(track_id=identity.local_track_id,state=TrackState.CONFIRMED,misses=0,bbox=identity.bbox,predicted_bbox=identity.bbox),))
        merge_visual_identity_results(SimpleNamespace(results=(confirmed,)),live,cache,3)
        lost=SimpleNamespace(camera_id="CAM-01",frame_id=2,tracks=(SimpleNamespace(track_id=identity.local_track_id,state=TrackState.LOST,misses=2,bbox=(11,10,31,50),predicted_bbox=(12,10,32,50)),))
        visual=merge_visual_identity_results(SimpleNamespace(results=(lost,)),(),cache,3)
        self.assertEqual(len(visual),1);self.assertEqual(visual[0].tracks[0].bbox,(12,10,32,50))
        self.assertEqual(visual[0].tracks[0].observation_type,"predicted")
        tentative=SimpleNamespace(camera_id="CAM-02",frame_id=2,tracks=(SimpleNamespace(track_id="T2",local_track_id="T2",state=TrackState.TENTATIVE,misses=0,bbox=(0,0,10,20),predicted_bbox=(0,0,10,20),confidence=.31,last_detection_timestamp=10.0,detection_source="FULL_FRAME",detection_id="D2",velocity=(0.0,0.0),state_timestamp=10.0,visual_expires_at=0.0,track_generation=1,geometry_monotonic=10.0,visual_visible=True,boundary_exit=False),))
        tentative_visual=merge_visual_identity_results(SimpleNamespace(results=(tentative,)),(),{},3)
        self.assertEqual(len(tentative_visual),1);self.assertEqual(tentative_visual[0].tracks[0].tracker_state,"TENTATIVE");self.assertIsNone(tentative_visual[0].tracks[0].global_id)
        missed_tentative=SimpleNamespace(camera_id="CAM-02",frame_id=3,tracks=(SimpleNamespace(track_id="T2",local_track_id="T2",state=TrackState.TENTATIVE,misses=1,bbox=(0,0,10,20),predicted_bbox=(1,0,11,20),confidence=.31),))
        self.assertEqual(merge_visual_identity_results(SimpleNamespace(results=(missed_tentative,)),(),{},3),())

    def test_tracker_numeric_local_id_and_identity_string_key_still_share_visual_cache(self):
        cache={};stable="CAM-05:TRACK-00007"
        identity=GlobalTrack(stable,"UNK-7",(10,10,30,60),.8,.8,IdentityStatus.ACTIVE,"test")
        detected_track=SimpleNamespace(track_id=stable,local_track_id=7,state=TrackState.CONFIRMED,misses=0,bbox=(10,10,30,60),predicted_bbox=(10,10,30,60),last_detection_timestamp=10.0,prediction_age_ms=0.0,velocity=(2.0,0.0),state_timestamp=10.0,visual_expires_at=11.8,track_generation=1,geometry_monotonic=10.0,visual_visible=True,boundary_exit=False)
        detected_camera=SimpleNamespace(camera_id="CAM-05",frame_id=10,tracks=(detected_track,))
        merged=merge_visual_identity_results(SimpleNamespace(results=(detected_camera,)),(GlobalTrackResult("CAM-05",10,(identity,)),),cache,18)
        self.assertEqual(len(merged),1);self.assertIn(("CAM-05",stable),cache)
        predicted_track=SimpleNamespace(track_id=stable,local_track_id=7,state=TrackState.LOST,misses=1,bbox=(11,10,31,60),predicted_bbox=(12,10,32,60),last_detection_timestamp=10.0,prediction_age_ms=400.0,velocity=(2.0,0.0),state_timestamp=10.4,visual_expires_at=11.8,track_generation=1,geometry_monotonic=10.4,visual_visible=True,boundary_exit=False)
        predicted_camera=SimpleNamespace(camera_id="CAM-05",frame_id=11,tracks=(predicted_track,))
        visual=merge_visual_identity_results(SimpleNamespace(results=(predicted_camera,)),(),cache,18)
        self.assertEqual(len(visual),1);self.assertEqual(visual[0].tracks[0].global_id,"UNK-7");self.assertEqual(visual[0].tracks[0].bbox,(12,10,32,60))

    def test_metadata_propagates_display_prediction_state_and_expiry(self):
        packet=FramePacket("CAM-01",5,10.5,10.5,None,640,360)
        track=GlobalTrack("T1","UNK-1",(10,10,30,50),.9,.8,IdentityStatus.ACTIVE,"test",velocity=(25.0,0.0,0.0,0.0),state_timestamp=10.5,visual_expires_at=11.1)
        payload=frame_metadata_messages((packet,),(GlobalTrackResult("CAM-01",5,(track,)),))[0]["tracks"][0]
        self.assertEqual(payload["velocity"],[25.0,0.0,0.0,0.0]);self.assertEqual(payload["state_timestamp"],10.5);self.assertEqual(payload["visual_expires_at"],11.1)

    def test_metadata_carries_identity_runtime_epoch(self):
        packet=FramePacket("CAM-01",1,10.0,10.1,None,640,360)
        self.assertEqual(frame_metadata_messages((packet,),(),runtime_epoch="run-2")[0]["identity_runtime_epoch"],"run-2")

    def test_empty_frame_still_clears_tracks(self):
        packet=FramePacket("CAM-02",7,10.0,10.1,None,640,360)
        self.assertEqual(frame_metadata_messages((packet,),())[0]["tracks"],[])
