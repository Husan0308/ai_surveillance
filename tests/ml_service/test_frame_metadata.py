import unittest
from types import SimpleNamespace
from services.ml_service.cameras.frame import FramePacket
from services.ml_service.events import frame_metadata_messages
from services.ml_service.identity.schemas import GlobalTrack,GlobalTrackResult,IdentityStatus
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
        buffer=MetadataBuffer();buffer.put(messages[0]);self.assertEqual(len(buffer.match("CAM-01",42,100.25)["tracks"]),3)

    def test_empty_frame_still_clears_tracks(self):
        packet=FramePacket("CAM-02",7,10.0,10.1,None,640,360)
        self.assertEqual(frame_metadata_messages((packet,),())[0]["tracks"],[])
