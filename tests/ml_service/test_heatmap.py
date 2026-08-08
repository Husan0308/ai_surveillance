import time
import unittest
from types import SimpleNamespace

import numpy as np

from services.ml_service.heatmap.accumulator import HeatmapAccumulator
from services.ml_service.heatmap.heatmap_manager import HeatmapManager
from services.ml_service.heatmap.position_resolver import PositionResolver
from services.ml_service.heatmap.schemas import CameraPosition,HeatmapMode


def position(camera="CAM-01",key="G1",x=.5,y=.75,timestamp=None,width=640,height=360):
    return CameraPosition(camera,1,key,x,y,width,height,timestamp or time.time())


class HeatmapTests(unittest.TestCase):
    def test_bbox_bottom_center_and_normalization(self):
        p=PositionResolver().resolve("CAM-01",1,(100,20,300,180),400,200,1.0,"G1",1,(0,0))
        self.assertEqual((p.x_norm,p.y_norm),(.5,.9))

    def test_grid_mapping_gaussian_and_edge_clipping(self):
        acc=HeatmapAccumulator("CAM-01",20,10,2,1,100,False)
        acc.update([position(x=1,y=1)],time.time())
        self.assertGreater(acc.live[-1,-1],0)
        self.assertGreater(np.count_nonzero(acc.live),1)
        self.assertEqual(acc.live.shape,(10,20))

    def test_time_weight_and_sampling_interval(self):
        now=time.time();acc=HeatmapAccumulator("CAM-01",20,10,1,1,100,False)
        self.assertEqual(acc.update([position(timestamp=now)],now),(1,0))
        first=float(acc.live.sum())
        self.assertEqual(acc.update([position(timestamp=now+.05)],now+.05),(0,1))
        self.assertAlmostEqual(float(acc.live.sum()),first)
        acc.update([position(timestamp=now+.3)],now+.3)
        self.assertGreater(float(acc.live.sum()),first*2)

    def test_different_people_and_cameras_are_independent(self):
        manager=HeatmapManager({"heatmap":{"sample_interval_ms":100,"live_decay_enabled":False}})
        tracks=[SimpleNamespace(bbox=(10,10,30,50),global_id="G1",local_track_id=1),
                SimpleNamespace(bbox=(40,10,60,50),global_id="G2",local_track_id=2)]
        self.assertEqual(manager.update("CAM-01",1,tracks,100,100,time.time()),2)
        manager.update("CAM-02",1,tracks[:1],100,100,time.time())
        self.assertIsNot(manager._accumulators["CAM-01"].live,manager._accumulators["CAM-02"].live)
        self.assertNotEqual(float(manager._accumulators["CAM-01"].live.sum()),float(manager._accumulators["CAM-02"].live.sum()))

    def test_reconnect_preserves_history(self):
        manager=HeatmapManager({"heatmap":{"live_decay_enabled":False}})
        track=SimpleNamespace(bbox=(10,10,30,50),global_id=None,local_track_id=1)
        manager.update("CAM-01",1,[track],100,100,time.time())
        before=manager.snapshot("CAM-01",HeatmapMode.DAILY,False).values
        manager.camera_reconnected("CAM-01",reset_live=True)
        np.testing.assert_array_equal(before,manager.snapshot("CAM-01",HeatmapMode.DAILY,False).values)

    def test_live_decay_does_not_decay_history(self):
        now=time.time();acc=HeatmapAccumulator("CAM-01",20,10,1,1,100,True,1)
        acc.update([position(timestamp=now)],now)
        live=float(acc.live.max());historical=float(acc.daily.max())
        acc.update([],now+1)
        self.assertAlmostEqual(float(acc.live.max()),live*.5,places=4)
        self.assertEqual(float(acc.daily.max()),historical)

    def test_empty_and_resolution_change(self):
        acc=HeatmapAccumulator("CAM-01",20,10,1,1,100,False)
        self.assertEqual(acc.update([],time.time()),(0,0))
        now=time.time();resolver=PositionResolver()
        p1=resolver.resolve("CAM-01",1,(100,0,300,180),400,200,now,"G1",1)
        p2=resolver.resolve("CAM-01",2,(200,0,600,360),800,400,now+.2,"G1",1)
        self.assertEqual((p1.x_norm,p1.y_norm),(p2.x_norm,p2.y_norm))

    def test_bounded_sampling_memory(self):
        acc=HeatmapAccumulator("CAM-01",20,10,1,1,0,False,max_tracked_keys=3)
        now=time.time()
        acc.update([position(key=f"G{i}",timestamp=now+i) for i in range(8)],now)
        self.assertLessEqual(len(acc.last_samples),3)

    def test_footprint_mode(self):
        point=HeatmapAccumulator("A",30,20,2,1,0,False,mode="POINT")
        footprint=HeatmapAccumulator("B",30,20,2,1,0,False,mode="FOOTPRINT")
        now=time.time();point.update([position()],now);footprint.update([position()],now)
        self.assertGreater(np.count_nonzero(footprint.live),np.count_nonzero(point.live))


if __name__=="__main__":
    unittest.main()
