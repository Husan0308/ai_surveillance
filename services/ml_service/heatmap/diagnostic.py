import time
from types import SimpleNamespace
from .heatmap_manager import HeatmapManager

def run():
    manager=HeatmapManager({"heatmap":{"sample_interval_ms":100}});now=time.time()
    for camera,count in (("CAM-01",3),("CAM-02",1),("CAM-03",0)):
        tracks=[SimpleNamespace(bbox=(10+i*30,10,40+i*30,100),global_id=f"G{i}",local_track_id=f"T{i}") for i in range(count)]
        manager.update(camera,1,tracks,640,360,now);metric=manager.metrics.cameras[camera];snapshot=manager.snapshot(camera)
        print(f"{camera}\npeople:{count}\nupdates:{metric.heatmap_people_samples}\nupdate_ms:{metric.heatmap_update_ms:.3f}\npeak:{snapshot.max_value:.3f}")
if __name__=="__main__":print("HEATMAP");run()
