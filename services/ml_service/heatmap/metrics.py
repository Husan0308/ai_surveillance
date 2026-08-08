from dataclasses import dataclass,asdict
@dataclass
class CameraHeatmapMetrics:
    heatmap_updates:int=0;heatmap_people_samples:int=0;heatmap_skipped_interval:int=0;heatmap_update_ms:float=0
class HeatmapMetrics:
    def __init__(self):self.cameras={}
    def snapshot(self):return {"cameras":{k:asdict(v) for k,v in self.cameras.items()},"active_heatmaps":len(self.cameras),"total_heatmap_update_ms":sum(v.heatmap_update_ms for v in self.cameras.values())}
