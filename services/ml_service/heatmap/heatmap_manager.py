import time,threading
from .accumulator import HeatmapAccumulator
from .position_resolver import PositionResolver
from .metrics import HeatmapMetrics,CameraHeatmapMetrics
from .schemas import HeatmapMode

class HeatmapManager:
    def __init__(self,config=None):
        self.cfg=(config or {}).get("heatmap",{});self.enabled=bool(self.cfg.get("enabled",True));self.resolver=PositionResolver();self.metrics=HeatmapMetrics();self._lock=threading.RLock();self._accumulators={}
    def _accumulator(self,camera_id):
        with self._lock:
            if camera_id not in self._accumulators:
                c=self.cfg;self._accumulators[camera_id]=HeatmapAccumulator(camera_id,c.get("grid_width",c.get("grid_w",160)),c.get("grid_height",c.get("grid_h",90)),c.get("kernel_radius",4),c.get("sigma",2),c.get("sample_interval_ms",250),c.get("live_decay_enabled",True),c.get("live_decay_half_life_seconds",30),c.get("accumulation_mode","POINT"),c.get("max_tracked_keys",4096));self.metrics.cameras[camera_id]=CameraHeatmapMetrics()
            return self._accumulators[camera_id]
    def update(self,camera_id,frame_id,tracks,frame_width,frame_height,timestamp=None):
        if not self.enabled:return 0
        now=timestamp or time.time();positions=[]
        for track in tracks:
            is_dict=isinstance(track,dict);bbox=track.get("bbox") if is_dict else getattr(track,"bbox",None);global_id=track.get("global_id") if is_dict else getattr(track,"global_id",None);local_id=track.get("local_track_id") if is_dict else getattr(track,"local_track_id",None)
            position=self.resolver.resolve(camera_id,frame_id,bbox,frame_width,frame_height,now,global_id,local_id)
            if position:positions.append(position)
        started=time.perf_counter();updated,skipped=self._accumulator(camera_id).update(positions,now);elapsed=(time.perf_counter()-started)*1000
        metric=self.metrics.cameras[camera_id];metric.heatmap_updates+=int(updated>0);metric.heatmap_people_samples+=updated;metric.heatmap_skipped_interval+=skipped;metric.heatmap_update_ms=elapsed;return updated
    def snapshot(self,camera_id,mode=HeatmapMode.LIVE,normalized=True):return self._accumulator(camera_id).snapshot(mode,normalized)
    def camera_reconnected(self,camera_id,reset_live=False):
        accumulator=self._accumulators.get(camera_id)
        if accumulator is not None and reset_live:accumulator.live.fill(0)
    def remove_camera(self,camera_id):
        with self._lock:self._accumulators.pop(camera_id,None);self.metrics.cameras.pop(camera_id,None)
