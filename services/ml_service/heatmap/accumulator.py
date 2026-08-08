import time
from datetime import datetime
import numpy as np
from .schemas import HeatmapMode,CameraHeatmapSnapshot
from .temporal import LiveDecay

class HeatmapAccumulator:
    def __init__(self,camera_id,width=160,height=90,kernel_radius=4,sigma=2,sample_interval_ms=250,live_decay_enabled=True,live_decay_half_life_seconds=30,mode="POINT",max_tracked_keys=4096):
        self.camera_id=camera_id;self.width=int(width);self.height=int(height);self.interval=float(sample_interval_ms)/1000;self.mode=mode
        self.live=np.zeros((self.height,self.width),np.float32);self.minute=np.zeros_like(self.live);self.hourly=np.zeros_like(self.live);self.daily=np.zeros_like(self.live)
        self.decay=LiveDecay(live_decay_enabled,live_decay_half_life_seconds);self.last_samples={};self.max_tracked_keys=max(1,int(max_tracked_keys));self.last_frame_size=(0,0)
        axis=np.arange(-kernel_radius,kernel_radius+1,dtype=np.float32);xx,yy=np.meshgrid(axis,axis);kernel=np.exp(-(xx*xx+yy*yy)/(2*float(sigma)**2));self.kernel=kernel/kernel.max();self.radius=kernel_radius
        self.bucket_minute=self.bucket_hour=self.bucket_day=None

    def update(self,positions,timestamp=None):
        now=timestamp or time.time();self.decay.apply(self.live,now);self._roll(now);updated=skipped=0
        for position in positions:
            previous=self.last_samples.get(position.identity_key)
            if previous is not None and now-previous<self.interval:skipped+=1;continue
            elapsed=max(self.interval,.001) if previous is None else max(0,now-previous);self.last_samples[position.identity_key]=now;self.last_frame_size=(position.frame_width,position.frame_height)
            if len(self.last_samples)>self.max_tracked_keys:
                self.last_samples.pop(min(self.last_samples,key=self.last_samples.get))
            gx=min(self.width-1,max(0,int(position.x_norm*(self.width-1))));gy=min(self.height-1,max(0,int(position.y_norm*(self.height-1))))
            self._add(self.live,gx,gy,elapsed);self._add(self.minute,gx,gy,elapsed);self._add(self.hourly,gx,gy,elapsed);self._add(self.daily,gx,gy,elapsed);updated+=1
        return updated,skipped

    def _roll(self,timestamp):
        dt=datetime.fromtimestamp(timestamp);minute=(dt.year,dt.month,dt.day,dt.hour,dt.minute);hour=minute[:-1];day=hour[:-1]
        if self.bucket_minute is not None and minute!=self.bucket_minute:self.minute.fill(0)
        if self.bucket_hour is not None and hour!=self.bucket_hour:self.hourly.fill(0)
        if self.bucket_day is not None and day!=self.bucket_day:self.daily.fill(0)
        self.bucket_minute,self.bucket_hour,self.bucket_day=minute,hour,day

    def _splat(self,grid,gx,gy,weight):
        r=self.radius;x1,x2=max(0,gx-r),min(self.width,gx+r+1);y1,y2=max(0,gy-r),min(self.height,gy+r+1)
        kx1,ky1=x1-(gx-r),y1-(gy-r);grid[y1:y2,x1:x2]+=self.kernel[ky1:ky1+y2-y1,kx1:kx1+x2-x1]*weight

    def _add(self,grid,gx,gy,weight):
        if str(self.mode).upper().endswith("FOOTPRINT"):
            spread=max(1,self.radius//2)
            self._splat(grid,max(0,gx-spread),gy,weight*.5)
            self._splat(grid,min(self.width-1,gx+spread),gy,weight*.5)
        else:
            self._splat(grid,gx,gy,weight)

    def snapshot(self,mode=HeatmapMode.LIVE,normalized=True,timestamp=None):
        mode=HeatmapMode(mode);grid={HeatmapMode.LIVE:self.live,HeatmapMode.MINUTE:self.minute,HeatmapMode.HOURLY:self.hourly,HeatmapMode.DAILY:self.daily}[mode]
        maximum=float(grid.max()) if grid.size else 0;values=(grid/maximum if normalized and maximum>0 else grid).copy();fw,fh=self.last_frame_size
        return CameraHeatmapSnapshot(self.camera_id,timestamp or time.time(),mode,self.width,self.height,fw,fh,maximum,values)
