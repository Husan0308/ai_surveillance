"""User-defined hidden ROI recovery scheduling, geometry, and fusion."""
from dataclasses import dataclass
import math,threading,time

@dataclass(frozen=True,slots=True)
class RecoveryROI:
    id:str;enabled:bool;polygon:tuple[tuple[float,float],...]

def validate_polygon(points):
    polygon=tuple((float(x),float(y)) for x,y in points)
    if len(polygon)<3:raise ValueError("ROI polygon requires at least 3 points")
    if any(not 0.0<=v<=1.0 for p in polygon for v in p):raise ValueError("ROI coordinates must be normalized to 0..1")
    area=abs(sum(polygon[i][0]*polygon[(i+1)%len(polygon)][1]-polygon[(i+1)%len(polygon)][0]*polygon[i][1] for i in range(len(polygon))))*.5
    if area<=1e-6:raise ValueError("ROI polygon must have non-zero area")
    return polygon

def source_polygon(roi,width,height):return tuple((x*width,y*height) for x,y in roi.polygon)
def crop_rectangle(roi,width,height):
    polygon=source_polygon(roi,width,height);xs=[p[0] for p in polygon];ys=[p[1] for p in polygon]
    return max(0,int(math.floor(min(xs)))),max(0,int(math.floor(min(ys)))),min(width,int(math.ceil(max(xs)))),min(height,int(math.ceil(max(ys))))
def point_in_polygon(point,polygon):
    x,y=point;inside=False;j=len(polygon)-1
    for i,(xi,yi) in enumerate(polygon):
        xj,yj=polygon[j]
        if (yi>y)!=(yj>y) and x<(xj-xi)*(y-yi)/(yj-yi)+xi:inside=not inside
        j=i
    return inside
def bbox_anchor(bbox):return ((bbox[0]+bbox[2])*.5,bbox[3])
def map_crop_bbox(bbox,offset):ox,oy=offset;return bbox[0]+ox,bbox[1]+oy,bbox[2]+ox,bbox[3]+oy
def bbox_iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]);x2,y2=min(a[2],b[2]),min(a[3],b[3]);inter=max(0,x2-x1)*max(0,y2-y1);aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);union=aa+bb-inter
    return inter/union if union else 0.0
def duplicate_detection(a,b):
    if bbox_iou(a,b)>=.45:return True
    ax,ay=bbox_anchor(a);bx,by=bbox_anchor(b);scale=max(1.0,min(a[2]-a[0],b[2]-b[0],a[3]-a[1],b[3]-b[1]));return math.hypot(ax-bx,ay-by)<=.35*scale
def fuse_detections(main,roi):
    output=list(main)
    for candidate in roi:
        duplicate=next((i for i,item in enumerate(output) if duplicate_detection(item.bbox_xyxy,candidate.bbox_xyxy)),None)
        if duplicate is None:output.append(candidate)
        elif candidate.confidence>output[duplicate].confidence:output[duplicate]=candidate
    return tuple(output)

class ROIRecoveryScheduler:
    """Normal detector first, then at most one fresh coalesced ROI per batch."""
    def __init__(self,discovery_interval_ms=2000,urgent_interval_ms=1000,max_task_age_ms=120):
        self.discovery_interval=max(.5,float(discovery_interval_ms)/1000);self.urgent_interval=max(.25,float(urgent_interval_ms)/1000);self.max_age=max(.01,float(max_task_age_ms)/1000);self._lock=threading.RLock();self._rois={};self._last={};self._hints={};self._pressure={"main_gpu_ms":0.0,"batch_fill":0.0,"pending":0,"gpu_utilization":0.0};self._metrics={"roi_inferences":0,"roi_recovered":0,"roi_duplicates_suppressed":0,"roi_stale_drops":0,"roi_skipped_main_covered":0,"roi_skipped_pressure":0,"roi_coalesced":0,"roi_urgent":0,"roi_discovery":0,"roi_inference_ms":0.0,"max_roi_per_batch":0}
    def configure(self,cameras):
        configured={}
        for camera in cameras:
            ids=set();items=[]
            for raw in camera.get("recovery_rois") or ():
                roi=RecoveryROI(str(raw["id"]),bool(raw.get("enabled",True)),validate_polygon(raw["polygon"]))
                if roi.id in ids:raise ValueError(f"duplicate ROI id for {camera['id']}: {roi.id}")
                ids.add(roi.id)
                if roi.enabled:items.append(roi)
            configured[str(camera["id"])]=tuple(items)
        with self._lock:self._rois=configured;self._last={k:v for k,v in self._last.items() if k[0] in configured};self._hints={k:v for k,v in self._hints.items() if k in configured}
    def update_track_hints(self,tracking):
        hints={}
        for camera in tracking.results:hints[camera.camera_id]=tuple({"bbox":track.predicted_bbox or track.bbox,"confirmed":bool(track.confirmed or track.state.value in ("CONFIRMED","LOST")),"misses":int(track.misses)} for track in camera.tracks)
        with self._lock:self._hints.update(hints)
    def update_pressure(self,main_gpu_ms=0.0,batch_size=0,batch_capacity=6,pending=0,gpu_utilization=0.0):
        """Record detector pressure without creating a work queue."""
        with self._lock:self._pressure={"main_gpu_ms":max(0.0,float(main_gpu_ms)),"batch_fill":min(1.0,float(batch_size)/max(1,int(batch_capacity))),"pending":max(0,int(pending)),"gpu_utilization":max(0.0,float(gpu_utilization))}
    def _intervals(self):
        p=self._pressure;overloaded=p["pending"]>0 or p["gpu_utilization"]>=90 or p["main_gpu_ms"]>=300
        busy=overloaded or p["batch_fill"]>=.99 or p["gpu_utilization"]>=75 or p["main_gpu_ms"]>=180
        return max(self.urgent_interval,3.0 if overloaded else 2.0 if busy else 0.0),max(self.discovery_interval,5.0 if overloaded else 3.0 if busy else 1.5),overloaded
    def select(self,packets,main_results,now=None):
        now=time.time() if now is None else float(now);main={item.camera_id:item.detections for item in main_results};options=[]
        with self._lock:
            urgent_interval,discovery_interval,overloaded=self._intervals()
            pressure=self._pressure
            busy=overloaded or pressure["batch_fill"]>=.99 or pressure["gpu_utilization"]>=75 or pressure["main_gpu_ms"]>=180
            for packet in packets:
                if now-packet.receive_timestamp>self.max_age:self._metrics["roi_stale_drops"]+=int(bool(self._rois.get(packet.camera_id)));continue
                for roi in self._rois.get(packet.camera_id,()):
                    polygon=source_polygon(roi,packet.width,packet.height);covered=any(point_in_polygon(bbox_anchor(item.bbox_xyxy),polygon) for item in main.get(packet.camera_id,()));hints=[h for h in self._hints.get(packet.camera_id,()) if h["confirmed"] and point_in_polygon(bbox_anchor(h["bbox"]),polygon)];urgent=any(h["misses"]>0 or not any(duplicate_detection(h["bbox"],item.bbox_xyxy) for item in main.get(packet.camera_id,())) for h in hints)
                    if covered and not urgent:self._metrics["roi_skipped_main_covered"]+=1;continue
                    interval=urgent_interval if urgent else discovery_interval;last=self._last.get((packet.camera_id,roi.id),0.0)
                    if busy and not urgent:self._metrics["roi_skipped_pressure"]+=1;continue
                    if now-last>=interval:options.append((0 if urgent else 1,last,packet.camera_id,roi.id,packet,roi,urgent))
            if not options:return None
            self._metrics["roi_coalesced"]+=max(0,len(options)-1)
            selected=min(options,key=lambda x:(x[0],x[1],x[2],x[3]));packet,roi,urgent=selected[4],selected[5],selected[6];self._last[(packet.camera_id,roi.id)]=now;self._metrics["roi_urgent" if urgent else "roi_discovery"]+=1;return packet,roi,urgent
    def record(self,elapsed_ms,recovered,duplicates):
        with self._lock:self._metrics["roi_inferences"]+=1;self._metrics["roi_recovered"]+=int(recovered);self._metrics["roi_duplicates_suppressed"]+=int(duplicates);self._metrics["roi_inference_ms"]+=float(elapsed_ms);self._metrics["max_roi_per_batch"]=max(self._metrics["max_roi_per_batch"],1)
    def snapshot(self):
        with self._lock:
            v=dict(self._metrics);urgent,discovery,overloaded=self._intervals();v.update(configured_cameras=sum(bool(x) for x in self._rois.values()),configured_rois=sum(map(len,self._rois.values())),mean_roi_inference_ms=v["roi_inference_ms"]/max(1,v["roi_inferences"]),effective_urgent_interval_ms=urgent*1000,effective_discovery_interval_ms=discovery*1000,overloaded=overloaded,pressure=dict(self._pressure));return v
