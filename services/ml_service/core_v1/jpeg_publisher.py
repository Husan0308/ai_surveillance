from __future__ import annotations
import threading,time
import cv2
from .visual_tracker import VisualTracker

class LatestJpegPublisher:
    """Publish one newest JPEG per camera with no presentation backlog."""
    def __init__(self,camera_id,store,fps=12,quality=82,max_width=960,max_height=540,detections=None,overlay_max_age_ms=350,tracker_config=None):
        self.camera_id=camera_id;self.store=store;self.interval=1/max(1.0,float(fps));self.quality=int(quality);self.max_width=int(max_width);self.max_height=int(max_height)
        self.detections=detections;self.overlay_max_age_ms=max(0.0,float(overlay_max_age_ms))
        cfg=dict(tracker_config or {})
        camera_zones=dict(cfg.get('camera_exclusion_zones') or {})
        self.visual_tracker=VisualTracker(
            hold_ms=cfg.get('hold_ms',800),memory_ms=cfg.get('memory_ms',6000),prediction_ms=cfg.get('prediction_ms',350),
            match_iou=cfg.get('match_iou',0.20),reacquire_distance=cfg.get('reacquire_distance',1.05),
            duplicate_iou=cfg.get('duplicate_iou',0.50),duplicate_containment=cfg.get('duplicate_containment',0.82),
            duplicate_center_distance=cfg.get('duplicate_center_distance',0.40),smoothing=cfg.get('smoothing',0.68),
            low_conf_confirm=cfg.get('low_conf_confirm',0.16),start_conf=cfg.get('start_conf',0.18),
            exclusion_zones=camera_zones.get(camera_id,[]),exclusion_max_box_height=cfg.get('exclusion_max_box_height',0.34),
        )
        self._stop=threading.Event();self._thread=None;self._lock=threading.Lock();self._condition=threading.Condition(self._lock)
        self._jpeg=None;self._version=0;self._published_monotonic=0.0;self._source_frame_id=0
        self.encoded=0;self.skipped_same_frame=0
    def start(self):
        self._thread=threading.Thread(target=self._run,name=f"core-v1-jpeg-{self.camera_id}",daemon=False);self._thread.start()
    def stop(self):
        self._stop.set()
        with self._condition:self._condition.notify_all()
    def join(self,timeout=3):
        if self._thread:self._thread.join(timeout)
    def latest(self):
        with self._lock:return self._jpeg,self._version
    def snapshot(self):
        with self._lock:return self._jpeg,self._version,self._published_monotonic,self._source_frame_id
    def wait_newer(self,last_version:int,timeout:float=0.25):
        deadline=time.monotonic()+max(0.0,float(timeout))
        with self._condition:
            while self._version<=last_version and not self._stop.is_set():
                remaining=deadline-time.monotonic()
                if remaining<=0:break
                self._condition.wait(remaining)
            return self._jpeg,self._version,self._published_monotonic,self._source_frame_id
    def _draw_detection(self,image,source_width,source_height,now):
        if self.detections is not None:
            result=self.detections.get(self.camera_id)
            if result is not None:
                self.visual_tracker.update(result,now,source_width,source_height)
        boxes=self.visual_tracker.visible(now)
        if not boxes:return image
        h,w=image.shape[:2];sx=w/max(1.0,float(source_width));sy=h/max(1.0,float(source_height))
        for box in boxes:
            x1=int(round(box.x1*sx));y1=int(round(box.y1*sy));x2=int(round(box.x2*sx));y2=int(round(box.y2*sy))
            x1=max(0,min(w-1,x1));x2=max(0,min(w-1,x2));y1=max(0,min(h-1,y1));y2=max(0,min(h-1,y2))
            if x2<=x1 or y2<=y1:continue
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(image,f"person {box.confidence:.2f}",(x1,max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.50,(0,255,0),2,cv2.LINE_AA)
        return image
    def _run(self):
        last_frame_id=-1;next_at=time.monotonic()
        while not self._stop.is_set():
            now=time.monotonic()
            if now<next_at:
                self._stop.wait(next_at-now);continue
            next_at=now+self.interval
            frame,_=self.store.get()
            if frame is None:continue
            if frame.frame_id==last_frame_id:self.skipped_same_frame+=1;continue
            image=frame.image;source_h,source_w=image.shape[:2];scale=min(1.0,self.max_width/max(1,source_w),self.max_height/max(1,source_h))
            if scale<1.0:image=cv2.resize(image,(max(1,round(source_w*scale)),max(1,round(source_h*scale))),interpolation=cv2.INTER_AREA)
            else:image=image.copy()
            image=self._draw_detection(image,source_w,source_h,now)
            ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,self.quality])
            if not ok:continue
            published=time.monotonic();payload=encoded.tobytes()
            with self._condition:
                self._jpeg=payload;self._version+=1;self._published_monotonic=published;self._source_frame_id=frame.frame_id
                self._condition.notify_all()
            last_frame_id=frame.frame_id;self.encoded+=1
