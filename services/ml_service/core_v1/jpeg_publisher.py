from __future__ import annotations
import threading,time
import cv2

class LatestJpegPublisher:
    """Publish one newest JPEG per camera with no presentation backlog.

    Camera capture owns the source frame. This publisher samples that one-slot
    store at the presentation rate, resizes first, then draws a fresh detector
    overlay on the smaller display image. Expensive full-resolution copies are
    avoided so detector overlays cannot push the six-camera display behind.
    """
    def __init__(self,camera_id,store,fps=12,quality=82,max_width=960,max_height=540,detections=None,overlay_max_age_ms=350):
        self.camera_id=camera_id;self.store=store;self.interval=1/max(1.0,float(fps));self.quality=int(quality);self.max_width=int(max_width);self.max_height=int(max_height)
        self.detections=detections;self.overlay_max_age_ms=max(0.0,float(overlay_max_age_ms))
        self._stop=threading.Event();self._thread=None;self._lock=threading.Lock();self._jpeg=None;self._version=0;self._published_monotonic=0.0;self._source_frame_id=0
        self.encoded=0;self.skipped_same_frame=0
    def start(self):
        self._thread=threading.Thread(target=self._run,name=f"core-v1-jpeg-{self.camera_id}",daemon=False);self._thread.start()
    def stop(self):self._stop.set()
    def join(self,timeout=3):
        if self._thread:self._thread.join(timeout)
    def latest(self):
        with self._lock:return self._jpeg,self._version
    def snapshot(self):
        with self._lock:return self._jpeg,self._version,self._published_monotonic,self._source_frame_id
    def _draw_detection(self,image,source_width,source_height):
        if self.detections is None:return image
        result=self.detections.get(self.camera_id)
        if result is None:return image
        age_ms=max(0.0,(time.monotonic()-result.produced_monotonic)*1000.0)
        if age_ms>self.overlay_max_age_ms:return image
        h,w=image.shape[:2];sx=w/max(1.0,float(source_width));sy=h/max(1.0,float(source_height))
        for box in result.boxes:
            x1=int(round(box.x1*sx));y1=int(round(box.y1*sy));x2=int(round(box.x2*sx));y2=int(round(box.y2*sy))
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(image,f"person {box.confidence:.2f}",(x1,max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.50,(0,255,0),2,cv2.LINE_AA)
        return image
    def _run(self):
        last_frame_id=-1;next_at=time.monotonic()
        while not self._stop.is_set():
            now=time.monotonic()
            if now<next_at:
                self._stop.wait(next_at-now);continue
            # If one encode tick was late, jump directly to 'now'. Never replay
            # missed presentation ticks and never build a JPEG backlog.
            next_at=now+self.interval
            frame,_=self.store.get()
            if frame is None:continue
            if frame.frame_id==last_frame_id:self.skipped_same_frame+=1;continue
            image=frame.image;source_h,source_w=image.shape[:2];scale=min(1.0,self.max_width/max(1,source_w),self.max_height/max(1,source_h))
            if scale<1.0:
                image=cv2.resize(image,(max(1,round(source_w*scale)),max(1,round(source_h*scale))),interpolation=cv2.INTER_AREA)
            else:
                # Draw overlays only on an owned image; the camera latest frame
                # must remain immutable for the detector and other consumers.
                image=image.copy()
            image=self._draw_detection(image,source_w,source_h)
            ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,self.quality])
            if not ok:continue
            published=time.monotonic()
            with self._lock:
                self._jpeg=encoded.tobytes();self._version+=1;self._published_monotonic=published;self._source_frame_id=frame.frame_id
            last_frame_id=frame.frame_id;self.encoded+=1
