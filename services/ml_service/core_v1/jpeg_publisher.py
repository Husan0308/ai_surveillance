from __future__ import annotations
import threading,time
import cv2

class LatestJpegPublisher:
    """Encode at most once per camera per presentation tick; all clients reuse bytes."""
    def __init__(self,camera_id,store,fps=12,quality=82,max_width=960,max_height=540):
        self.camera_id=camera_id;self.store=store;self.interval=1/max(1.0,float(fps));self.quality=int(quality);self.max_width=int(max_width);self.max_height=int(max_height)
        self._stop=threading.Event();self._thread=None;self._lock=threading.Lock();self._jpeg=None;self._version=0;self.encoded=0;self.skipped_same_frame=0
    def start(self):
        self._thread=threading.Thread(target=self._run,name=f"core-v1-jpeg-{self.camera_id}",daemon=False);self._thread.start()
    def stop(self):self._stop.set()
    def join(self,timeout=3):
        if self._thread:self._thread.join(timeout)
    def latest(self):
        with self._lock:return self._jpeg,self._version
    def _run(self):
        last_frame_id=-1;next_at=time.monotonic()
        while not self._stop.is_set():
            now=time.monotonic()
            if now<next_at:
                self._stop.wait(next_at-now);continue
            next_at=max(next_at+self.interval,now)
            frame,_=self.store.get()
            if frame is None:continue
            if frame.frame_id==last_frame_id:self.skipped_same_frame+=1;continue
            image=frame.image;h,w=image.shape[:2];scale=min(1.0,self.max_width/max(1,w),self.max_height/max(1,h))
            if scale<1.0:image=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
            ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,self.quality])
            if not ok:continue
            with self._lock:self._jpeg=encoded.tobytes();self._version+=1
            last_frame_id=frame.frame_id;self.encoded+=1
