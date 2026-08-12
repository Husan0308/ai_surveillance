"""Exclusive owner of the optional on-demand display reader.

AI readers are owned only by CameraManager.  Equal AI/display sources use a
non-owning subscription and never enter this manager.
"""
import threading
from .buffer import LatestFrameBuffer
from .reader import CameraReader

def reuses_ai_reader(config):
    return bool(config.get("display_source")) and config.get("display_source")==config.get("ai_source")

class OnDemandDisplayManager:
    def __init__(self,on_frame,capture_factory=None):
        self._on_frame=on_frame;self._factory=capture_factory;self._lock=threading.RLock();self._reader=None;self._buffer=None;self._camera_id=None;self._starts=0;self._stops=0;self._failed_joins=0
    @property
    def camera_id(self):
        with self._lock:return self._camera_id
    def start(self,camera_id,config):
        source=config.get("display_source")
        if not source or reuses_ai_reader(config):return False
        with self._lock:
            if self._camera_id==camera_id and self._reader is not None:return True
        self.stop()
        item={**config,"id":camera_id,"source":source,"codec":config.get("display_codec") or config.get("codec")};buffer=LatestFrameBuffer(capacity=1);reader=CameraReader(item,buffer,self._factory,self._on_frame)
        with self._lock:self._camera_id=camera_id;self._reader=reader;self._buffer=buffer;self._starts+=1
        reader.start();return True
    def stop(self,camera_id=None):
        with self._lock:
            if camera_id is not None and camera_id!=self._camera_id:return False
            reader,buffer=self._reader,self._buffer;self._reader=self._buffer=None;self._camera_id=None
        if reader:
            reader.stop();joined=reader.join(6)
            with self._lock:self._stops+=1;self._failed_joins+=int(not joined)
        if buffer:buffer.close()
        return reader is not None
    def snapshot(self):
        with self._lock:reader=self._reader;camera_id=self._camera_id;starts=self._starts;stops=self._stops;failed=self._failed_joins
        status=reader.metrics() if reader is not None else {}
        return {"active_reader_count":int(reader is not None),"camera_id":camera_id,"display_online":bool(status.get("online",False)),"starts":starts,"stops":stops,"failed_joins":failed}
    def shutdown(self):self.stop()
