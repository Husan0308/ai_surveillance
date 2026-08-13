from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Any

@dataclass(frozen=True, slots=True)
class Frame:
    camera_id: str
    frame_id: int
    captured_at: float
    captured_monotonic: float
    image: Any
    width: int
    height: int

class LatestFrameStore:
    """Latest-only hot path plus bounded exact-frame history for ReID.

    Normal camera/display/detector consumers still read one newest frame only.
    History is never replayed and cannot create presentation backlog. Ten source
    frames cover roughly 0.5 s at 20 FPS, enough for the measured detector p95
    latency while keeping memory usage bounded.
    """
    def __init__(self, history_size: int = 10) -> None:
        self._lock=threading.Lock();self._frame=None;self._version=0;self.replaced=0
        self._history=deque(maxlen=max(1,int(history_size)))
        self.history_hits=0;self.history_misses=0
    def put(self,frame:Frame)->None:
        with self._lock:
            if self._frame is not None:self.replaced+=1
            self._frame=frame;self._version+=1;self._history.append(frame)
    def get(self):
        with self._lock:return self._frame,self._version
    def get_frame(self,frame_id:int):
        target=int(frame_id)
        with self._lock:
            for frame in reversed(self._history):
                if int(frame.frame_id)==target:
                    self.history_hits+=1;return frame
            self.history_misses+=1;return None
    def history_metrics(self):
        with self._lock:return {"size":len(self._history),"capacity":self._history.maxlen,"hits":self.history_hits,"misses":self.history_misses}
    def wait_newer(self,last_version:int,timeout:float=1.0):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            frame,version=self.get()
            if frame is not None and version>last_version:return frame,version
            time.sleep(.003)
        return None,last_version
