from __future__ import annotations
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
    """One slot only. Producers replace old frames; consumers never build backlog."""
    def __init__(self) -> None:
        self._lock=threading.Lock();self._frame=None;self._version=0;self.replaced=0
    def put(self,frame:Frame)->None:
        with self._lock:
            if self._frame is not None:self.replaced+=1
            self._frame=frame;self._version+=1
    def get(self):
        with self._lock:return self._frame,self._version
    def wait_newer(self,last_version:int,timeout:float=1.0):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            frame,version=self.get()
            if frame is not None and version>last_version:return frame,version
            time.sleep(.003)
        return None,last_version
