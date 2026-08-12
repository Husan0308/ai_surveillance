"""Thread-safe single-slot latest-frame buffer."""
from __future__ import annotations
from collections import deque
import threading
from typing import Callable,Optional
from .frame import FramePacket

class LatestFrameBuffer:
    """Replace an unconsumed frame atomically; a backlog is impossible."""
    def __init__(self,on_available:Optional[Callable[[],None]]=None,capacity:int=1)->None:
        if int(capacity)!=1:raise ValueError("LatestFrameBuffer capacity is fixed at 1")
        self.maxsize=1;self._condition=threading.Condition(threading.Lock());self._packets=deque(maxlen=1);self._closed=False;self._on_available=on_available;self.dropped_old=0
    def put(self,packet:FramePacket)->None:
        with self._condition:
            if self._closed:return
            if len(self._packets)==self.maxsize:self.dropped_old+=1
            self._packets.append(packet);self._condition.notify_all()
        if self._on_available is not None:self._on_available()
    def take(self)->FramePacket|None:
        with self._condition:
            if not self._packets:return None
            packet=self._packets[-1];self.dropped_old+=max(0,len(self._packets)-1);self._packets.clear();return packet
    def peek(self)->FramePacket|None:
        with self._condition:return self._packets[-1] if self._packets else None
    def packets(self)->tuple[FramePacket,...]:
        with self._condition:return tuple(self._packets)
    def wait_and_take(self,timeout:float|None=None)->FramePacket|None:
        with self._condition:self._condition.wait_for(lambda:bool(self._packets) or self._closed,timeout)
        return self.take()
    def close(self)->None:
        with self._condition:self._closed=True;self._packets.clear();self._condition.notify_all()
    def __len__(self)->int:
        with self._condition:return len(self._packets)
