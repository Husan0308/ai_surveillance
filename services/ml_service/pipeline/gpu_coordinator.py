"""Process-wide GPU admission gate with named, auditable owners."""
from contextlib import contextmanager
from collections import deque
import threading,time

class GPUInferenceCoordinator:
 def __init__(self,secondary_min_interval=0.0):
  self._condition=threading.Condition();self._primary_waiting=0;self._active=None;self._max_active=0;self._overlap_violations=0;self._primary_wait_ms=0.0;self._secondary_wait_ms=0.0;self._roi_wait_ms=0.0;self._primary_entries=0;self._secondary_entries=0;self._roi_entries=0;self._secondary_min_interval=max(0.0,float(secondary_min_interval));self._last_secondary=0.0;self._history=deque(maxlen=256);self._request_id=0
 def _request(self,kind,owner):
  self._request_id+=1
  item={"request_id":self._request_id,"kind":kind,"owner":str(owner),"request_time":time.time(),"request_monotonic":time.monotonic()}
  self._history.append({"event":"request",**item});return item
 def _enter(self,item,started):
  kind,owner=item["kind"],item["owner"]
  if self._active is not None:self._overlap_violations+=1
  now=time.monotonic();wait=(time.perf_counter()-started)*1000
  item.update(acquire_time=time.time(),acquire_monotonic=now,wait_ms=wait,start_time=time.time(),start_monotonic=now);self._active=item;self._max_active=max(self._max_active,1)
  if kind=="primary":self._primary_wait_ms+=wait
  elif kind=="roi":self._roi_wait_ms+=wait
  else:self._secondary_wait_ms+=wait
  self._history.append({"event":"acquire",**item});return item
 def _leave(self,owner):
  now=time.monotonic();active=self._active or {}
  active.update(end_time=time.time(),end_monotonic=now,release_time=time.time(),release_monotonic=now,duration_ms=max(0.0,(now-active.get("start_monotonic",now))*1000))
  self._history.append({"event":"release",**active});self._active=None
 @contextmanager
 def primary(self,owner="YOLO"):
  started=time.perf_counter()
  with self._condition:
   item=self._request("primary",owner)
   self._primary_waiting+=1
   while self._active is not None:self._condition.wait()
   self._primary_waiting-=1;self._enter(item,started);self._primary_entries+=1
  try:yield item
  finally:
   with self._condition:self._leave(owner);self._condition.notify_all()
 @contextmanager
 def secondary(self,owner="SECONDARY"):
  started=time.perf_counter()
  with self._condition:
   item=self._request("secondary",owner)
   while True:
    cooldown=max(0.0,self._secondary_min_interval-(time.monotonic()-self._last_secondary))
    if self._active is None and not self._primary_waiting and cooldown<=0:break
    self._condition.wait(cooldown if cooldown>0 else None)
   self._enter(item,started);self._last_secondary=time.monotonic();self._secondary_entries+=1
  try:yield item
  finally:
   with self._condition:self._leave(owner);self._condition.notify_all()
 @contextmanager
 def roi(self,owner="ROI"):
  started=time.perf_counter()
  with self._condition:
   item=self._request("roi",owner)
   while self._active is not None or self._primary_waiting:self._condition.wait()
   self._enter(item,started);self._roi_entries+=1
  try:yield item
  finally:
   with self._condition:self._leave(owner);self._condition.notify_all()
 def snapshot(self,include_history=True):
  with self._condition:
   item={"active":dict(self._active) if self._active else None,"primary_waiting":self._primary_waiting,"max_active":self._max_active,"overlap_violations":self._overlap_violations,"primary_entries":self._primary_entries,"secondary_entries":self._secondary_entries,"roi_entries":self._roi_entries,"primary_wait_ms":self._primary_wait_ms,"secondary_wait_ms":self._secondary_wait_ms,"roi_wait_ms":self._roi_wait_ms}
   if include_history:item["history"]=tuple(self._history)
   return item

gpu_coordinator=GPUInferenceCoordinator()
