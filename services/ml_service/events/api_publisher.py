"""Bounded, non-blocking publisher for semantic ML events."""
import json,queue,threading
from urllib.request import Request,urlopen
from shared.logging import get_logger
from shared.event_taxonomy import is_persistent,event_type
import time
from collections import Counter
log=get_logger(__name__)
class APIEventPublisher:
 def __init__(self,base_url,capacity=128,timeout=.75):self.base_url=base_url.rstrip("/");self.event_url=self.base_url+"/api/v1/internal/ml/events";self.realtime_url=self.base_url+"/api/v1/internal/ml/realtime";self.timeout=timeout;self.queue=queue.Queue(capacity);self.stop_event=threading.Event();self.thread=None;self.dropped=0;self.available=False;self.started=time.monotonic();self.counts=Counter();self.persistent_counts=Counter();self._metrics_lock=threading.Lock()
 def start(self):self.thread=threading.Thread(target=self._run,name="ml-api-publisher",daemon=True);self.thread.start()
 def publish(self,payload):
  item=dict(payload);kind=event_type(item);persistent=is_persistent(item)
  with self._metrics_lock:self.counts[kind]+=1;self.persistent_counts[kind]+=int(persistent)
  try:self.queue.put_nowait((persistent,item));return True
  except queue.Full:
   try:self.queue.get_nowait();self.queue.put_nowait((persistent,item))
   except queue.Empty:pass
   self.dropped+=1;return False
 def _run(self):
  while not self.stop_event.is_set():
   try:item=self.queue.get(timeout=.5)
   except queue.Empty:continue
   try:
    persistent,item=item;request=Request(self.event_url if persistent else self.realtime_url,data=json.dumps(item,default=str).encode(),headers={"Content-Type":"application/json"},method="POST")
    with urlopen(request,timeout=self.timeout) as response:response.read()
    self.available=True
   except Exception as exc:
    if self.available:log.warning("API event delivery unavailable: %s",exc)
    self.available=False
 def snapshot(self):
  elapsed=max(1e-6,time.monotonic()-self.started)
  with self._metrics_lock:return {"requests_total":sum(self.counts.values()),"requests_per_sec":sum(self.counts.values())/elapsed,"persistent_requests_total":sum(self.persistent_counts.values()),"persistent_requests_per_sec":sum(self.persistent_counts.values())/elapsed,"types":dict(self.counts),"persistent_types":dict(self.persistent_counts),"dropped":self.dropped}
 def close(self):
  self.stop_event.set()
  if self.thread:self.thread.join(2)
