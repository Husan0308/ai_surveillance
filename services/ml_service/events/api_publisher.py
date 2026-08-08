"""Bounded, non-blocking publisher for semantic ML events."""
import json,queue,threading
from urllib.request import Request,urlopen
from shared.logging import get_logger
log=get_logger(__name__)
class APIEventPublisher:
 def __init__(self,base_url,capacity=128,timeout=.75):self.url=base_url.rstrip("/")+"/api/v1/internal/ml/events";self.timeout=timeout;self.queue=queue.Queue(capacity);self.stop_event=threading.Event();self.thread=None;self.dropped=0;self.available=False
 def start(self):self.thread=threading.Thread(target=self._run,name="ml-api-publisher",daemon=True);self.thread.start()
 def publish(self,payload):
  try:self.queue.put_nowait(dict(payload));return True
  except queue.Full:
   try:self.queue.get_nowait();self.queue.put_nowait(dict(payload))
   except queue.Empty:pass
   self.dropped+=1;return False
 def _run(self):
  while not self.stop_event.is_set():
   try:item=self.queue.get(timeout=.5)
   except queue.Empty:continue
   try:
    request=Request(self.url,data=json.dumps(item,default=str).encode(),headers={"Content-Type":"application/json"},method="POST")
    with urlopen(request,timeout=self.timeout) as response:response.read()
    self.available=True
   except Exception as exc:
    if self.available:log.warning("API event delivery unavailable: %s",exc)
    self.available=False
 def close(self):
  self.stop_event.set()
  if self.thread:self.thread.join(2)
