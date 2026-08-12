"""Bounded frame/metadata synchronization; camera acquisition remains ML-owned."""
import threading,time
from collections import OrderedDict
class MetadataBuffer:
 def __init__(self,capacity=32,max_age_ms=500):self.capacity=capacity;self.max_age=max_age_ms/1000;self._items={};self._lock=threading.Lock()
 def put(self,message):
  camera_id=str(message["camera_id"]);frame_id=int(message["frame_id"])
  with self._lock:
   bucket=self._items.setdefault(camera_id,OrderedDict());bucket[frame_id]=dict(message);bucket.move_to_end(frame_id)
   while len(bucket)>self.capacity:bucket.popitem(last=False)
 def clear(self):
  with self._lock:self._items.clear()
 def match(self,camera_id,frame_id,timestamp=None,independent_frame_domain=False):
  reference=float(timestamp if timestamp is not None else time.time())
  with self._lock:
   bucket=self._items.get(str(camera_id),{})
   if independent_frame_domain:
    candidates=[value for value in bucket.values() if float(value.get("timestamp",reference))<=reference];item=max(candidates,key=lambda value:float(value.get("timestamp",0.0))) if candidates else None
   else:
    candidates=[item for fid,item in bucket.items() if fid<=int(frame_id)];item=candidates[-1] if candidates else None
  if item is None:return None
  age=abs(reference-float(item.get("timestamp",reference))) if independent_frame_domain else reference-float(item.get("timestamp",reference))
  return item if 0<=age<=self.max_age else None
class VideoRenderer:
 def __init__(self,metadata=None):self.metadata=metadata or MetadataBuffer()
 def render(self,frame,camera_id=None,frame_id=None,timestamp=None):
  item=self.metadata.match(camera_id,frame_id,timestamp) if camera_id is not None and frame_id is not None else None
  return frame,(() if item is None else item.get("tracks",()))
