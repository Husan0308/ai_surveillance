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
 def match(self,camera_id,frame_id,timestamp=None):
  reference=float(timestamp if timestamp is not None else time.time())
  with self._lock:
   bucket=self._items.get(str(camera_id),{});candidates=[item for fid,item in bucket.items() if fid<=int(frame_id)];item=candidates[-1] if candidates else None
  if item is None:return None
  age=reference-float(item.get("timestamp",reference));return item if 0<=age<=self.max_age else None
class VideoRenderer:
 def __init__(self,metadata=None):self.metadata=metadata or MetadataBuffer()
 def render(self,frame,camera_id=None,frame_id=None,timestamp=None):
  item=self.metadata.match(camera_id,frame_id,timestamp) if camera_id is not None and frame_id is not None else None
  return frame,(() if item is None else item.get("tracks",()))
