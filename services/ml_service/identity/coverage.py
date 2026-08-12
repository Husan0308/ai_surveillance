"""Bounded, thread-safe observability for per-track ReID execution."""
from collections import OrderedDict
import threading,time

class ReIDTaskCoverage:
 def __init__(self,max_tracks=512):self.max_tracks=max(1,int(max_tracks));self._lock=threading.RLock();self._tracks=OrderedDict()
 def update(self,camera_id,track_id,**values):
  key=(str(camera_id),str(track_id))
  with self._lock:
   item=self._tracks.setdefault(key,{"camera_id":key[0],"local_track_id":key[1],"reid_eligible":False,"reid_task_submitted":False,"reid_embeddings_fresh":False,"independent_evidence_count":0,"candidate_count":0,"top1":None,"top2":None,"margin":None,"room_relation":None,"decision":"WAITING","reason":"not_evaluated","attempts":0,"completed":0})
   item.update(values);item["updated_at"]=time.time();self._tracks.move_to_end(key)
   while len(self._tracks)>self.max_tracks:self._tracks.popitem(last=False)
 def submitted(self,camera_id,track_id):
  key=(str(camera_id),str(track_id))
  with self._lock:attempts=int(self._tracks.get(key,{}).get("attempts",0))+1
  self.update(*key,reid_task_submitted=True,attempts=attempts,decision="EXTRACTING",reason="submitted")
 def completed(self,camera_id,track_id,quality,reason,usable,fresh,crop_width=0,crop_height=0,next_retry_at=None):
  key=(str(camera_id),str(track_id))
  with self._lock:completed=int(self._tracks.get(key,{}).get("completed",0))+1;evidence=int(self._tracks.get(key,{}).get("independent_evidence_count",0))+int(usable)
  self.update(*key,completed=completed,independent_evidence_count=evidence,reid_embeddings_fresh=bool(fresh),quality=round(float(quality),4),crop_width=int(crop_width),crop_height=int(crop_height),next_retry_at=next_retry_at,decision="EVIDENCE_READY" if usable else "RETRY",reason="fresh_embedding" if usable else str(reason))
 def prune(self,active_keys):
  active={(str(camera),str(track)) for camera,track in active_keys}
  with self._lock:
   for key in tuple(self._tracks):
    if key not in active:self._tracks.pop(key,None)
 def snapshot(self,decisions=None):
  decisions=decisions or {}
  with self._lock:
   output=[]
   for key,item in self._tracks.items():merged=dict(item);merged.update(decisions.get(key,{}));output.append(merged)
   return {"active_tracks":len(output),"eligible":sum(bool(item["reid_eligible"]) for item in output),"submitted":sum(bool(item["reid_task_submitted"]) for item in output),"tracks":output}
