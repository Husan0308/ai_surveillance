import threading
from .identity import GlobalIdentity

class IdentityStore:
    def __init__(self): self._lock=threading.RLock(); self._identities={}; self._bindings={}; self._next_id=1
    def create(self, observation):
        with self._lock:
            global_id=f"UNK-{self._next_id:06d}"; self._next_id+=1
            identity=GlobalIdentity(global_id,observation.timestamp,observation.timestamp,observation.camera_id,observation.local_track_id)
            self._identities[global_id]=identity; return identity
    def bind(self,camera_id,track_id,global_id):
        with self._lock: self._bindings[(camera_id,track_id)]=global_id
    def binding(self,camera_id,track_id):
        with self._lock: return self._bindings.get((camera_id,track_id))
    def identities(self):
        with self._lock: return tuple(self._identities.values())
    def get(self,global_id):
        with self._lock: return self._identities.get(global_id)
    def prune_archived(self,max_identities=2000):
        with self._lock:
            excess=len(self._identities)-max(1,int(max_identities))
            archived=sorted((item for item in self._identities.values() if str(getattr(item.status,"value",item.status))=="ARCHIVED"),key=lambda item:item.last_seen_at)
            removed={item.global_id for item in archived[:max(0,excess)]}
            for global_id in removed:self._identities.pop(global_id,None)
            for key,value in list(self._bindings.items()):
                if value in removed:self._bindings.pop(key,None)
            return len(removed)
    def remove_camera_bindings(self,camera_id):
        with self._lock:
            for key in [key for key in self._bindings if key[0]==camera_id]: self._bindings.pop(key,None)
