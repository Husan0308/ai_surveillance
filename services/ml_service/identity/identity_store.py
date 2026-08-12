import threading
from .identity import GlobalIdentity

class IdentityStore:
    def __init__(self):
        self._lock=threading.RLock();self._identities={};self._bindings={};self._aliases={};self._next_id=1;self._version=0

    def create(self,observation):
        with self._lock:
            global_id=f"UNK {self._next_id}";self._next_id+=1
            identity=GlobalIdentity(global_id,observation.timestamp,observation.timestamp,observation.camera_id,observation.local_track_id,display_name=global_id)
            self._identities[global_id]=identity
            return identity

    def canonicalize(self,global_id):
        if global_id is None:return None
        with self._lock:
            current=str(global_id);visited=[]
            while current in self._aliases and self._aliases[current]!=current:
                if current in visited:raise RuntimeError("identity alias cycle")
                visited.append(current);current=self._aliases[current]
            for alias in visited:self._aliases[alias]=current
            return current

    @property
    def version(self):
        with self._lock:return self._version

    def aliases(self):
        with self._lock:return {key:self.canonicalize(value) for key,value in self._aliases.items()}

    def bind(self,camera_id,track_id,global_id):
        with self._lock:self._bindings[(camera_id,track_id)]=self.canonicalize(global_id)

    def binding(self,camera_id,track_id):
        with self._lock:return self.canonicalize(self._bindings.get((camera_id,track_id)))

    def identities(self):
        with self._lock:return tuple(self._identities.values())

    def get(self,global_id):
        with self._lock:return self._identities.get(self.canonicalize(global_id))

    def merge(self,first_id,second_id,max_history=20):
        """Merge two runtime identities into the older canonical identity."""
        with self._lock:
            first=self._identities.get(first_id);second=self._identities.get(second_id)
            if first is None or second is None:return first or second
            canonical,duplicate=sorted((first,second),key=lambda item:(item.created_at,self._ordinal(item.global_id)))[0],None
            duplicate=second if canonical is first else first
            canonical_seen=canonical.last_seen_at;duplicate_seen=duplicate.last_seen_at
            if duplicate_seen>=canonical_seen:
                canonical.last_camera_id=duplicate.last_camera_id;canonical.last_local_track_id=duplicate.last_local_track_id
            canonical.last_seen_at=max(canonical_seen,duplicate_seen)
            canonical.active_tracks.update(duplicate.active_tracks);canonical.active_track_seen.update(duplicate.active_track_seen)
            canonical.camera_history.extend(duplicate.camera_history);canonical.camera_history.sort(key=lambda item:item[-1])
            canonical.track_history.extend(duplicate.track_history);canonical.track_history.sort(key=lambda item:item[-1])
            for embedding,quality,stamp in duplicate.appearance_history:
                canonical.last_seen_at=max(canonical.last_seen_at,stamp);canonical.add_embedding(embedding,quality,max(1,int(max_history)))
            if canonical.person_id is None and duplicate.person_id is not None:
                canonical.person_id=duplicate.person_id;canonical.display_name=duplicate.display_name
            duplicate_id=duplicate.global_id;canonical_id=self.canonicalize(canonical.global_id)
            self._aliases[duplicate_id]=canonical_id
            for alias,target in list(self._aliases.items()):
                if target==duplicate_id:self._aliases[alias]=canonical_id
            for key,value in list(self._bindings.items()):
                self._bindings[key]=self.canonicalize(value)
            self._identities.pop(duplicate_id,None);self._version+=1
            return canonical

    @staticmethod
    def _ordinal(global_id):
        try:return int(str(global_id).rsplit(" ",1)[-1])
        except ValueError:return 2**63-1

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
            for key in [key for key in self._bindings if key[0]==camera_id]:self._bindings.pop(key,None)
