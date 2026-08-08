import threading,time,numpy as np
from .schemas import KnownPersonIdentity

class GalleryRepository:
    def load(self):return []
    def save(self,person):raise NotImplementedError

class InMemoryGalleryRepository(GalleryRepository):
    def __init__(self):self.people={}
    def load(self):return list(self.people.values())
    def save(self,person):self.people[person.person_id]=person

class KnownPersonGallery:
    def __init__(self,repository=None,max_embeddings=20):self.repository=repository or InMemoryGalleryRepository();self.max_embeddings=max_embeddings;self._lock=threading.RLock();self.people={p.person_id:p for p in self.repository.load()}
    def add(self,person_id,name,embeddings):
        normalized=[]
        for emb in embeddings:
            value=np.asarray(emb,np.float32);norm=np.linalg.norm(value)
            if norm:normalized.append(value/norm)
        now=time.time();person=self.people.get(person_id) or KnownPersonIdentity(person_id,name,[],now,now)
        person.name=name;person.embeddings=(person.embeddings+normalized)[-self.max_embeddings:];person.updated_at=now;self.people[person_id]=person;self.repository.save(person);return person
    def enabled(self):return [person for person in self.people.values() if person.enabled and person.embeddings]
