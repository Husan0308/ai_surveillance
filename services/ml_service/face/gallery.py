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
    def remove(self,person_id):
        with self._lock:self.people.pop(str(person_id),None)
    def update_name(self,person_id,name):
        with self._lock:
            person=self.people.get(str(person_id))
            if person:person.name=name
    def reload(self):
        with self._lock:self.people={p.person_id:p for p in self.repository.load()}
        return len(self.people)


class SQLiteGalleryRepository(GalleryRepository):
    """Read-only ML view of the API-owned canonical SQLite person gallery."""
    def __init__(self,path,model_version="buffalo_l:w600k_r50",embedding_dimension=512):
        self.path=str(path);self.model_version=model_version;self.dimension=int(embedding_dimension)
    def load(self):
        import json,sqlite3
        from array import array
        people={}
        with sqlite3.connect(self.path,timeout=5) as db:
            db.row_factory=sqlite3.Row
            for row in db.execute("SELECT id,data,created_at,updated_at FROM api_resources WHERE resource='persons'"):
                data=json.loads(row["data"] or "{}")
                if not bool(data.get("enabled",True)) or str(data.get("status","active")).lower() in ("deleted","inactive"):continue
                people[str(row["id"])]=KnownPersonIdentity(str(row["id"]),data.get("name") or "Unknown",[],0,0,True)
            columns={row[1] for row in db.execute("PRAGMA table_info(api_face_embeddings)")}
            select="SELECT person_id,embedding"+(",dimension" if "dimension" in columns else "")+(",model_version" if "model_version" in columns else "")+" FROM api_face_embeddings"+(" WHERE enabled=1" if "enabled" in columns else "")
            for row in db.execute(select):
                person=people.get(str(row["person_id"]))
                if person is None:continue
                dimension=int(row["dimension"]) if "dimension" in columns and row["dimension"] else len(row["embedding"])//4
                version=row["model_version"] if "model_version" in columns else self.model_version
                if dimension!=self.dimension or version!=self.model_version:continue
                values=array("f");values.frombytes(row["embedding"])
                if len(values)!=self.dimension:continue
                value=np.asarray(values,np.float32);norm=np.linalg.norm(value)
                if norm:person.embeddings.append(value/norm)
        return list(people.values())
    def save(self,person):
        # API owns writes; enrollment completion persists before gallery reload.
        return None
