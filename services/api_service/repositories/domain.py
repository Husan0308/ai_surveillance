import json
from array import array
from datetime import datetime,timezone
from .resources import ResourceRepository

class PersonRepository(ResourceRepository):
 def __init__(self,db):super().__init__(db,"persons")
class CameraRepository(ResourceRepository):
 def __init__(self,db):super().__init__(db,"cameras")
class SettingsRepository(ResourceRepository):
 def __init__(self,db):super().__init__(db,"settings")
class HeatmapRepository(ResourceRepository):
 def __init__(self,db):super().__init__(db,"heatmaps")
class EventRepository(ResourceRepository):
 def __init__(self,db):super().__init__(db,"events")
 async def acknowledge(self,event_id,acknowledged_by=None):return await self.update(event_id,{"acknowledged":True,"acknowledged_at":datetime.now(timezone.utc).isoformat(),"acknowledged_by":acknowledged_by})
class EnrollmentRepository(ResourceRepository):
 def __init__(self,db):super().__init__(db,"enrollment_sessions")
 async def complete(self,payload):
  public={k:v for k,v in payload.items() if k not in ("embedding","embeddings")};pid=str(payload["person_id"])
  def transaction(db):
   current=db.execute("SELECT data FROM api_resources WHERE resource='enrollment_sessions' AND id=?",(payload["session_id"],)).fetchone();data=json.loads(current[0]) if current else {}
   person_data={"name":payload.get("name","Unknown"),"department":payload.get("department"),"status":"active","enabled":True}
   db.execute("INSERT INTO api_resources(resource,id,name,data) VALUES('persons',?,?,?) ON CONFLICT(resource,id) DO UPDATE SET name=excluded.name,data=excluded.data,updated_at=CURRENT_TIMESTAMP",(pid,person_data["name"],json.dumps(person_data)))
   db.execute("DELETE FROM api_face_embeddings WHERE person_id=?",(pid,))
   dimension=int(payload.get("dimension",512));version=str(payload.get("model_version","buffalo_l:w600k_r50"))
   for sample in payload.get("embeddings",()):
    values=list(sample.get("embedding",()))
    if len(values)!=dimension:continue
    db.execute("INSERT INTO api_face_embeddings(person_id,embedding,quality,dimension,model_version,source_metadata,enabled) VALUES(?,?,?,?,?,?,1)",(pid,array("f",values).tobytes(),sample.get("quality"),dimension,version,json.dumps(sample.get("source_metadata") or {},default=str)))
   merged={**data,**public,"status":"completed"}
   db.execute("INSERT INTO api_resources(resource,id,name,data) VALUES('enrollment_sessions',?,?,?) ON CONFLICT(resource,id) DO UPDATE SET data=excluded.data,updated_at=CURRENT_TIMESTAMP",(payload["session_id"],None,json.dumps(merged,default=str)))
  await self.database.run(transaction);return public
