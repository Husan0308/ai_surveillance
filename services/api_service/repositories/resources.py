import json
from datetime import datetime,timezone
from uuid import uuid4
from fastapi import HTTPException
class ResourceRepository:
 def __init__(self,database,resource):self.database=database;self.resource=resource
 @staticmethod
 def _record(row):return None if row is None else {"id":str(row["id"]),**json.loads(row["data"] or "{}")}
 async def list(self):return [self._record(x) for x in await self.database.run(lambda db:db.execute("SELECT * FROM api_resources WHERE resource=? ORDER BY updated_at DESC",(self.resource,)).fetchall())]
 async def get(self,item_id):return self._record(await self.database.run(lambda db:db.execute("SELECT * FROM api_resources WHERE resource=? AND id=?",(self.resource,str(item_id))).fetchone()))
 async def create(self,payload):
  data=dict(payload);item_id=str(data.pop("id",uuid4()));now=datetime.now(timezone.utc).isoformat()
  try:await self.database.run(lambda db:db.execute("INSERT INTO api_resources(resource,id,name,data,created_at,updated_at) VALUES(?,?,?,?,?,?)",(self.resource,item_id,data.get("name"),json.dumps(data,default=str),now,now)))
  except Exception as exc:raise HTTPException(409,"Resource already exists") from exc
  return {"id":item_id,**data}
 async def update(self,item_id,payload):
  current=await self.get(item_id)
  if current is None:raise HTTPException(404,"Resource not found")
  merged={**current,**dict(payload)};merged.pop("id",None);now=datetime.now(timezone.utc).isoformat()
  await self.database.run(lambda db:db.execute("UPDATE api_resources SET name=?,data=?,updated_at=? WHERE resource=? AND id=?",(merged.get("name"),json.dumps(merged,default=str),now,self.resource,str(item_id))))
  return {"id":str(item_id),**merged}
 async def upsert(self,item_id,payload):return await self.update(item_id,payload) if await self.get(item_id) else await self.create({"id":item_id,**payload})
 async def delete(self,item_id):
  changed=await self.database.run(lambda db:db.execute("DELETE FROM api_resources WHERE resource=? AND id=?",(self.resource,str(item_id))).rowcount)
  if not changed:raise HTTPException(404,"Resource not found")
