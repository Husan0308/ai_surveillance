from datetime import datetime,timezone
from fastapi import HTTPException
from shared.schemas.messages import (CameraConfigChangedCommand,EnrollmentCancelCommand,
 EnrollmentStartCommand,MLSettingsChangedCommand)
from services.api_service.repositories import *

class PersonService:
    def __init__(self,db):self.repo=PersonRepository(db)
    async def list(self):return await self.repo.list()
    async def get(self,pid):return await self.repo.get(pid)
    async def create(self,data):return await self.repo.create(data)
    async def update(self,pid,data):return await self.repo.update(pid,{k:v for k,v in data.items() if v is not None})
    async def delete(self,pid):
        # Repository session provides the transaction boundary; dependent face/enrollment
        # rows use FK ON DELETE CASCADE in the canonical migration.
        await self.repo.delete(pid)

class EventService:
    def __init__(self,db):self.repo=EventRepository(db)
    async def list(self,**filters):
        items=await self.repo.list()
        for key,value in filters.items():
            if value is not None and key not in ("limit","offset","from_ts","to_ts"):
                items=[x for x in items if x.get(key)==value]
        if filters.get("from_ts"):items=[x for x in items if str(x.get("timestamp") or x.get("time") or "")>=filters["from_ts"]]
        if filters.get("to_ts"):items=[x for x in items if str(x.get("timestamp") or x.get("time") or "")<=filters["to_ts"]]
        start=int(filters.get("offset") or 0);return items[start:start+int(filters.get("limit") or 100)]
    async def get(self,eid):return await self.repo.get(eid)
    async def acknowledge(self,eid,user=None):return await self.repo.acknowledge(eid,user)

class CameraService:
    def __init__(self,db,ml):self.repo=CameraRepository(db);self.ml=ml
    async def list(self):return await self.repo.list()
    async def get(self,cid):return await self.repo.get(cid)
    async def create(self,data):
        item=await self.repo.create(data)
        try:await self.ml.command(CameraConfigChangedCommand(action="created",camera_id=item["id"],config=item))
        except Exception:await self.repo.delete(item["id"]);raise
        return item
    async def update(self,cid,data):
        old=await self.repo.get(cid);item=await self.repo.update(cid,data)
        try:await self.ml.command(CameraConfigChangedCommand(action="updated",camera_id=cid,config=item))
        except Exception:
            if old:await self.repo.update(cid,old)
            raise
        return item
    async def delete(self,cid):await self.ml.command(CameraConfigChangedCommand(action="deleted",camera_id=cid));await self.repo.delete(cid)

class EnrollmentService:
    def __init__(self,db,ml):self.repo=EnrollmentRepository(db);self.ml=ml
    async def start(self,data):
        cmd=EnrollmentStartCommand(**data);record=await self.repo.create({"id":cmd.session_id,**data,"status":"started","captured":0,"required":10})
        try:await self.ml.command(cmd)
        except Exception:await self.repo.update(cmd.session_id,{"status":"failed","error":"ML unavailable"});raise
        return record
    async def get(self,sid):return await self.repo.get(sid)
    async def cancel(self,sid):await self.ml.command(EnrollmentCancelCommand(session_id=sid));return await self.repo.update(sid,{"status":"cancelled"})

class SettingsService:
    def __init__(self,db,ml):self.repo=SettingsRepository(db);self.ml=ml
    async def get(self):return await self.repo.get("application") or {}
    async def update(self,data):
        import hashlib,secrets
        data=dict(data);password=data.pop("login_password",None)
        if password:
            salt=secrets.token_hex(16);digest=hashlib.pbkdf2_hmac("sha256",password.encode(),bytes.fromhex(salt),200000).hex()
            data["login_password_hash"]=f"pbkdf2_sha256$200000${salt}${digest}"
        current=await self.get();values={**current,**{k:v for k,v in data.items() if v is not None}};values.pop("id",None)
        ml={k:values[k] for k in ("detection_confidence","face_threshold","heatmap_enabled","heatmap_sample_interval_ms","tracking_enabled") if k in values};requires_restart=False
        if ml:await self.ml.command(MLSettingsChangedCommand(settings=ml,requires_restart=requires_restart))
        if current:result=await self.repo.update("application",values)
        else:result=await self.repo.create({"id":"application",**values})
        result.pop("login_password_hash",None);return {**result,"requires_restart":requires_restart}

class HeatmapService:
    def __init__(self,db):self.repo=HeatmapRepository(db)
    async def get(self,camera_id,mode):
        item=await self.repo.get(f"{camera_id}:{mode.upper()}")
        if item is None:raise HTTPException(404,"Heatmap not available")
        return item
