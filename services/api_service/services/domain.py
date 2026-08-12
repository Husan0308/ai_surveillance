from fastapi import HTTPException
import logging
from shared.schemas.messages import (CameraConfigChangedCommand,EnrollmentCancelCommand,
 EnrollmentStartCommand,MLSettingsChangedCommand)
from services.api_service.repositories import *
from shared.event_taxonomy import classify,event_type
log=logging.getLogger(__name__)

class PersonService:
    def __init__(self,db,ml=None):self.repo=PersonRepository(db);self.ml=ml
    async def list(self):return await self.repo.list()
    async def get(self,pid):return await self.repo.get(pid)
    async def create(self,data):return await self.repo.create(data)
    async def update(self,pid,data):
        item=await self.repo.update(pid,{k:v for k,v in data.items() if v is not None})
        if self.ml is not None:await self.ml.command({"type":"gallery.person.updated","person_id":pid,"name":item.get("name","Unknown")})
        return item
    async def delete(self,pid):
        def transaction(db):
            row=db.execute("DELETE FROM api_resources WHERE resource='persons' AND id=?",(str(pid),))
            if not row.rowcount:raise HTTPException(404,"Person not found")
            db.execute("DELETE FROM api_face_embeddings WHERE person_id=?",(str(pid),))
        await self.repo.database.run(transaction)
        if self.ml is not None:await self.ml.command({"type":"gallery.person.deleted","person_id":pid})

class EventService:
    def __init__(self,db):self.repo=EventRepository(db)
    async def list(self,**filters):
        items=await self.repo.list()
        for item in items:item["taxonomy_status"]=classify(event_type(item))
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
    @staticmethod
    def _public(item):
        if item is None:return None
        result=dict(item);result.pop("password",None);result.pop("username",None)
        for key in ("source","rtsp_url","ai_source","display_source"):
            value=result.get(key)
            if isinstance(value,str) and "://" in value and "@" in value:
                scheme,rest=value.split("://",1);userinfo,host=rest.rsplit("@",1);user=userinfo.split(":",1)[0];result[key]=scheme+"://"+user+":***@"+host
        return result
    async def list(self):return [self._public(item) for item in await self.repo.list()]
    async def get(self,cid):return self._public(await self.repo.get(cid))
    async def create(self,data):
        item=await self.repo.create(data)
        try:await self.ml.command(CameraConfigChangedCommand(action="created",camera_id=item["id"],config=item))
        except Exception:await self.repo.delete(item["id"]);raise
        return self._public(item)
    async def update(self,cid,data):
        item=await self.repo.update(cid,data)
        try:await self.ml.command(CameraConfigChangedCommand(action="updated",camera_id=cid,config=item))
        except Exception as exc:
            # SQLite is authoritative. A stopped ML service must not make an
            # operator calibration disappear; ML reconciles rows on startup.
            log.warning("Camera %s persisted; live ML notification deferred: %s",cid,exc)
        return self._public(item)
    async def delete(self,cid):await self.ml.command(CameraConfigChangedCommand(action="deleted",camera_id=cid));await self.repo.delete(cid)

class EnrollmentService:
    def __init__(self,db,ml):self.repo=EnrollmentRepository(db);self.ml=ml
    async def start(self,data):
        import asyncio
        from shared.enrollment_paths import validate_staged_paths,cleanup_staging
        data=dict(data)
        try:data["sample_paths"]=await asyncio.to_thread(validate_staged_paths,data.get("sample_paths",[]))
        except (OSError,ValueError) as exc:raise HTTPException(422,str(exc)) from exc
        await asyncio.to_thread(cleanup_staging)
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
