from fastapi import APIRouter,HTTPException,Request,Query
from services.api_service.services import EventService
router=APIRouter(prefix="/events",tags=["events"])
def svc(r):return EventService(r.app.state.database)
@router.get("")
async def events(request:Request,camera_id:str|None=None,person_id:str|None=None,event_type:str|None=None,acknowledged:bool|None=None,
 from_ts:str|None=None,to_ts:str|None=None,limit:int=Query(100,ge=1,le=1000),offset:int=Query(0,ge=0)):
 return await svc(request).list(camera_id=camera_id,person_id=person_id,event_type=event_type,acknowledged=acknowledged,from_ts=from_ts,to_ts=to_ts,limit=limit,offset=offset)
@router.get("/{event_id}")
async def event(event_id:str,request:Request):
 item=await svc(request).get(event_id)
 if item is None:raise HTTPException(404,"Event not found")
 return item
@router.post("/{event_id}/acknowledge")
async def acknowledge(event_id:str,request:Request):return await svc(request).acknowledge(event_id)
