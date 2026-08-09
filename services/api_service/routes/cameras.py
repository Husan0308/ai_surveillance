from fastapi import APIRouter,HTTPException,Request,status
from services.api_service.schemas import CameraCreate,CameraUpdate
from services.api_service.services import CameraService
from services.api_service.websocket.routes import manager
router=APIRouter(prefix="/cameras",tags=["cameras"])
def svc(r):return CameraService(r.app.state.database,r.app.state.ml_client)
@router.get("")
async def cameras(request:Request):return await svc(request).list()
@router.get("/{camera_id}")
async def camera(camera_id:str,request:Request):
 item=await svc(request).get(camera_id)
 if item is None:raise HTTPException(404,"Camera not found")
 return item
@router.post("",status_code=status.HTTP_201_CREATED)
async def create(payload:CameraCreate,request:Request):
 item=await svc(request).create(payload.model_dump());await manager.broadcast_json({"type":"camera.config.changed","action":"created","camera_id":item["id"]});return item
@router.patch("/{camera_id}")
async def update(camera_id:str,payload:CameraUpdate,request:Request):
 item=await svc(request).update(camera_id,payload.model_dump(exclude_unset=True));await manager.broadcast_json({"type":"camera.config.changed","action":"updated","camera_id":camera_id});return item
@router.delete("/{camera_id}",status_code=204)
async def delete(camera_id:str,request:Request):
 await svc(request).delete(camera_id);await manager.broadcast_json({"type":"camera.config.changed","action":"deleted","camera_id":camera_id})
