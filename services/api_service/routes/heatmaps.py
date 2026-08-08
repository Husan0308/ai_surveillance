from fastapi import APIRouter,Request
from services.api_service.services import HeatmapService
router=APIRouter(prefix="/heatmaps",tags=["heatmaps"])
@router.get("/{camera_id}/{mode}")
async def heatmap(camera_id:str,mode:str,request:Request):
 if mode.lower() not in ("live","hourly","daily"):from fastapi import HTTPException;raise HTTPException(422,"Invalid heatmap mode")
 return await HeatmapService(request.app.state.database).get(camera_id,mode)
