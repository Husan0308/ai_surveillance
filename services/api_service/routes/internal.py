from fastapi import APIRouter,Request
from services.api_service.repositories import EnrollmentRepository,ResourceRepository
from services.api_service.websocket.routes import manager
router=APIRouter(prefix="/internal/ml",include_in_schema=False)
@router.post("/events",status_code=202)
async def event(payload:dict,request:Request):
 data=dict(payload);kind=data.get("type","")
 if kind.startswith("enrollment.") and data.get("session_id"):
  repo=EnrollmentRepository(request.app.state.database)
  if kind=="enrollment.completed":await repo.complete(data)
  else:await repo.upsert(data["session_id"],data)
 elif kind=="heatmap.updated":await ResourceRepository(request.app.state.database,"heatmaps").upsert(f"{data['camera_id']}:{data['mode'].upper()}",data)
 elif kind in ("event.created","person.identified"):await ResourceRepository(request.app.state.database,"events").create(data)
 elif kind=="system.metrics":request.app.state.ml_metrics=data
 elif kind in ("system.status","camera.online","camera.offline"):request.app.state.ml_status=data
 await manager.broadcast_json({k:v for k,v in data.items() if k!="embedding"});return {"accepted":True}
