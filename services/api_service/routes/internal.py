from fastapi import APIRouter,Request
from services.api_service.repositories import EnrollmentRepository,ResourceRepository
from services.api_service.websocket.routes import manager
from shared.event_taxonomy import is_persistent
router=APIRouter(prefix="/internal/ml",include_in_schema=False)
@router.post("/events",status_code=202)
@router.post("/realtime",status_code=202)
async def event(payload:dict,request:Request):
 data=dict(payload);kind=data.get("type","")
 if kind=="frame.metadata.batch":
  for message in data.get("messages",()):await manager.broadcast_json(message)
  return {"accepted":True,"broadcast":len(data.get("messages",()))}
 if kind.startswith("enrollment.") and data.get("session_id"):
  repo=EnrollmentRepository(request.app.state.database)
  if kind=="enrollment.completed":await repo.complete(data)
  else:await repo.upsert(data["session_id"],data)
 elif kind=="heatmap.updated":await ResourceRepository(request.app.state.database,"heatmaps").upsert(f"{data['camera_id']}:{data['mode'].upper()}",data)
 elif kind=="identity.merged" and data.get("identity_runtime_epoch"):
  alias_key=f"{data['identity_runtime_epoch']}:{data.get('old_global_id')}";await ResourceRepository(request.app.state.database,"identity_aliases").upsert(alias_key,data)
 elif is_persistent(data) or (kind=="event.created" and is_persistent(data)):
  await ResourceRepository(request.app.state.database,"events").create({**data,"event_type":data.get("event_type",kind)})
  if kind in ("camera.online","camera.offline"):request.app.state.ml_status=data
 elif kind=="system.metrics":request.app.state.ml_metrics=data
 elif kind=="system.status":request.app.state.ml_status=data
 await manager.broadcast_json({k:v for k,v in data.items() if k not in ("embedding","embeddings")});return {"accepted":True}
