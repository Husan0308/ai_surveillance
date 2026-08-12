from fastapi import APIRouter,Request
router=APIRouter(prefix="/system",tags=["system"])
@router.get("/status")
async def status(request:Request):return {"api":"online","ml":getattr(request.app.state,"ml_status",{"status":"unknown"}),"dependencies":request.app.state.dependencies}
@router.get("/metrics")
async def metrics(request:Request):return getattr(request.app.state,"ml_metrics",{})
@router.get("/metrics/summary")
async def metrics_summary(request:Request):
 data=getattr(request.app.state,"ml_metrics",{}) or {};profile=data.get("detector_profile",{}) or {}
 return {"batch_rate":data.get("batch_rate"),"system":data.get("system",{}),"detector_profile":{"pure_detector_wall":profile.get("pure_detector_wall",{})}}
