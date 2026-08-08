from fastapi import APIRouter,Request
router=APIRouter(prefix="/system",tags=["system"])
@router.get("/status")
async def status(request:Request):return {"api":"online","ml":getattr(request.app.state,"ml_status",{"status":"unknown"}),"dependencies":request.app.state.dependencies}
@router.get("/metrics")
async def metrics(request:Request):return getattr(request.app.state,"ml_metrics",{})
