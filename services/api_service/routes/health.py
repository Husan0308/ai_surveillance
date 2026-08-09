from fastapi import APIRouter,Request
router=APIRouter(tags=["health"])
@router.get("/health")
async def health(request:Request):
    sqlite_ok=await request.app.state.database.ping();ml=await request.app.state.ml_client.health();ml_level=(ml or {}).get("status","unhealthy")
    dependencies={"sqlite":sqlite_ok,"ml_service":bool(ml)};request.app.state.dependencies=dependencies
    level="healthy" if sqlite_ok and ml_level=="healthy" else "degraded" if sqlite_ok else "unhealthy"
    return {"service":"api-service","status":level,"dependencies":dependencies,"components":{"database":"healthy" if sqlite_ok else "unhealthy","ml_service":ml_level},"ml":ml}
@router.get("/ready")
async def ready(request:Request):
    body=await health(request);return {**body,"ready":body["status"] in ("healthy","degraded") and body["dependencies"]["sqlite"]}
