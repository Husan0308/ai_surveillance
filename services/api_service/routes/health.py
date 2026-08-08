from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request):
    sqlite_ok = await request.app.state.database.ping()
    ml_ok = bool(await request.app.state.ml_client.health())
    dependencies = {"sqlite": sqlite_ok, "ml_service": ml_ok}
    request.app.state.dependencies = dependencies
    return {"service": "api-service", "status": "ok", "dependencies": dependencies}
