from fastapi import APIRouter,Request
from services.api_service.schemas import SettingsPatch
from services.api_service.services import SettingsService
router=APIRouter(prefix="/settings",tags=["settings"])
def svc(r):return SettingsService(r.app.state.database,r.app.state.ml_client)
@router.get("")
async def get(request:Request):return await svc(request).get()
@router.patch("")
async def update(payload:SettingsPatch,request:Request):return await svc(request).update(payload.model_dump(exclude_unset=True))
