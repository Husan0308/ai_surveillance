from fastapi import APIRouter,HTTPException,Request,status
from services.api_service.schemas import EnrollmentCreate
from services.api_service.services import EnrollmentService
router=APIRouter(prefix="/enrollment/sessions",tags=["enrollment"])
def svc(r):return EnrollmentService(r.app.state.database,r.app.state.ml_client)
@router.post("",status_code=status.HTTP_201_CREATED)
async def start(payload:EnrollmentCreate,request:Request):return await svc(request).start(payload.model_dump())
@router.get("/{session_id}")
async def get(session_id:str,request:Request):
 item=await svc(request).get(session_id)
 if item is None:raise HTTPException(404,"Enrollment session not found")
 return item
@router.post("/{session_id}/cancel")
async def cancel(session_id:str,request:Request):return await svc(request).cancel(session_id)
