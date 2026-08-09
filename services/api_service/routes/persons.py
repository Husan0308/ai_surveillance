from fastapi import APIRouter,HTTPException,Request,status
from services.api_service.schemas import PersonCreate,PersonUpdate
from services.api_service.services import PersonService
router=APIRouter(prefix="/persons",tags=["persons"])
def svc(r):return PersonService(r.app.state.database,r.app.state.ml_client)
@router.get("")
async def list_persons(request:Request):return await svc(request).list()
@router.get("/{person_id}")
async def get_person(person_id:str,request:Request):
    item=await svc(request).get(person_id)
    if item is None:raise HTTPException(404,"Person not found")
    return item
@router.post("",status_code=status.HTTP_201_CREATED)
async def create_person(payload:PersonCreate,request:Request):return await svc(request).create(payload.model_dump())
@router.patch("/{person_id}")
async def update_person(person_id:str,payload:PersonUpdate,request:Request):return await svc(request).update(person_id,payload.model_dump(exclude_unset=True))
@router.delete("/{person_id}",status_code=204)
async def delete_person(person_id:str,request:Request):await svc(request).delete(person_id)
