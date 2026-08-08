from typing import Any
from pydantic import BaseModel,Field

class PersonCreate(BaseModel):
    name:str=Field(min_length=1,max_length=255);department:str|None=None;employee_id:str|None=None
    status:str="active";notes:str|None=None;enabled:bool=True;profile_image:str|None=None;metadata:dict[str,Any]=Field(default_factory=dict)
class PersonUpdate(BaseModel):
    name:str|None=Field(None,min_length=1,max_length=255);department:str|None=None;employee_id:str|None=None
    status:str|None=None;notes:str|None=None;enabled:bool|None=None;profile_image:str|None=None;metadata:dict[str,Any]|None=None
class CameraCreate(BaseModel):
    id:str=Field(pattern=r"^[A-Za-z0-9_-]+$");name:str=Field(min_length=1);source:str=Field(min_length=1)
    enabled:bool=True;room_id:str|None=None;heatmap_enabled:bool=True
class CameraUpdate(BaseModel):
    name:str|None=None;source:str|None=None;enabled:bool|None=None;room_id:str|None=None;heatmap_enabled:bool|None=None
class EnrollmentCreate(BaseModel):
    name:str=Field(min_length=1);camera_id:str=Field(pattern=r"^[A-Za-z0-9_-]+$");department:str|None=None
class SettingsPatch(BaseModel):
    detection_confidence:float|None=Field(None,ge=0,le=1);face_threshold:float|None=Field(None,ge=0,le=1)
    heatmap_enabled:bool|None=None;heatmap_sample_interval_ms:int|None=Field(None,ge=10,le=60000)
    tracking_enabled:bool|None=None;retention_days:int|None=Field(None,ge=1,le=3650)
    sound_enabled:bool|None=None;auto_lock_minutes:int|None=Field(None,ge=1,le=120)
    login_password:str|None=Field(None,min_length=8,max_length=256)
