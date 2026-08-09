from typing import Any,Literal
from pydantic import BaseModel,Field

Codec=Literal["h264","h265","hevc"]
class PersonCreate(BaseModel):
    name:str=Field(min_length=1,max_length=255);department:str|None=None;employee_id:str|None=None
    status:str="active";notes:str|None=None;enabled:bool=True;profile_image:str|None=None;metadata:dict[str,Any]=Field(default_factory=dict)
class PersonUpdate(BaseModel):
    name:str|None=Field(None,min_length=1,max_length=255);department:str|None=None;employee_id:str|None=None
    status:str|None=None;notes:str|None=None;enabled:bool|None=None;profile_image:str|None=None;metadata:dict[str,Any]|None=None
class CameraCreate(BaseModel):
    id:str=Field(pattern=r"^[A-Za-z0-9_-]+$");name:str=Field(min_length=1);source:str=Field(min_length=1)
    ai_source:str|None=Field(None,min_length=1);display_source:str|None=Field(None,min_length=1)
    codec:Codec;ai_codec:Codec|None=None;display_codec:Codec|None=None
    enabled:bool=True;room_id:str|None=None;heatmap_enabled:bool=True
    latency_ms:int=Field(default=50,ge=0,le=5000);decoder_backend:Literal["nvv4l2decoder","nvcodec"]="nvv4l2decoder"
class CameraUpdate(BaseModel):
    name:str|None=None;source:str|None=None;ai_source:str|None=None;display_source:str|None=None
    codec:Codec|None=None;ai_codec:Codec|None=None;display_codec:Codec|None=None
    enabled:bool|None=None;room_id:str|None=None;heatmap_enabled:bool|None=None
    latency_ms:int|None=Field(None,ge=0,le=5000);decoder_backend:Literal["nvv4l2decoder","nvcodec"]|None=None
class EnrollmentCreate(BaseModel):
    name:str=Field(min_length=1);sample_paths:list[str]=Field(min_length=10,max_length=30);department:str|None=None
class SettingsPatch(BaseModel):
    detection_confidence:float|None=Field(None,ge=0,le=1);face_threshold:float|None=Field(None,ge=0,le=1)
    heatmap_enabled:bool|None=None;heatmap_sample_interval_ms:int|None=Field(None,ge=10,le=60000)
    tracking_enabled:bool|None=None;retention_days:int|None=Field(None,ge=1,le=3650)
    sound_enabled:bool|None=None;auto_lock_minutes:int|None=Field(None,ge=1,le=120)
    login_password:str|None=Field(None,min_length=8,max_length=256)
