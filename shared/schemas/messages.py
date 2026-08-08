from datetime import datetime,timezone
from typing import Any,Literal
from uuid import uuid4
from pydantic import BaseModel,Field,model_validator

def utcnow():return datetime.now(timezone.utc)

class Message(BaseModel):
    type:str
    timestamp:datetime=Field(default_factory=utcnow)
    @model_validator(mode="after")
    def metadata_only(self):
        def check(value):
            if isinstance(value,(bytes,bytearray,memoryview)):raise ValueError("Binary data is forbidden in service messages")
            if isinstance(value,dict):
                for key,child in value.items():
                    if str(key).lower() in {"frame","image","video","frame_bytes","image_bytes"}:raise ValueError("Frame/image fields are forbidden in service messages")
                    check(child)
            elif isinstance(value,(list,tuple)):
                for child in value:check(child)
        check(self.__dict__);return self

class EnrollmentStartCommand(Message):
    type:Literal["enrollment.start"]="enrollment.start"
    session_id:str=Field(default_factory=lambda:str(uuid4()))
    name:str=Field(min_length=1,max_length=255)
    camera_id:str=Field(pattern=r"^[A-Za-z0-9_-]+$")
    department:str|None=None

class EnrollmentCancelCommand(Message):
    type:Literal["enrollment.cancel"]="enrollment.cancel";session_id:str

class EnrollmentProgressEvent(Message):
    type:Literal["enrollment.progress"]="enrollment.progress";session_id:str;captured:int=Field(ge=0);required:int=Field(gt=0);quality:float=Field(ge=0,le=1);message:str|None=None

class EnrollmentCompletedEvent(Message):
    type:Literal["enrollment.completed"]="enrollment.completed";session_id:str;person_id:str

class CameraConfigChangedCommand(Message):
    type:Literal["camera.config.changed"]="camera.config.changed";action:Literal["created","updated","deleted"];camera_id:str;config:dict[str,Any]=Field(default_factory=dict)

class MLSettingsChangedCommand(Message):
    type:Literal["settings.ml.updated"]="settings.ml.updated";settings:dict[str,Any];requires_restart:bool=False

class CameraStatusEvent(Message):
    type:Literal["camera.online","camera.offline"];camera_id:str;details:dict[str,Any]=Field(default_factory=dict)

class PersonIdentifiedEvent(Message):
    type:Literal["person.identified"]="person.identified";camera_id:str;person_id:str;track_id:str|None=None;name:str|None=None
