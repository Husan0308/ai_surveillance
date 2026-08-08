from enum import Enum


class CameraStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"


class EventType(str, Enum):
    PERSON_ENTERED = "person_entered"
    PERSON_LEFT = "person_left"
    PERSON_RECOGNIZED = "person_recognized"
    UNKNOWN_PERSON = "unknown_person"
    CAMERA_STATUS = "camera_status"
    SYSTEM = "system"


class EventSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
