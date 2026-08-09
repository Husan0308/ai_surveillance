from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any

class SecondaryTaskType(str, Enum):
    REID = "REID"
    FACE = "FACE"
    POSE = "POSE"

@dataclass(frozen=True, slots=True)
class SecondaryTask:
    task_type: SecondaryTaskType
    camera_id: str
    local_track_id: str
    global_id: str | None
    frame_id: int
    capture_timestamp: float
    bbox: tuple[float, float, float, float]
    crop: Any
    priority: int = 0
    context: Any = None
