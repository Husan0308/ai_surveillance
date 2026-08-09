from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel,ConfigDict, Field

from shared.enums import CameraStatus, EventSeverity, EventType


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class ContractModel(BaseModel):
    model_config=ConfigDict(use_enum_values=True,from_attributes=True)


class Camera(ContractModel):
    id: str
    name: str
    room_id: Optional[str] = None
    source: Optional[str] = Field(default=None, description="Legacy RTSP source; API responses redact credentials")
    ai_source: Optional[str] = Field(default=None, description="Inference RTSP source")
    display_source: Optional[str] = Field(default=None, description="Optional on-demand display RTSP source")
    status: CameraStatus = CameraStatus.OFFLINE
    ai_enabled: bool = True
    heatmap_enabled: bool = False
    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)


class Room(ContractModel):
    id: str
    name: str
    description: Optional[str] = None
    floor: Optional[str] = None
    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)


class Detection(ContractModel):
    camera_id: str
    frame_id: str
    class_name: str = "person"
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: Tuple[float, float, float, float]
    track_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=now_utc)


class Track(ContractModel):
    id: str
    camera_id: str
    person_id: Optional[str] = None
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    first_seen: datetime = Field(default_factory=now_utc)
    last_seen: datetime = Field(default_factory=now_utc)


class Person(ContractModel):
    id: Optional[str] = None
    name: str
    employee_id: Optional[str] = None
    department: Optional[str] = None
    status: str = "active"
    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)


class HeatmapPoint(ContractModel):
    camera_id: str
    room_id: Optional[str] = None
    track_id: Optional[str] = None
    x: float
    y: float
    weight: float = Field(default=1.0, ge=0.0)
    timestamp: datetime = Field(default_factory=now_utc)


class CameraMetrics(ContractModel):
    camera_id: str
    status: CameraStatus = CameraStatus.OFFLINE
    capture_fps: float = 0.0
    inference_fps: float = 0.0
    latency_ms: float = 0.0
    active_tracks: int = 0
    dropped_frames: int = 0
    timestamp: datetime = Field(default_factory=now_utc)


class SystemMetrics(ContractModel):
    service: str = "ml-service"
    gpu_utilization_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    cameras_online: int = 0
    timestamp: datetime = Field(default_factory=now_utc)


class Event(ContractModel):
    id: UUID = Field(default_factory=uuid4)
    type: EventType
    severity: EventSeverity = EventSeverity.INFO
    camera_id: Optional[str] = None
    room_id: Optional[str] = None
    person_id: Optional[str] = None
    track_id: Optional[str] = None
    message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=now_utc)


class Events(ContractModel):
    items: List[Event] = Field(default_factory=list)
    total: int = 0
