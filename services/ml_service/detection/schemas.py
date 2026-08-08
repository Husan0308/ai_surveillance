"""Structured person-detection output contracts."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0
    class_name: str = "person"

@dataclass(frozen=True, slots=True)
class CameraDetectionResult:
    camera_id: str
    frame_id: int
    capture_timestamp: float
    receive_timestamp: float
    detections: tuple[Detection, ...]

@dataclass(frozen=True, slots=True)
class DetectionBatchResult:
    batch_id: int
    started_at: float
    completed_at: float
    results: tuple[CameraDetectionResult, ...]
