from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PoseKeypoint:
    x: float
    y: float
    confidence: float


@dataclass(frozen=True, slots=True)
class PosePerson:
    bbox: tuple[float, float, float, float]
    confidence: float
    keypoints: tuple[PoseKeypoint, ...]


@dataclass(frozen=True, slots=True)
class PoseResult:
    camera_id: str
    frame_id: int
    frame_captured_monotonic: float
    produced_monotonic: float
    people: tuple[PosePerson, ...]
