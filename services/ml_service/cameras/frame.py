"""Structured frame contracts used only inside the ML service process."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FramePacket:
    camera_id: str
    frame_id: int
    capture_timestamp: float
    receive_timestamp: float
    frame: Any
    width: int
    height: int
    scheduler_selected_timestamp: float = 0.0
