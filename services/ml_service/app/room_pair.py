"""Room-pair identity endpoints kept separate from the six-camera pipeline."""
from __future__ import annotations

from fastapi import APIRouter

from services.mv3dt_room.room_pair_state import RoomPairState

router = APIRouter(prefix="/api/v1/room-pair", tags=["room-pair"])
state = RoomPairState()


@router.get("/identity")
def room_pair_identity() -> dict:
    return state.snapshot()


@router.get("/metrics")
def room_pair_metrics() -> dict:
    return state.snapshot()["metrics"]


@router.get("/acceptance-candidates")
def room_pair_acceptance_candidates() -> dict:
    """Read-only native-track candidates, called only by opt-in acceptance UI."""
    return state.acceptance_candidates()
