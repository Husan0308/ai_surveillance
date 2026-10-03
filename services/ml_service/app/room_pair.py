"""Room-pair identity endpoints kept separate from the six-camera pipeline."""
from __future__ import annotations

from fastapi import APIRouter

from services.mv3dt_room.room_pair_state import RoomPairState
from services.shared.deployment import analytics_enabled

router = APIRouter(prefix="/api/v1/room-pair", tags=["room-pair"])
state = RoomPairState()


@router.get("/identity")
def room_pair_identity() -> dict:
    if not analytics_enabled():
        return {"people": [], "source_mode": "live", "status": "disabled", "metrics": {},
                "readiness": {"ready": False, "status": "disabled"}, "presence": {}}
    return state.snapshot()


@router.get("/metrics")
def room_pair_metrics() -> dict:
    return room_pair_identity()["metrics"]


@router.get("/acceptance-candidates")
def room_pair_acceptance_candidates() -> dict:
    """Read-only native-track candidates, called only by opt-in acceptance UI."""
    return state.acceptance_candidates() if analytics_enabled() else {"candidates": [], "status": "disabled"}
