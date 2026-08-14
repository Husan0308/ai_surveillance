from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import threading
import time


@dataclass
class _PendingVisit:
    global_id: str
    room_id: str
    first_seen: float
    last_seen: float
    camera_id: str
    cameras: set[str] = field(default_factory=set)


@dataclass
class _ActiveVisit:
    session_id: int
    global_id: str
    room_id: str
    entered_monotonic: float
    last_seen: float
    entered_at: str
    last_seen_at: str
    camera_id: str
    cameras: set[str] = field(default_factory=set)


class RoomVisitSessionManager:
    """Turn noisy per-camera ReID observations into one room visit session.

    Same-room cameras do not create separate events. A visit must remain stable
    for ``enter_confirm_sec`` before an ENTER event is emitted; this gives the
    same-room ReID pair matcher time to merge provisional identities first.
    Short detector/ReID gaps keep the existing room session alive.
    """

    def __init__(self, config: dict | None = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.enter_confirm_sec = max(0.0, float(cfg.get("enter_confirm_sec", 0.8)))
        self.inactive_timeout_sec = max(0.5, float(cfg.get("inactive_timeout_sec", 4.0)))
        self.pending_timeout_sec = max(
            self.enter_confirm_sec + 0.1,
            float(cfg.get("pending_timeout_sec", 2.0)),
        )
        self.max_events = max(10, int(cfg.get("max_events", 100)))
        self.max_recent = max(10, int(cfg.get("max_recent_sessions", 100)))

        self._lock = threading.RLock()
        self._pending: dict[tuple[str, str], _PendingVisit] = {}
        self._active: dict[str, _ActiveVisit] = {}
        self._recent: deque[dict] = deque(maxlen=self.max_recent)
        self._events: deque[dict] = deque(maxlen=self.max_events)
        self._next_session_id = 1
        self._created = 0
        self._room_changes = 0
        self._closed = 0
        self._suppressed_observations = 0

    @staticmethod
    def _wall_now() -> tuple[str, str]:
        now = datetime.now().astimezone()
        return now.isoformat(timespec="seconds"), now.strftime("%H:%M:%S")

    @staticmethod
    def _latest_room_observations(reid_state: dict, now: float, max_age: float) -> dict[str, dict]:
        observations: dict[str, dict] = {}
        cameras = (reid_state or {}).get("cameras") or {}
        for camera_id, tracks in cameras.items():
            for track in tracks or []:
                gid = str(track.get("global_id") or "").strip()
                room_id = str(track.get("room_id") or "").strip()
                if not gid or not room_id or room_id.lower() == "unknown":
                    continue
                try:
                    last_seen = float(track.get("last_seen"))
                except (TypeError, ValueError):
                    continue
                if now - last_seen > max_age:
                    continue
                current = observations.get(gid)
                if current is None or last_seen > current["last_seen"]:
                    observations[gid] = {
                        "global_id": gid,
                        "room_id": room_id,
                        "last_seen": last_seen,
                        "camera_id": str(camera_id),
                        "cameras": {str(camera_id)},
                    }
                elif current["room_id"] == room_id:
                    current["cameras"].add(str(camera_id))
                    if last_seen > current["last_seen"]:
                        current["last_seen"] = last_seen
                        current["camera_id"] = str(camera_id)
        return observations

    def _close_locked(self, gid: str, closed_at_monotonic: float) -> None:
        visit = self._active.pop(gid, None)
        if visit is None:
            return
        timestamp, clock = self._wall_now()
        duration = max(0.0, float(closed_at_monotonic) - visit.entered_monotonic)
        self._recent.appendleft({
            "session_id": visit.session_id,
            "global_id": visit.global_id,
            "room_id": visit.room_id,
            "entered_at": visit.entered_at,
            "entered_time": visit.entered_at[11:19] if len(visit.entered_at) >= 19 else "",
            "last_seen_at": visit.last_seen_at,
            "closed_at": timestamp,
            "closed_time": clock,
            "duration_sec": round(duration, 3),
            "camera": visit.camera_id,
            "cameras": sorted(visit.cameras),
        })
        self._closed += 1

    def _start_locked(self, pending: _PendingVisit, now: float) -> _ActiveVisit:
        timestamp, clock = self._wall_now()
        visit = _ActiveVisit(
            session_id=self._next_session_id,
            global_id=pending.global_id,
            room_id=pending.room_id,
            entered_monotonic=now,
            last_seen=pending.last_seen,
            entered_at=timestamp,
            last_seen_at=timestamp,
            camera_id=pending.camera_id,
            cameras=set(pending.cameras),
        )
        self._next_session_id += 1
        self._active[pending.global_id] = visit
        event = {
            "event_id": f"room-enter-{visit.session_id}",
            "session_id": visit.session_id,
            "type": "room_enter",
            "time": clock,
            "timestamp": timestamp,
            "global_id": visit.global_id,
            "room_id": visit.room_id,
            "camera": visit.camera_id,
            "cameras": sorted(visit.cameras),
            "reason": "room_visit_started",
        }
        self._events.appendleft(event)
        self._created += 1
        return visit

    def update(self, reid_state: dict | None, now: float | None = None) -> None:
        if not self.enabled:
            return
        now = float(time.monotonic() if now is None else now)
        observations = self._latest_room_observations(
            dict(reid_state or {}), now, self.inactive_timeout_sec
        )
        with self._lock:
            for key, pending in list(self._pending.items()):
                if now - pending.last_seen > self.pending_timeout_sec:
                    self._pending.pop(key, None)

            observed_gids = set(observations)
            for gid, observation in observations.items():
                room_id = observation["room_id"]
                active = self._active.get(gid)
                if active is not None and active.room_id == room_id:
                    active.last_seen = max(active.last_seen, observation["last_seen"])
                    active.camera_id = observation["camera_id"]
                    active.cameras.update(observation["cameras"])
                    active.last_seen_at = self._wall_now()[0]
                    self._suppressed_observations += 1
                    continue

                key = (gid, room_id)
                pending = self._pending.get(key)
                if pending is None:
                    pending = _PendingVisit(
                        global_id=gid,
                        room_id=room_id,
                        first_seen=now,
                        last_seen=observation["last_seen"],
                        camera_id=observation["camera_id"],
                        cameras=set(observation["cameras"]),
                    )
                    self._pending[key] = pending
                else:
                    pending.last_seen = max(pending.last_seen, observation["last_seen"])
                    pending.camera_id = observation["camera_id"]
                    pending.cameras.update(observation["cameras"])

                if now - pending.first_seen < self.enter_confirm_sec:
                    continue

                if active is not None and active.room_id != room_id:
                    self._close_locked(gid, now)
                    self._room_changes += 1
                self._start_locked(pending, now)
                for pending_key in [item for item in self._pending if item[0] == gid]:
                    self._pending.pop(pending_key, None)

            for gid, active in list(self._active.items()):
                if gid in observed_gids:
                    continue
                if now - active.last_seen > self.inactive_timeout_sec:
                    self._close_locked(gid, now)

    def snapshot(self) -> dict:
        with self._lock:
            active = [
                {
                    "session_id": visit.session_id,
                    "global_id": visit.global_id,
                    "room_id": visit.room_id,
                    "entered_at": visit.entered_at,
                    "time": visit.entered_at[11:19] if len(visit.entered_at) >= 19 else "",
                    "last_seen_at": visit.last_seen_at,
                    "camera": visit.camera_id,
                    "cameras": sorted(visit.cameras),
                }
                for visit in self._active.values()
            ]
            return {
                "enabled": self.enabled,
                "active_sessions": sorted(active, key=lambda item: item["session_id"]),
                "recent_sessions": list(self._recent),
                "events": list(self._events),
                "metrics": {
                    "active": len(self._active),
                    "pending": len(self._pending),
                    "created": self._created,
                    "room_changes": self._room_changes,
                    "closed": self._closed,
                    "suppressed_observations": self._suppressed_observations,
                },
            }
