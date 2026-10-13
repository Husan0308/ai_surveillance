"""Sentinel VMS UI state for the staged real-camera rollout.

Camera demo rows are intentionally removed. During this milestone the UI
contains only runtime-backed camera cards selected by SENTINEL_CAMERA_IDS. Other demo
domains (people/events/rooms) remain temporarily so the rest of the supplied
interface is preserved while cameras are replaced one by one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path


@dataclass
class Room:
    id: str
    name: str
    capacity: int


@dataclass
class Camera:
    id: str
    name: str
    room_id: str
    rtsp: str
    online: bool
    fps: float
    enabled: bool = True
    last_error: str | None = None
    frame_age: int = 0
    queue: int | None = None
    dropped: int | None = None
    reconnects: int | None = None
    latency: int | None = None
    source_fps: float | None = None
    render_fps: float | None = None
    infer_hz: float | None = None
    current_box_count: int | None = None
    detector_enabled: bool | None = None
    telemetry_status: str = "offline"
    username: str = ""
    password: str = ""


@dataclass
class Person:
    id: str
    name: str | None
    known: bool
    first_seen: datetime
    last_seen: datetime
    room_id: str | None
    in_building: bool
    hue: int
    cameras: list[str] = field(default_factory=list)
    profile_image: str | None = None
    note: str = ""
    entered_at: datetime | None = None

    @property
    def label(self) -> str:
        return self.name or f"Unknown_{self.id[-2:]}"


@dataclass
class Event:
    id: str
    type: str
    at: datetime
    camera_id: str | None
    room_id: str | None
    person_id: str | None
    message: str
    snapshot_path: str | None = None
    bbox: tuple[float, float, float, float] = (0.36, 0.14, 0.26, 0.72)


ROOMS = [
    Room("lobby", "Lobbi", 40),
    Room("office", "Ofis", 25),
    Room("warehouse", "Ombor", 15),
]


# Camera cards are authoritative runtime cards, not demo RTSP rows.
# Add the next real camera here only after its pipeline milestone passes.
RUNTIME_CAMERA_IDS = ("CAM-01", "CAM-02")
_ALLOWED_RUNTIME_CAMERA_IDS = tuple(f"CAM-{index:02d}" for index in range(1, 7))
_raw_runtime_camera_ids = os.environ.get("SENTINEL_CAMERA_IDS", "").strip()
if _raw_runtime_camera_ids:
    _requested_runtime_camera_ids = tuple(
        value.strip() for value in _raw_runtime_camera_ids.split(",") if value.strip()
    )
    if not _requested_runtime_camera_ids:
        raise RuntimeError("SENTINEL_CAMERA_IDS resolved to zero camera cards")
    if len(set(_requested_runtime_camera_ids)) != len(_requested_runtime_camera_ids):
        raise RuntimeError("SENTINEL_CAMERA_IDS contains duplicate camera cards")
    invalid = [
        camera_id
        for camera_id in _requested_runtime_camera_ids
        if camera_id not in _ALLOWED_RUNTIME_CAMERA_IDS
    ]
    if invalid:
        raise RuntimeError(f"unsupported Sentinel camera cards: {invalid}")
    RUNTIME_CAMERA_IDS = _requested_runtime_camera_ids


def _runtime_camera(camera_id: str) -> Camera:
    return Camera(
        id=camera_id,
        name=camera_id,
        room_id="",
        rtsp="",
        online=False,
        fps=0.0,
        enabled=True,
        last_error="Ulanish kutilmoqda",
    )


CAMERAS: list[Camera] = [_runtime_camera(camera_id) for camera_id in RUNTIME_CAMERA_IDS]
CAMERAS_FILE = Path(__file__).with_name("cameras.json")


def load_cameras() -> None:
    """Apply saved settings only to staged runtime cameras.

    Legacy demo ids such as cam-1..cam-6 are deliberately ignored so an old
    cameras.json cannot resurrect demo camera cards or hide the real runtime
    preview cards.
    """
    if not CAMERAS_FILE.exists():
        return
    try:
        rows = json.loads(CAMERAS_FILE.read_text(encoding="utf-8"))
        rows_by_id = {
            str(row.get("id", "")).strip(): row
            for row in rows
            if isinstance(row, dict)
        }
        for camera in CAMERAS:
            row = rows_by_id.get(camera.id)
            if not row:
                continue
            camera.name = str(row.get("name", camera.name)).strip() or camera.name
            camera.room_id = str(row.get("room_id", camera.room_id)).strip()
            camera.rtsp = str(row.get("rtsp", camera.rtsp)).strip()
            camera.enabled = bool(row.get("enabled", camera.enabled))
            camera.username = str(row.get("username", camera.username))
            camera.password = str(row.get("password", camera.password))
            camera.online = False
            camera.fps = 0.0
            camera.last_error = "Ulanish kutilmoqda" if camera.enabled else "Kamera o'chirilgan"
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return


def save_cameras() -> None:
    """Persist editable connection settings with owner-only file permissions."""
    rows = [
        {
            "id": camera.id,
            "name": camera.name,
            "room_id": camera.room_id,
            "rtsp": camera.rtsp,
            "enabled": camera.enabled,
            "username": camera.username,
            "password": camera.password,
        }
        for camera in CAMERAS
    ]
    temporary = CAMERAS_FILE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(CAMERAS_FILE)


def next_camera_id() -> str:
    numbers: list[int] = []
    for camera in CAMERAS:
        value = camera.id.upper()
        if value.startswith("CAM-") and value[4:].isdigit():
            numbers.append(int(value[4:]))
    return f"CAM-{max(numbers, default=0) + 1:02d}"


load_cameras()


# No fabricated face-recognition identities or events in a live monitoring UI.
# Real profiles/events must arrive from verified services or persisted records.
PEOPLE: list[Person] = []
EVENTS: list[Event] = []
NOTIFICATIONS: list[tuple[str, str, str]] = []

TYPE_LABEL = {
    "entry": "Kirish",
    "exit": "Chiqish",
}


def room_name(room_id: str | None) -> str:
    return next((room.name for room in ROOMS if room.id == room_id), room_id or "—")


def camera_name(camera_id: str | None) -> str:
    return next((camera.name for camera in CAMERAS if camera.id == camera_id), "—")


def fmt(value: datetime) -> str:
    return value.astimezone().strftime("%d.%m.%Y, %H:%M:%S")
