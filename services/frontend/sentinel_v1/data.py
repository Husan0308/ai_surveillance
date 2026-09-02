"""Sentinel VMS UI state for the staged real-camera rollout.

Camera demo rows are intentionally removed. During this milestone the UI
contains only runtime-backed camera cards selected by SENTINEL_CAMERA_IDS. Other demo
domains (people/events/rooms) remain temporarily so the rest of the supplied
interface is preserved while cameras are replaced one by one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import random


DAY0 = datetime(2026, 8, 16, 8, 0, tzinfo=timezone.utc)


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


KNOWN_NAMES = [
    "Javohir Zarifov",
    "Dilnoza Karimova",
    "Sardor Umarov",
    "Malika Yusupova",
    "Bekzod Rasulov",
    "Nilufar Tosheva",
    "Aziz Qodirov",
    "Kamola Ergasheva",
]


def build_people() -> list[Person]:
    rng = random.Random(42)
    result: list[Person] = []
    for i, name in enumerate(KNOWN_NAMES):
        first = rng.randrange(90)
        last = first + 20 + rng.randrange(200)
        seen = [c.id for c in CAMERAS if rng.random() > 0.5][:3]
        if not seen and CAMERAS:
            seen = [CAMERAS[0].id]
        inside = bool(CAMERAS) and rng.random() > 0.35
        room_id = next((c.room_id for c in CAMERAS if seen and c.id == seen[-1]), None)
        entered_at = datetime.now(timezone.utc) - timedelta(minutes=5 + rng.randrange(180)) if inside else None
        result.append(Person(f"P-{1000+i}", name, True, DAY0 + timedelta(minutes=first),
                             DAY0 + timedelta(minutes=last), room_id if inside else None,
                             inside, rng.randrange(360), seen, entered_at=entered_at))
    for i in range(6):
        first = rng.randrange(200)
        last = first + 5 + rng.randrange(120)
        seen = [c.id for c in CAMERAS if rng.random() > 0.6]
        if not seen and CAMERAS:
            seen = [CAMERAS[0].id]
        inside = bool(CAMERAS) and rng.random() > 0.4
        room_id = next((c.room_id for c in CAMERAS if seen and c.id == seen[-1]), None)
        entered_at = datetime.now(timezone.utc) - timedelta(minutes=5 + rng.randrange(180)) if inside else None
        result.append(Person(f"P-{2000+i}", None, False, DAY0 + timedelta(minutes=first),
                             DAY0 + timedelta(minutes=last), room_id if inside else None,
                             inside, rng.randrange(360), seen, entered_at=entered_at))
    return result


PEOPLE = build_people()


TYPE_LABEL = {
    "entry": "Kirish",
    "exit": "Chiqish",
}


def build_events() -> list[Event]:
    if not CAMERAS:
        return []
    rng = random.Random(7)
    result: list[Event] = []
    event_number = 5000

    def add_event(person, kind, at, camera, bbox_seed):
        nonlocal event_number
        action = "binoga kirdi" if kind == "entry" else "binodan chiqdi"
        identity = person.label if person.known else f"Unknown shaxs ({person.label})"
        bbox_x = 0.18 + (bbox_seed * 17 % 32) / 100
        bbox = (bbox_x, 0.13, 0.24, 0.74)
        result.append(Event(
            f"E-{event_number}", kind, at, camera.id, camera.room_id,
            person.id, f"{identity} {action}", bbox=bbox,
        ))
        event_number += 1

    for index, person in enumerate(PEOPLE):
        available = [camera for camera in CAMERAS if camera.id in person.cameras] or CAMERAS
        entry_camera = rng.choice(available)
        entry_at = DAY0 + timedelta(minutes=rng.randrange(15, 125))
        add_event(person, "entry", entry_at, entry_camera, index * 2)

        if not person.in_building:
            exit_camera = rng.choice(available)
            visit_minutes = rng.randrange(45, 151)
            add_event(person, "exit", entry_at + timedelta(minutes=visit_minutes), exit_camera, index * 2 + 1)
    return sorted(result, key=lambda event: event.at, reverse=True)


EVENTS = build_events()


NOTIFICATIONS = [
    ("critical", "Kamera offline", "Ofis ochiq zona (cam-4) ulanmayapti — RTSP timeout"),
    ("warning", "Yangi Unknown odam", "Unknown_05 lobbida aniqlandi"),
    ("info", "Servis holati", "Re-ID servisi qayta ishga tushdi va normal ishlamoqda"),
]


def room_name(room_id: str | None) -> str:
    return next((room.name for room in ROOMS if room.id == room_id), room_id or "—")


def camera_name(camera_id: str | None) -> str:
    return next((camera.name for camera in CAMERAS if camera.id == camera_id), "—")


def fmt(value: datetime) -> str:
    return value.astimezone().strftime("%d.%m.%Y, %H:%M:%S")
