"""Deterministic demo data mirrored from the Sentinel VMS web source."""

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
    queue: int = 0
    dropped: int = 0
    reconnects: int = 0
    latency: int = 0
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


CAMERA_ROWS = [
    ("cam-1", "Kirish eshigi", "lobby", True, 24.6),
    ("cam-2", "Lobbi janubi", "lobby", True, 25.1),
    ("cam-3", "Ofis koridori", "office", True, 22.8),
    ("cam-4", "Ofis ochiq zona", "office", False, 0.0),
    ("cam-5", "Ombor kirishi", "warehouse", True, 19.4),
    ("cam-6", "Ombor tokchalari", "warehouse", True, 20.2),
]


CAMERAS: list[Camera] = []
for i, (cam_id, name, room_id, online, fps) in enumerate(CAMERA_ROWS):
    CAMERAS.append(
        Camera(
            cam_id,
            name,
            room_id,
            f"rtsp://192.168.10.{11 + i}:554/stream1",
            online,
            fps,
            last_error=None if online else "RTSP timeout (10s) — qayta ulanmoqda",
            frame_age=40 + i * 12 if online else 98_000,
            queue=(i % 3) + 1 if online else 0,
            dropped=12 * (i + 1) if online else 4210,
            reconnects=i if online else 37,
            latency=28 + i * 6 if online else 0,
        )
    )


CAMERAS_FILE = Path(__file__).with_name("cameras.json")


def load_cameras() -> None:
    """Load locally saved camera settings, keeping demo cameras as a safe fallback."""
    if not CAMERAS_FILE.exists():
        return
    try:
        rows = json.loads(CAMERAS_FILE.read_text(encoding="utf-8"))
        loaded = []
        for row in rows:
            loaded.append(
                Camera(
                    id=str(row["id"]),
                    name=str(row["name"]),
                    room_id=str(row.get("room_id", "")).strip(),
                    rtsp=str(row["rtsp"]),
                    online=False,
                    fps=0.0,
                    enabled=bool(row.get("enabled", True)),
                    last_error="Ulanish kutilmoqda" if row.get("enabled", True) else "Kamera o'chirilgan",
                    username=str(row.get("username", "")),
                    password=str(row.get("password", "")),
                )
            )
        CAMERAS[:] = loaded
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        # A damaged settings file must not prevent the VMS interface from opening.
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
    numbers = [int(camera.id[4:]) for camera in CAMERAS if camera.id.startswith("cam-") and camera.id[4:].isdigit()]
    return f"cam-{max(numbers, default=0) + 1}"


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
        # Normalized coordinates are replaced by the detector's real bbox in production.
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

        # One person represents one visit: no repeated detections become new events.
        # People still in the building have no exit event yet.
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
