"""Deterministic demo data mirrored from the Sentinel VMS web source."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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
    restricted: bool = False
    roi: str = "To'liq kadr"
    last_error: str | None = None
    frame_age: int = 0
    queue: int = 0
    dropped: int = 0
    reconnects: int = 0
    latency: int = 0


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


ROOMS = [
    Room("lobby", "Lobbi", 40),
    Room("office", "Ofis", 25),
    Room("warehouse", "Ombor", 15),
]


CAMERA_ROWS = [
    ("cam-1", "Kirish eshigi", "lobby", True, 24.6, False),
    ("cam-2", "Lobbi janubi", "lobby", True, 25.1, False),
    ("cam-3", "Ofis koridori", "office", True, 22.8, False),
    ("cam-4", "Ofis ochiq zona", "office", False, 0.0, False),
    ("cam-5", "Ombor kirishi", "warehouse", True, 19.4, True),
    ("cam-6", "Ombor tokchalari", "warehouse", True, 20.2, True),
]


CAMERAS: list[Camera] = []
for i, (cam_id, name, room_id, online, fps, restricted) in enumerate(CAMERA_ROWS):
    CAMERAS.append(
        Camera(
            cam_id,
            name,
            room_id,
            f"rtsp://192.168.10.{11 + i}:554/stream1",
            online,
            fps,
            restricted=restricted,
            roi="Polygon (6 nuqta)" if restricted else "To'liq kadr",
            last_error=None if online else "RTSP timeout (10s) — qayta ulanmoqda",
            frame_age=40 + i * 12 if online else 98_000,
            queue=(i % 3) + 1 if online else 0,
            dropped=12 * (i + 1) if online else 4210,
            reconnects=i if online else 37,
            latency=28 + i * 6 if online else 0,
        )
    )


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
        seen = [c.id for c in CAMERAS if rng.random() > 0.5][:3] or ["cam-1"]
        inside = rng.random() > 0.35
        room_id = next(c.room_id for c in CAMERAS if c.id == seen[-1])
        result.append(Person(f"P-{1000+i}", name, True, DAY0 + timedelta(minutes=first),
                             DAY0 + timedelta(minutes=last), room_id if inside else None,
                             inside, rng.randrange(360), seen))
    for i in range(6):
        first = rng.randrange(200)
        last = first + 5 + rng.randrange(120)
        seen = [c.id for c in CAMERAS if rng.random() > 0.6] or ["cam-2"]
        inside = rng.random() > 0.4
        room_id = next(c.room_id for c in CAMERAS if c.id == seen[-1])
        result.append(Person(f"P-{2000+i}", None, False, DAY0 + timedelta(minutes=first),
                             DAY0 + timedelta(minutes=last), room_id if inside else None,
                             inside, rng.randrange(360), seen))
    return result


PEOPLE = build_people()


TYPE_LABEL = {
    "entry": "Kirish",
    "exit": "Chiqish",
    "transition": "Xonalar orasida",
    "unknown": "Unknown paydo bo'ldi",
    "restricted": "Restricted zone",
    "camera_offline": "Kamera offline",
    "service": "Servis",
}


def build_events() -> list[Event]:
    rng = random.Random(7)
    result: list[Event] = []
    known_types = ["entry", "exit", "transition"]
    unknown_types = ["unknown", "restricted"]
    for i in range(46):
        p = rng.choice(PEOPLE)
        cam = rng.choice(CAMERAS)
        kind = rng.choice(known_types if p.known else unknown_types)
        room = next(r for r in ROOMS if r.id == cam.room_id)
        messages = {
            "entry": f"{p.label} binoga kirdi",
            "exit": f"{p.label} binodan chiqdi",
            "transition": f"{p.label} {room.name} xonasiga o'tdi",
            "unknown": f"Yangi Unknown odam aniqlandi ({p.label})",
            "restricted": f"{p.label} cheklangan hududda",
        }
        result.append(Event(f"E-{5000+i}", kind, DAY0 + timedelta(minutes=rng.randrange(300)),
                            cam.id, cam.room_id, p.id, messages[kind]))
    result.extend([
        Event("E-9001", "camera_offline", DAY0 + timedelta(minutes=212), "cam-4", "office", None,
              "Ofis ochiq zona kamerasi offline bo'ldi"),
        Event("E-9002", "service", DAY0 + timedelta(minutes=150), None, None, None,
              "Re-ID servisi qayta ishga tushirildi"),
    ])
    return sorted(result, key=lambda event: event.at, reverse=True)


EVENTS = build_events()


NOTIFICATIONS = [
    ("critical", "Kamera offline", "Ofis ochiq zona (cam-4) ulanmayapti — RTSP timeout"),
    ("warning", "Yangi Unknown odam", "Unknown_05 lobbida aniqlandi"),
    ("critical", "Restricted zone", "Ombor tokchalari zonasida ruxsatsiz shaxs"),
    ("info", "Servis holati", "Re-ID servisi qayta ishga tushdi va normal ishlamoqda"),
]


def room_name(room_id: str | None) -> str:
    return next((room.name for room in ROOMS if room.id == room_id), "—")


def camera_name(camera_id: str | None) -> str:
    return next((camera.name for camera in CAMERAS if camera.id == camera_id), "—")


def fmt(value: datetime) -> str:
    return value.astimezone().strftime("%d.%m.%Y, %H:%M:%S")
