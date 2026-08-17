from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "cameras.yaml"
ROOMS = ("Entrance", "Devs", "Main Rooms")
MAX_CAMERAS = 16


def load_raw() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Camera config not found: {CONFIG_PATH}")
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    raw.setdefault("cameras", [])
    raw.setdefault("deepstream", {})
    raw.setdefault("display", {})
    return raw


def list_cameras(*, include_disabled: bool = True) -> list[dict]:
    rows = []
    for row in load_raw().get("cameras", []):
        item = dict(row)
        item["id"] = str(item.get("id", "")).strip()
        item["name"] = str(item.get("name", item["id"])).strip() or item["id"]
        item["room"] = str(item.get("room", "")).strip()
        item["codec"] = str(item.get("codec", "h264")).strip().lower()
        item["enabled"] = bool(item.get("enabled", True))
        if include_disabled or item["enabled"]:
            rows.append(item)
    return rows


def next_camera_id() -> str:
    used = {str(row.get("id", "")) for row in list_cameras()}
    for number in range(1, 100):
        candidate = f"CAM-{number:02d}"
        if candidate not in used:
            return candidate
    raise RuntimeError("Could not allocate a camera id")


def _validate_camera(row: dict, *, existing_id: str | None = None) -> dict:
    camera = deepcopy(row)
    camera_id = str(camera.get("id", "")).strip()
    if not camera_id:
        raise ValueError("Camera ID is required")
    uri = str(camera.get("uri", "")).strip()
    if not uri.startswith("rtsp://"):
        raise ValueError("RTSP URL must start with rtsp://")
    codec = str(camera.get("codec", "h264")).strip().lower()
    if codec not in {"h264", "h265"}:
        raise ValueError("Codec must be h264 or h265")
    room = str(camera.get("room", "")).strip()
    if room and room not in ROOMS:
        raise ValueError(f"Room must be one of: {', '.join(ROOMS)}")

    for current in list_cameras():
        current_id = str(current.get("id", ""))
        if current_id == camera_id and current_id != existing_id:
            raise ValueError(f"Duplicate camera ID: {camera_id}")

    return {
        "id": camera_id,
        "name": str(camera.get("name", camera_id)).strip() or camera_id,
        "room": room,
        "enabled": bool(camera.get("enabled", True)),
        "uri": uri,
        "codec": codec,
        "username": str(camera.get("username", "")),
        "password": str(camera.get("password", "")),
    }


def save_camera(row: dict, *, existing_id: str | None = None) -> dict:
    camera = _validate_camera(row, existing_id=existing_id)
    raw = load_raw()
    rows = list(raw.get("cameras") or [])
    if existing_id is None and len(rows) >= MAX_CAMERAS:
        raise ValueError(f"Maximum camera count is {MAX_CAMERAS}")

    replaced = False
    for index, current in enumerate(rows):
        if str(current.get("id", "")) == str(existing_id or camera["id"]):
            # Settings is authoritative after a manual edit. Do not preserve an old
            # env_uri indirection that could silently override the RTSP URL the user
            # just entered in the UI.
            rows[index] = dict(camera)
            replaced = True
            break
    if not replaced:
        rows.append(dict(camera))

    if not any(bool(item.get("enabled", True)) for item in rows):
        raise ValueError("At least one camera must remain enabled")

    raw["cameras"] = rows
    CONFIG_PATH.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return camera


def delete_camera(camera_id: str) -> None:
    raw = load_raw()
    rows = [
        row for row in (raw.get("cameras") or [])
        if str(row.get("id", "")) != str(camera_id)
    ]
    if len(rows) == len(raw.get("cameras") or []):
        raise ValueError(f"Unknown camera: {camera_id}")
    if not rows or not any(bool(item.get("enabled", True)) for item in rows):
        raise ValueError("At least one enabled camera must remain")
    raw["cameras"] = rows
    CONFIG_PATH.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def room_cameras() -> dict[str, list[dict]]:
    output = {room: [] for room in ROOMS}
    for row in list_cameras(include_disabled=False):
        room = str(row.get("room", ""))
        if room in output:
            output[room].append(row)
    return output
