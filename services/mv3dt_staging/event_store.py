from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


CAMERA_MAPS = {
    "dev-room": {
    "Camera1": ("CAM-01", 0),
    "Camera2": ("CAM-04", 1),
    },
    "entrance-room": {
        "Camera1": ("CAM-02", 0),
        "Camera2": ("CAM-05", 1),
    },
}
CAMERA_MAP = CAMERA_MAPS["dev-room"]
DEFAULT_FPS = 30.0
EXPECTED_FRAMES = 3600


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_kafka_json_messages(path: str | Path) -> Iterator[dict[str, Any]]:
    """Parse the official kafka_client.py text dump without altering source data."""
    text = Path(path).read_text(encoding="utf-8")
    for chunk in text.split("JSON Data:\n")[1:]:
        raw = chunk.split("\n--------------------------------------------------", 1)[0].strip()
        if not raw:
            continue
        message = json.loads(raw)
        if not isinstance(message, dict):
            raise ValueError("MV3DT Kafka JSON record must be an object")
        yield message


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parse_pair(value: Any) -> tuple[float, float] | None:
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, (tuple, list)):
        parts = value
    else:
        return None
    if len(parts) != 2:
        return None
    left, right = (_finite_number(part.strip() if isinstance(part, str) else part) for part in parts)
    if left is None or right is None:
        return None
    return left, right


def _normalise_timestamp(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("MV3DT event is missing its publication timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid MV3DT timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError("MV3DT event timestamp must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _normalise_bbox(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    fields = ("leftX", "topY", "rightX", "bottomY")
    result = {name: _finite_number(value.get(name)) for name in fields}
    if any(number is None for number in result.values()):
        return None
    return {name: float(number) for name, number in result.items()}


def normalise_message(
    message: dict[str, Any],
    fps: float = DEFAULT_FPS,
    camera_mapping: dict[str, tuple[str, int]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    camera_mapping = camera_mapping or CAMERA_MAP
    sensor = message.get("sensorId")
    if sensor not in camera_mapping:
        raise ValueError(f"unsupported MV3DT sensorId: {sensor!r}")
    camera_id, source_id = camera_mapping[sensor]
    frame_text = message.get("id")
    try:
        frame_id = int(frame_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid MV3DT frame id: {frame_text!r}") from exc
    if frame_id < 0 or fps <= 0:
        raise ValueError("frame id and nominal fps must be nonnegative/positive")
    event_timestamp = _normalise_timestamp(message.get("timestamp"))
    message_row = {
        "camera_id": camera_id,
        "source_id": source_id,
        "frame_id": frame_id,
        "event_timestamp": event_timestamp,
        "object_count": len(message.get("objects", [])) if isinstance(message.get("objects", []), list) else 0,
        "raw_json": json.dumps(message, separators=(",", ":"), sort_keys=True, ensure_ascii=False),
    }
    objects = message.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("MV3DT objects field must be a list")
    rows: list[dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict) or obj.get("id") is None:
            raise ValueError("MV3DT object must have an id")
        info = obj.get("info") if isinstance(obj.get("info"), dict) else {}
        foot_2d = _parse_pair(info.get("footLocation2D"))
        foot_3d = _parse_pair(info.get("footLocation3D"))
        confidence = _finite_number(obj.get("confidence"))
        rows.append({
            "camera_id": camera_id,
            "source_id": source_id,
            "frame_id": frame_id,
            # This is the MV3DT output object ID, not the local NvDCF/ByteTrack
            # ID and not a verified real-world person identity.
            "mv3dt_global_id": str(obj["id"]),
            "event_timestamp": event_timestamp,
            # Approximate file position derived from normalized frame index;
            # the Kafka event does not carry source PTS.
            "media_time_seconds_nominal": frame_id / fps,
            "object_type": str(obj.get("type", "")),
            "mv3dt_object_confidence": confidence,
            "bbox": _normalise_bbox(obj.get("bbox")),
            "foot_x_px": foot_2d[0] if foot_2d else None,
            "foot_y_px": foot_2d[1] if foot_2d else None,
            "world_x": foot_3d[0] if foot_3d else None,
            "world_y": foot_3d[1] if foot_3d else None,
            "raw_json": json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False),
        })
    return message_row, rows


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA user_version=1;
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS frame_messages (
            camera_id TEXT NOT NULL,
            source_id INTEGER NOT NULL,
            frame_id INTEGER NOT NULL,
            event_timestamp TEXT NOT NULL,
            object_count INTEGER NOT NULL,
            raw_json TEXT NOT NULL,
            PRIMARY KEY(camera_id, frame_id)
        );
        CREATE TABLE IF NOT EXISTS observations (
            camera_id TEXT NOT NULL,
            source_id INTEGER NOT NULL,
            frame_id INTEGER NOT NULL,
            mv3dt_global_id TEXT NOT NULL,
            event_timestamp TEXT NOT NULL,
            media_time_seconds_nominal REAL NOT NULL,
            object_type TEXT NOT NULL,
            mv3dt_object_confidence REAL,
            bbox_json TEXT,
            foot_x_px REAL,
            foot_y_px REAL,
            world_x REAL,
            world_y REAL,
            raw_json TEXT NOT NULL,
            PRIMARY KEY(camera_id, frame_id, mv3dt_global_id)
        );
        CREATE INDEX IF NOT EXISTS idx_observations_frame
            ON observations(frame_id, source_id, mv3dt_global_id);
        CREATE INDEX IF NOT EXISTS idx_observations_camera_track
            ON observations(camera_id, mv3dt_global_id, frame_id);
        """
    )


def import_event_log(
    event_log: str | Path,
    database: str | Path,
    *,
    fps: float = DEFAULT_FPS,
    expected_frames: int = EXPECTED_FRAMES,
    provenance: dict[str, str] | None = None,
    camera_mapping: dict[str, tuple[str, int]] | None = None,
) -> dict[str, int | str]:
    """Create an isolated SQLite staging store; repeated exact import is idempotent."""
    source = Path(event_log).resolve(strict=True)
    camera_mapping = camera_mapping or CAMERA_MAP
    if (not camera_mapping or len(set(camera_mapping.values())) != len(camera_mapping)
            or any(not sensor or not camera or source_id < 0
                   for sensor, (camera, source_id) in camera_mapping.items())):
        raise ValueError("camera mapping must contain unique camera/source pairs")
    db_path = Path(database)
    db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    old_umask = os.umask(0o077)
    try:
        connection = sqlite3.connect(str(db_path), timeout=10.0)
    finally:
        os.umask(old_umask)
    inserted_messages = inserted_observations = duplicate_messages = duplicate_observations = 0
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        _create_schema(connection)
        source_hash = sha256_file(source)
        prior = connection.execute("SELECT value FROM metadata WHERE key='source_log_sha256'").fetchone()
        if prior and prior[0] != source_hash:
            raise ValueError("staging database already belongs to a different event-log hash")
        connection.execute("BEGIN")
        for message in iter_kafka_json_messages(source):
            frame, observations = normalise_message(message, fps=fps, camera_mapping=camera_mapping)
            before = connection.total_changes
            connection.execute(
                "INSERT OR IGNORE INTO frame_messages VALUES (?, ?, ?, ?, ?, ?)",
                (frame["camera_id"], frame["source_id"], frame["frame_id"], frame["event_timestamp"],
                 frame["object_count"], frame["raw_json"]),
            )
            if connection.total_changes == before:
                existing = connection.execute(
                    "SELECT raw_json FROM frame_messages WHERE camera_id=? AND frame_id=?",
                    (frame["camera_id"], frame["frame_id"]),
                ).fetchone()
                if not existing or existing[0] != frame["raw_json"]:
                    raise ValueError(f"conflicting duplicate MV3DT frame message {frame['camera_id']}:{frame['frame_id']}")
                duplicate_messages += 1
            else:
                inserted_messages += 1
            for row in observations:
                before = connection.total_changes
                connection.execute(
                    "INSERT OR IGNORE INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (row["camera_id"], row["source_id"], row["frame_id"], row["mv3dt_global_id"],
                     row["event_timestamp"], row["media_time_seconds_nominal"], row["object_type"],
                     row["mv3dt_object_confidence"],
                     json.dumps(row["bbox"], separators=(",", ":"), sort_keys=True) if row["bbox"] else None,
                     row["foot_x_px"], row["foot_y_px"], row["world_x"], row["world_y"], row["raw_json"]),
                )
                if connection.total_changes == before:
                    existing = connection.execute(
                        "SELECT raw_json FROM observations WHERE camera_id=? AND frame_id=? AND mv3dt_global_id=?",
                        (row["camera_id"], row["frame_id"], row["mv3dt_global_id"]),
                    ).fetchone()
                    if not existing or existing[0] != row["raw_json"]:
                        raise ValueError(
                            "conflicting duplicate MV3DT observation "
                            f"{row['camera_id']}:{row['frame_id']}:{row['mv3dt_global_id']}"
                        )
                    duplicate_observations += 1
                else:
                    inserted_observations += 1
        metadata = {
            "source_log_sha256": source_hash,
            "nominal_fps": str(float(fps)),
            "expected_frames": str(int(expected_frames)),
            "camera_mapping_json": json.dumps({k: v[0] for k, v in camera_mapping.items()}, sort_keys=True),
            "camera_source_ids_json": json.dumps({k: v[1] for k, v in camera_mapping.items()}, sort_keys=True),
            "timestamp_semantics": "MV3DT message publication timestamp; source PTS is not present in this log",
            "media_time_semantics": "nominal frame_id / 30; not verified source PTS",
        }
        if provenance:
            for key, value in provenance.items():
                metadata[f"provenance_{key}"] = str(value)
        for key, value in metadata.items():
            connection.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)", (key, value))
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()
    os.chmod(db_path, 0o600)
    with sqlite3.connect(str(db_path)) as connection:
        message_count = int(connection.execute("SELECT COUNT(*) FROM frame_messages").fetchone()[0])
        observation_count = int(connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0])
    return {
        "source_log_sha256": source_hash,
        "inserted_messages": inserted_messages,
        "inserted_observations": inserted_observations,
        "duplicate_messages": duplicate_messages,
        "duplicate_observations": duplicate_observations,
        "database_messages": message_count,
        "database_observations": observation_count,
    }


def parse_world_to_map_transform(path: str | Path) -> list[float]:
    """Read T_ov2px's documented row-major 3x3 matrix from transforms.yml."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    start = next((index for index, line in enumerate(lines) if line.strip() == "T_ov2px:"), None)
    if start is None:
        raise ValueError("transforms.yml is missing T_ov2px")
    values: list[float] = []
    for line in lines[start + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("-"):
            break
        token = stripped[1:].strip().split("#", 1)[0].strip()
        value = _finite_number(token)
        if value is None:
            raise ValueError("T_ov2px contains a non-finite/non-numeric value")
        values.append(value)
    if len(values) != 9:
        raise ValueError(f"T_ov2px must contain 9 values, found {len(values)}")
    return values

