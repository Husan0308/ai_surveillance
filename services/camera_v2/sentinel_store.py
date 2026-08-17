from __future__ import annotations

import shutil
import sqlite3
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / ".runtime" / "sentinel"
DB_PATH = DATA_DIR / "sentinel.db"
PEOPLE_DIR = DATA_DIR / "people"
EVENTS_DIR = DATA_DIR / "events"


class SentinelStore:
    """Small local persistence layer for UI state.

    It intentionally does not perform recognition. Known worker profiles, one-shot
    event snapshots and UI metadata live here while the camera/detection hot path
    remains isolated in the DeepStream process.
    """

    def __init__(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        PEOPLE_DIR.mkdir(parents=True, exist_ok=True)
        EVENTS_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(DB_PATH, timeout=5.0)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        return db

    def _init_db(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS people (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT '',
                    department TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    profile_photo TEXT NOT NULL DEFAULT '',
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    person_id TEXT NOT NULL DEFAULT '',
                    local_id TEXT NOT NULL DEFAULT '',
                    camera_id TEXT NOT NULL DEFAULT '',
                    room TEXT NOT NULL DEFAULT '',
                    snapshot_path TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    dedup_key TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_events_created_at
                    ON events(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_events_dedup
                    ON events(dedup_key, created_at DESC);
                """
            )

    def list_people(self) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM people WHERE active=1 ORDER BY name COLLATE NOCASE"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_person(self, person_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM people WHERE id=?",
                (str(person_id),),
            ).fetchone()
        return dict(row) if row else None

    def enroll_person(
        self,
        *,
        name: str,
        role: str,
        department: str,
        notes: str,
        image_paths: list[str],
        profile_index: int,
    ) -> dict:
        name = str(name).strip()
        if not name:
            raise ValueError("Full name is required")
        if len(image_paths) != 10:
            raise ValueError("Exactly 10 face images are required")
        if not 0 <= int(profile_index) < len(image_paths):
            raise ValueError("Select one of the 10 images as profile photo")

        for path in image_paths:
            p = Path(path)
            if not p.is_file():
                raise ValueError(f"Image not found: {p}")

        person_id = "P-" + uuid.uuid4().hex[:8].upper()
        target = PEOPLE_DIR / person_id
        target.mkdir(parents=True, exist_ok=False)
        copied: list[Path] = []
        try:
            for index, source in enumerate(image_paths, start=1):
                src = Path(source)
                suffix = src.suffix.lower() if src.suffix else ".jpg"
                dst = target / f"face_{index:02d}{suffix}"
                shutil.copy2(src, dst)
                copied.append(dst)

            profile = copied[int(profile_index)]
            with self._connect() as db:
                db.execute(
                    """
                    INSERT INTO people(
                        id,name,role,department,notes,profile_photo,active,created_at
                    ) VALUES(?,?,?,?,?,?,1,?)
                    """,
                    (
                        person_id,
                        name,
                        str(role).strip(),
                        str(department).strip(),
                        str(notes).strip(),
                        str(profile),
                        time.time(),
                    ),
                )
            return self.get_person(person_id) or {}
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise

    def update_person(
        self,
        person_id: str,
        *,
        name: str,
        role: str,
        department: str,
        notes: str,
    ) -> None:
        with self._connect() as db:
            db.execute(
                """
                UPDATE people SET name=?, role=?, department=?, notes=? WHERE id=?
                """,
                (
                    str(name).strip(),
                    str(role).strip(),
                    str(department).strip(),
                    str(notes).strip(),
                    str(person_id),
                ),
            )

    def deactivate_person(self, person_id: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE people SET active=0 WHERE id=?", (str(person_id),))

    def record_event_once(
        self,
        *,
        event_type: str,
        person_id: str = "",
        local_id: str = "",
        camera_id: str = "",
        room: str = "",
        snapshot_bytes: bytes | None = None,
        dedup_seconds: float = 15.0,
        created_at: float | None = None,
    ) -> tuple[dict, bool]:
        """Insert one event/snapshot and suppress repeated frames of the same event.

        The caller may invoke this on every tracking frame. Only the first event in
        the dedup window is persisted, so entry/exit does not generate hundreds of
        snapshots while a person remains near the boundary.
        """
        event_type = str(event_type).strip().lower()
        if not event_type:
            raise ValueError("event_type is required")
        now = float(created_at or time.time())
        identity = str(person_id or local_id or "anonymous")
        dedup_key = "|".join(
            (event_type, identity, str(camera_id).strip(), str(room).strip())
        )

        with self._connect() as db:
            existing = db.execute(
                """
                SELECT * FROM events
                WHERE dedup_key=? AND created_at>=?
                ORDER BY created_at DESC LIMIT 1
                """,
                (dedup_key, now - max(1.0, float(dedup_seconds))),
            ).fetchone()
            if existing:
                return dict(existing), False

            cursor = db.execute(
                """
                INSERT INTO events(
                    event_type,person_id,local_id,camera_id,room,
                    snapshot_path,created_at,dedup_key
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    event_type,
                    str(person_id),
                    str(local_id),
                    str(camera_id),
                    str(room),
                    "",
                    now,
                    dedup_key,
                ),
            )
            event_id = int(cursor.lastrowid)

            snapshot_path = ""
            if snapshot_bytes:
                snapshot = EVENTS_DIR / f"event_{event_id:08d}.jpg"
                snapshot.write_bytes(snapshot_bytes)
                snapshot_path = str(snapshot)
                db.execute(
                    "UPDATE events SET snapshot_path=? WHERE id=?",
                    (snapshot_path, event_id),
                )

            row = db.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
            return dict(row), True

    def list_events(self, *, limit: int = 250) -> list[dict]:
        limit = max(1, min(2000, int(limit)))
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT e.*, p.name AS person_name
                FROM events e
                LEFT JOIN people p ON p.id=e.person_id
                ORDER BY e.created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
