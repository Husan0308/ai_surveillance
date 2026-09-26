#!/usr/bin/env python3
"""Durable, bounded canonical-identity embedding galleries.

The store contains identity memory only.  It deliberately does not store
active tracks or visible positions, so reopening it cannot recreate a BEV
marker before a new observation arrives.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np


class IdentityGalleryStore:
    """SQLite store for application-id mappings and quality-filtered vectors."""

    def __init__(self, path: str | Path, max_entries_per_identity: int = 16):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_entries_per_identity = int(max_entries_per_identity)
        self.connection = sqlite3.connect(str(self.path), timeout=10.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS canonical_identities (
                canonical_id INTEGER PRIMARY KEY,
                application_id TEXT NOT NULL UNIQUE,
                created_frame INTEGER NOT NULL,
                created_at REAL NOT NULL,
                last_gallery_update REAL NOT NULL,
                last_world_x REAL,
                last_world_y REAL,
                last_source_timestamp TEXT,
                last_camera_id TEXT
            );
            CREATE TABLE IF NOT EXISTS identity_gallery (
                gallery_id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_id INTEGER NOT NULL,
                camera_id TEXT NOT NULL,
                frame INTEGER NOT NULL,
                quality REAL NOT NULL,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL,
                source_timestamp TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY(canonical_id) REFERENCES canonical_identities(canonical_id)
            );
            CREATE INDEX IF NOT EXISTS idx_identity_gallery_canonical
                ON identity_gallery(canonical_id);
            """
        )
        # Existing galleries predate the non-visible continuity anchor.
        columns = {row[1] for row in self.connection.execute(
            "PRAGMA table_info(canonical_identities)"
        )}
        for name, kind in (("last_world_x", "REAL"), ("last_world_y", "REAL"),
                           ("last_source_timestamp", "TEXT"), ("last_camera_id", "TEXT")):
            if name not in columns:
                self.connection.execute(
                    f"ALTER TABLE canonical_identities ADD COLUMN {name} {kind}"
                )
        self.connection.commit()

    def identities(self) -> list[dict]:
        rows = self.connection.execute(
            "SELECT canonical_id, application_id, created_frame, last_world_x, last_world_y, "
            "last_source_timestamp, last_camera_id FROM canonical_identities "
            "ORDER BY canonical_id"
        ).fetchall()
        return [
            {"canonical_id": int(row[0]), "application_id": str(row[1]),
             "created_frame": int(row[2]),
             "last_world": (float(row[3]), float(row[4]))
             if row[3] is not None and row[4] is not None else None,
             "last_source_timestamp": row[5], "last_camera_id": row[6]}
            for row in rows
        ]

    def gallery(self, canonical_id: int) -> list[dict]:
        rows = self.connection.execute(
            "SELECT camera_id, frame, quality, dimension, vector, source_timestamp "
            "FROM identity_gallery WHERE canonical_id = ? ORDER BY gallery_id",
            (int(canonical_id),),
        ).fetchall()
        result = []
        for camera_id, frame, quality, dimension, vector, source_timestamp in rows:
            array = np.frombuffer(vector, dtype=np.float32).copy()
            if int(dimension) != array.size:
                continue
            result.append({
                "camera_id": str(camera_id),
                "frame": int(frame),
                "quality": float(quality),
                "vector": array,
                "source_timestamp": source_timestamp,
            })
        return result

    def ensure_identity(self, canonical_id: int, application_id: str, created_frame: int) -> None:
        self.connection.execute(
            "INSERT INTO canonical_identities "
            "(canonical_id, application_id, created_frame, created_at, last_gallery_update) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(canonical_id) DO UPDATE SET application_id=excluded.application_id",
            (int(canonical_id), str(application_id), int(created_frame), time.time(), time.time()),
        )
        self.connection.commit()

    def update_last_observation(
        self, canonical_id: int, camera_id: str, world: tuple[float, float],
        source_timestamp: str | None,
    ) -> None:
        """Persist a continuity anchor, never an active/visible presence flag."""
        self.connection.execute(
            "UPDATE canonical_identities SET last_world_x=?, last_world_y=?, "
            "last_source_timestamp=?, last_camera_id=? WHERE canonical_id=?",
            (float(world[0]), float(world[1]), source_timestamp, str(camera_id), int(canonical_id)),
        )
        self.connection.commit()

    def _entries(self, canonical_id: int) -> list[tuple[int, float, np.ndarray]]:
        rows = self.connection.execute(
            "SELECT gallery_id, quality, dimension, vector FROM identity_gallery "
            "WHERE canonical_id = ? ORDER BY gallery_id",
            (int(canonical_id),),
        ).fetchall()
        entries = []
        for gallery_id, quality, dimension, vector in rows:
            array = np.frombuffer(vector, dtype=np.float32).copy()
            if int(dimension) == array.size:
                entries.append((int(gallery_id), float(quality), array))
        return entries

    def add_embedding(
        self,
        canonical_id: int,
        camera_id: str,
        frame: int,
        quality: float,
        vector: np.ndarray,
        source_timestamp: str | None = None,
    ) -> str:
        """Persist a usable vector, retaining quality and viewpoint diversity.

        Returns ``added``, ``duplicate``, ``rejected_quality`` or ``replaced``.
        A low-quality sample cannot replace a better representative.
        """
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(array))
        if array.size == 0 or norm <= 1e-8 or not np.isfinite(array).all():
            return "rejected_quality"
        array = array / norm
        quality = float(max(0.0, min(1.0, quality)))
        entries = self._entries(canonical_id)
        for _, _, existing in entries:
            if float(array @ existing) >= 0.995:
                return "duplicate"
        payload = (
            int(canonical_id), str(camera_id), int(frame), quality, int(array.size),
            sqlite3.Binary(array.tobytes()), source_timestamp, time.time(),
        )
        if len(entries) < self.max_entries_per_identity:
            self.connection.execute(
                "INSERT INTO identity_gallery "
                "(canonical_id, camera_id, frame, quality, dimension, vector, source_timestamp, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", payload,
            )
            result = "added"
        else:
            weakest = min(entries, key=lambda item: item[1])
            if quality <= weakest[1]:
                return "rejected_quality"
            self.connection.execute(
                "UPDATE identity_gallery SET camera_id=?, frame=?, quality=?, dimension=?, vector=?, "
                "source_timestamp=?, created_at=? WHERE gallery_id=?",
                (str(camera_id), int(frame), quality, int(array.size), sqlite3.Binary(array.tobytes()),
                 source_timestamp, time.time(), weakest[0]),
            )
            result = "replaced"
        self.connection.execute(
            "UPDATE canonical_identities SET last_gallery_update=? WHERE canonical_id=?",
            (time.time(), int(canonical_id)),
        )
        self.connection.commit()
        return result

    def merge_identity(self, keep: int, remove: int) -> None:
        """Persist a canonical merge without leaving a reloadable alias."""
        if int(keep) == int(remove):
            return
        self.connection.execute(
            "UPDATE identity_gallery SET canonical_id=? WHERE canonical_id=?",
            (int(keep), int(remove)),
        )
        self.connection.execute(
            "DELETE FROM canonical_identities WHERE canonical_id=?",
            (int(remove),),
        )
        self.connection.execute(
            "UPDATE canonical_identities SET last_gallery_update=? WHERE canonical_id=?",
            (time.time(), int(keep)),
        )
        self.connection.commit()

    def application_id_map(self) -> dict[int, str]:
        return {item["canonical_id"]: item["application_id"] for item in self.identities()}

    def stats(self) -> dict:
        identity_count = self.connection.execute("SELECT COUNT(*) FROM canonical_identities").fetchone()[0]
        gallery_count = self.connection.execute("SELECT COUNT(*) FROM identity_gallery").fetchone()[0]
        return {
            "path": str(self.path),
            "identities": int(identity_count),
            "gallery_entries": int(gallery_count),
            "max_entries_per_identity": self.max_entries_per_identity,
        }

    def close(self) -> None:
        self.connection.close()
