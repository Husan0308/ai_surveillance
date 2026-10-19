"""Explicit visual-review annotations, bound to unchanged offline evidence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from services.mv3dt_staging.event_store import sha256_file


class FrameReviews:
    def __init__(self, database: Path | None, document_path: Path) -> None:
        self.document: dict[str, Any] = {}
        try:
            document = json.loads(document_path.read_text(encoding="utf-8"))
            if (isinstance(document, dict) and database is not None and database.is_file() and document.get("schema_version") == 1
                    and sha256_file(database) == document.get("database_sha256")):
                self.document = document
        except (OSError, ValueError, TypeError):
            pass

    def for_frame(self, metadata: dict[str, str], batch: dict[str, Any]) -> dict[str, Any] | None:
        if not self.document:
            return None
        expected = self.document.get("expected_metadata", {})
        if not expected or any(metadata.get(key) != value for key, value in expected.items()):
            return None
        frame_id = batch["frame_id"]
        review = self.document.get("frames", {}).get(str(frame_id))
        if not isinstance(review, dict):
            return None
        rows = batch.get("observations", [])
        actual = {(str(row["camera_id"]), str(row["mv3dt_global_id"])) for row in rows
                  if row.get("frame_id") == frame_id}
        required = {tuple(item) for item in review.get("expected_observations", [])}
        if actual != required or len(rows) != len(required):
            return None
        ids = {item[1] for item in actual}
        seen: set[str] = set()
        aliases: dict[str, str] = {}
        groups = review.get("people_groups", [])
        if not groups:
            return None
        for group in groups:
            primary = group.get("primary_global_id")
            members = group.get("member_global_ids", [])
            if primary not in members or not members or seen.intersection(members):
                return None
            seen.update(members)
            aliases.update({member: primary for member in members if member != primary})
        if seen != ids:
            return None
        return {"frame_id": frame_id, "confirmed_visible_people": len(groups),
                "duplicate_global_ids": aliases, "people_groups": groups,
                "note": review["note"], "scope": "exact reviewed frame only",
                "evidence": review.get("evidence", []), "verified_for_source": True}

