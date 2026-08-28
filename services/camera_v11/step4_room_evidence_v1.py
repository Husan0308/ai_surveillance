from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np


class V11RoomEvidenceWriterV1:
    """Debug-only bounded crop writer for manually auditing Room Identity fusion."""

    def __init__(
        self,
        root: str,
        *,
        max_per_member: int = 4,
        min_interval_sec: float = 0.75,
        max_cached_members: int = 64,
    ) -> None:
        text = str(root).strip()
        self.enabled = bool(text)
        self.root = Path(text).expanduser() if text else None
        self.max_per_member = max(1, min(20, int(max_per_member)))
        self.min_interval_ns = int(max(0.1, float(min_interval_sec)) * 1_000_000_000.0)
        self.max_cached_members = max(8, int(max_cached_members))
        self._cache: OrderedDict[
            tuple[str, str], tuple[np.ndarray, str, int]
        ] = OrderedDict()
        self._saved: dict[tuple[str, str, str], int] = {}
        self._last_saved_ns: dict[tuple[str, str, str], int] = {}
        self.write_ok = 0
        self.write_fail = 0
        if self.enabled and self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    def remember(
        self,
        *,
        camera_id: str,
        camera_identity: str,
        local_track: str,
        crop_bgr: np.ndarray,
        captured_ns: int,
    ) -> None:
        if not self.enabled or not camera_identity or crop_bgr.size == 0:
            return
        key = (str(camera_id), str(camera_identity))
        self._cache[key] = (
            np.ascontiguousarray(crop_bgr.copy()),
            str(local_track),
            int(captured_ns),
        )
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_cached_members:
            self._cache.popitem(last=False)

    def _write_one(
        self,
        *,
        room_id: str,
        room_identity: str,
        member: tuple[str, str],
        force: bool,
    ) -> bool:
        if not self.enabled or self.root is None:
            return False
        cached = self._cache.get(member)
        if cached is None:
            return False
        crop, local_track, captured_ns = cached
        camera_id, camera_identity = member
        save_key = (str(room_identity), camera_id, camera_identity)
        count = self._saved.get(save_key, 0)
        if count >= self.max_per_member:
            return False
        previous_ns = self._last_saved_ns.get(save_key, 0)
        if not force and previous_ns and captured_ns - previous_ns < self.min_interval_ns:
            return False

        folder = self.root / str(room_identity)
        folder.mkdir(parents=True, exist_ok=True)
        index = count + 1
        safe_track = str(local_track).replace("/", "_")
        filename = f"{camera_id}__{camera_identity}__{index:02d}__{safe_track}.jpg"
        path = folder / filename

        try:
            import cv2

            ok = bool(cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]))
        except Exception:
            ok = False
        if not ok:
            self.write_fail += 1
            return False

        manifest = folder / "manifest.tsv"
        if not manifest.exists():
            manifest.write_text(
                "room\troom_identity\tcamera\tcamera_identity\tlocal_track\tfile\n",
                encoding="utf-8",
            )
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{room_id}\t{room_identity}\t{camera_id}\t{camera_identity}\t"
                f"{local_track}\t{filename}\n"
            )
        self._saved[save_key] = index
        self._last_saved_ns[save_key] = captured_ns
        self.write_ok += 1
        return True

    def capture_room(
        self,
        *,
        room_id: str,
        room_identity: str,
        members: set[tuple[str, str]],
        current_member: tuple[str, str],
    ) -> list[str]:
        if not self.enabled or len(members) < 2:
            return []
        saved: list[str] = []
        # Backfill one cached crop for every fused camera identity immediately so a
        # newly-created folder already contains evidence from both cameras.
        for member in sorted(members):
            save_key = (str(room_identity), member[0], member[1])
            if self._saved.get(save_key, 0) == 0 and self._write_one(
                room_id=room_id,
                room_identity=room_identity,
                member=member,
                force=True,
            ):
                saved.append(f"{member[0]}/{member[1]}")

        # Then collect a few temporally separated crops from the currently observed
        # member. This is bounded by max_per_member and does not grow indefinitely.
        if self._write_one(
            room_id=room_id,
            room_identity=room_identity,
            member=current_member,
            force=False,
        ):
            saved.append(f"{current_member[0]}/{current_member[1]}")
        return saved

    def snapshot(self) -> dict[str, int]:
        return {
            "enabled": int(self.enabled),
            "cached_members": len(self._cache),
            "write_ok": self.write_ok,
            "write_fail": self.write_fail,
        }
