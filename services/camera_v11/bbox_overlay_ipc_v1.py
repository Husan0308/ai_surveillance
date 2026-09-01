from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

DEFAULT_BBOX_STATE = Path("/dev/shm/ai_surveillance/v11_bbox_overlay_v1.json")
SCHEMA = "camera_v11_bbox_overlay_v1"


def local_track_number(track_id: str) -> int:
    text = str(track_id)
    marker = text.rfind("-T")
    if marker < 0:
        raise ValueError(f"invalid V11 local track id: {track_id!r}")
    value = int(text[marker + 2 :])
    if value < 0:
        raise ValueError(f"invalid V11 local track id: {track_id!r}")
    return value


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, float(value)))


def predict_bbox_norm(
    bbox_norm: Iterable[float],
    velocity_norm_s: Iterable[float],
    dt_sec: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in bbox_norm)
    vx, vy, vw, vh = (float(v) for v in velocity_norm_s)
    dt = max(0.0, min(0.45, float(dt_sec)))
    cx = 0.5 * (x1 + x2) + vx * dt
    cy = 0.5 * (y1 + y2) + vy * dt
    width = max(0.002, (x2 - x1) + vw * dt)
    height = max(0.002, (y2 - y1) + vh * dt)
    px1 = _clamp(cx - 0.5 * width)
    py1 = _clamp(cy - 0.5 * height)
    px2 = _clamp(cx + 0.5 * width)
    py2 = _clamp(cy + 0.5 * height)
    return px1, py1, max(px1, px2), max(py1, py2)


def tracker_box_to_display(
    bbox_norm: Iterable[float],
    velocity_norm_s: Iterable[float],
    dt_sec: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Map V11 Step3's 672x384 padded tracker coordinates to display pixels."""
    x1n, y1n, x2n, y2n = predict_bbox_norm(bbox_norm, velocity_norm_s, dt_sec)
    # Detector content is 672x378 at rows 3..380 inside the 672x384 TRT canvas.
    y1n = _clamp((y1n * 384.0 - 3.0) / 378.0)
    y2n = _clamp((y2n * 384.0 - 3.0) / 378.0)
    w = max(1.0, float(width))
    h = max(1.0, float(height))
    return x1n * w, y1n * h, x2n * w, y2n * h


class BboxStateWriter:
    """Atomic tiny tmpfs snapshot. There is never a metadata queue/backlog."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path or os.getenv("V11_BBOX_STATE_PATH", str(DEFAULT_BBOX_STATE)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._generation = 0
        self._cameras: dict[str, dict[str, Any]] = {}

    def reset(self) -> None:
        self._generation = 0
        self._cameras.clear()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def publish(self, camera_id: str, captured_ns: int, tracks: list[dict[str, Any]]) -> None:
        self._generation += 1
        self._cameras[str(camera_id)] = {
            "captured_ns": int(captured_ns),
            "tracks": tracks,
        }
        payload = {
            "schema": SCHEMA,
            "generation": self._generation,
            "written_ns": time.monotonic_ns(),
            "cameras": self._cameras,
        }
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, self.path)


class BboxStateReader:
    """Non-blocking-by-design cache: parse only when the atomic file changes."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path or os.getenv("V11_BBOX_STATE_PATH", str(DEFAULT_BBOX_STATE)))
        self._mtime_ns = -1
        self._payload: dict[str, Any] = {"schema": SCHEMA, "generation": 0, "cameras": {}}
        self.errors = 0

    def snapshot(self) -> dict[str, Any]:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return self._payload
        except OSError:
            self.errors += 1
            return self._payload
        if stat.st_mtime_ns == self._mtime_ns:
            return self._payload
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema") != SCHEMA or not isinstance(payload.get("cameras"), dict):
                raise ValueError("bad bbox snapshot schema")
        except (OSError, ValueError, json.JSONDecodeError):
            self.errors += 1
            return self._payload
        self._mtime_ns = stat.st_mtime_ns
        self._payload = payload
        return self._payload

    def camera_tracks(
        self,
        camera_id: str,
        *,
        now_ns: int | None = None,
        stale_sec: float = 1.10,
        width: int = 640,
        height: int = 360,
    ) -> list[tuple[int, float, float, float, float, float]]:
        payload = self.snapshot()
        row = payload.get("cameras", {}).get(str(camera_id))
        if not isinstance(row, dict):
            return []
        captured_ns = int(row.get("captured_ns") or 0)
        if captured_ns <= 0:
            return []
        current_ns = int(now_ns if now_ns is not None else time.monotonic_ns())
        age_sec = max(0.0, (current_ns - captured_ns) / 1_000_000_000.0)
        if age_sec > max(0.2, float(stale_sec)):
            return []
        output: list[tuple[int, float, float, float, float, float]] = []
        for track in row.get("tracks", []):
            if not isinstance(track, dict):
                continue
            try:
                local_id = int(track["local_id"])
                score = float(track.get("confidence", 0.0))
                if not math.isfinite(score):
                    continue
                box = tracker_box_to_display(
                    track["bbox_norm"],
                    track.get("velocity_norm_s", (0.0, 0.0, 0.0, 0.0)),
                    age_sec,
                    width,
                    height,
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            x1, y1, x2, y2 = box
            if x2 - x1 < 2.0 or y2 - y1 < 2.0:
                continue
            output.append((local_id, x1, y1, x2, y2, min(1.0, max(0.0, score))))
        return output
