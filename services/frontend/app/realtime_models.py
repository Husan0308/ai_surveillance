from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CameraRow:
    camera_id: str
    online: bool
    fps: float
    width: int
    height: int
    last_error: str
    stream_url: str


@dataclass(frozen=True)
class TrackRow:
    track_id: str
    state: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass
class CameraMetadata:
    frame_seq: int = 0
    timestamp_ns: int = 0
    source_width: int = 1
    source_height: int = 1
    online: bool = False
    fps: float = 0.0
    last_error: str = ""
    tracks: tuple[TrackRow, ...] = field(default_factory=tuple)


def parse_camera_rows(payload: dict[str, Any]) -> list[CameraRow]:
    raw = payload.get("cameras", [])
    if not isinstance(raw, list):
        return []
    out: list[CameraRow] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        camera_id = str(item.get("id", "")).strip()
        if not camera_id or camera_id in seen:
            continue
        seen.add(camera_id)
        out.append(
            CameraRow(
                camera_id=camera_id,
                online=bool(item.get("online", False)),
                fps=float(item.get("fps") or 0.0),
                width=max(1, int(item.get("width") or 1)),
                height=max(1, int(item.get("height") or 1)),
                last_error=str(item.get("last_error") or ""),
                stream_url=str(item.get("stream_url") or f"/api/v1/cameras/{camera_id}/stream.mjpg"),
            )
        )
    return out


def parse_track_message(payload: dict[str, Any]) -> tuple[str, CameraMetadata] | None:
    if payload.get("type") != "tracks":
        return None
    camera_id = str(payload.get("camera_id", "")).strip()
    if not camera_id:
        return None
    tracks: list[TrackRow] = []
    raw_tracks = payload.get("tracks", [])
    if isinstance(raw_tracks, list):
        for item in raw_tracks:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox_xyxy")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                values = tuple(float(v) for v in bbox)
            except (TypeError, ValueError):
                continue
            if values[2] <= values[0] or values[3] <= values[1]:
                continue
            tracks.append(
                TrackRow(
                    track_id=str(item.get("track_id", "")).strip() or "T-?",
                    state=str(item.get("state", "tracked")),
                    confidence=float(item.get("confidence") or 0.0),
                    bbox_xyxy=values,
                )
            )
    meta = CameraMetadata(
        frame_seq=max(0, int(payload.get("frame_seq") or 0)),
        timestamp_ns=max(0, int(payload.get("timestamp_ns") or 0)),
        source_width=max(1, int(payload.get("source_width") or 1)),
        source_height=max(1, int(payload.get("source_height") or 1)),
        online=bool(payload.get("online", False)),
        fps=float(payload.get("fps") or 0.0),
        last_error=str(payload.get("last_error") or ""),
        tracks=tuple(tracks),
    )
    return camera_id, meta


class LatestMetadataStore:
    def __init__(self) -> None:
        self._data: dict[str, CameraMetadata] = {}

    def update(self, camera_id: str, metadata: CameraMetadata) -> bool:
        old = self._data.get(camera_id)
        if old is not None:
            if metadata.timestamp_ns and old.timestamp_ns and metadata.timestamp_ns < old.timestamp_ns:
                return False
            if metadata.timestamp_ns == old.timestamp_ns and metadata.frame_seq < old.frame_seq:
                return False
        self._data[camera_id] = metadata
        return True

    def get(self, camera_id: str) -> CameraMetadata | None:
        return self._data.get(camera_id)


def letterbox_rect(
    area_x: float,
    area_y: float,
    area_width: float,
    area_height: float,
    source_width: int,
    source_height: int,
) -> tuple[float, float, float, float, float]:
    if area_width <= 0 or area_height <= 0 or source_width <= 0 or source_height <= 0:
        return area_x, area_y, 0.0, 0.0, 0.0
    scale = min(area_width / source_width, area_height / source_height)
    draw_width = source_width * scale
    draw_height = source_height * scale
    offset_x = area_x + (area_width - draw_width) * 0.5
    offset_y = area_y + (area_height - draw_height) * 0.5
    return offset_x, offset_y, draw_width, draw_height, scale


def map_bbox_to_widget(
    bbox_xyxy: tuple[float, float, float, float],
    image_rect: tuple[float, float, float, float, float],
) -> tuple[float, float, float, float]:
    offset_x, offset_y, _draw_width, _draw_height, scale = image_rect
    x1, y1, x2, y2 = bbox_xyxy
    return (
        offset_x + x1 * scale,
        offset_y + y1 * scale,
        offset_x + x2 * scale,
        offset_y + y2 * scale,
    )
