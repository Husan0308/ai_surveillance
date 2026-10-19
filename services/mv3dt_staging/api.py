from __future__ import annotations

import asyncio
from contextlib import closing
import json
import os
import sqlite3
import struct
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from services.mv3dt_staging.adoption_aliases import AdoptionAliasIndex
from services.mv3dt_staging.event_store import CAMERA_MAP, DEFAULT_FPS, parse_world_to_map_transform
from services.mv3dt_staging.reviews import FrameReviews


def create_app(
    db_path: str | Path | None = None,
    map_path: str | Path | None = None,
    transforms_path: str | Path | None = None,
    review_path: str | Path | None = None,
    adoption_alias_path: str | Path | None = None,
) -> FastAPI:
    database = Path(db_path or os.getenv("MV3DT_STAGING_DB", "")).expanduser() if (db_path or os.getenv("MV3DT_STAGING_DB")) else None
    map_file = Path(map_path or os.getenv("MV3DT_STAGING_MAP", "")).expanduser() if (map_path or os.getenv("MV3DT_STAGING_MAP")) else None
    transform_file = Path(
        transforms_path or os.getenv("MV3DT_STAGING_TRANSFORMS", "")
    ).expanduser() if (transforms_path or os.getenv("MV3DT_STAGING_TRANSFORMS")) else None
    alias_file = Path(adoption_alias_path or os.getenv("MV3DT_STAGING_ADOPTION_ALIASES", "")).expanduser() \
        if (adoption_alias_path or os.getenv("MV3DT_STAGING_ADOPTION_ALIASES")) else None
    app = FastAPI(title="AI Surveillance — MV3DT Offline Staging", version="0.1.0")
    app.state.database = database
    app.state.map_file = map_file
    app.state.transforms_file = transform_file
    app.state.adoption_alias_file = alias_file
    reviews = FrameReviews(database, Path(review_path or os.getenv("MV3DT_STAGING_REVIEWS", "")
                                          or Path(__file__).with_name("frame_reviews.json")))

    alias_index = None
    if alias_file is not None:
        if database is None or not database.is_file():
            raise ValueError("adoption aliases require an existing offline event database")
        with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=2.0)) as connection:
            metadata_rows = dict(connection.execute("SELECT key,value FROM metadata"))
        try:
            camera_names = json.loads(metadata_rows.get("camera_mapping_json", "{}"))
            source_ids = json.loads(metadata_rows.get("camera_source_ids_json", "{}"))
            if not source_ids:
                source_ids = {sensor: source_id for sensor, (camera_id, source_id) in CAMERA_MAP.items()
                              if camera_names.get(sensor) == camera_id}
            camera_mapping = {
                sensor: (camera_id, int(source_ids[sensor]))
                for sensor, camera_id in camera_names.items()
            }
            source_log_sha256 = metadata_rows["source_log_sha256"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("offline database lacks a valid source hash or camera mapping for aliases") from exc
        alias_index = AdoptionAliasIndex.from_file(
            alias_file, expected_source_log_sha256=source_log_sha256,
            camera_mapping=camera_mapping,
        )

    def connect_readonly() -> sqlite3.Connection:
        if database is None or not database.is_file():
            raise HTTPException(status_code=503, detail="offline event store is not configured")
        uri = database.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=2.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        return connection

    def metadata(connection: sqlite3.Connection) -> dict[str, str]:
        return {row["key"]: row["value"] for row in connection.execute("SELECT key,value FROM metadata")}

    def map_transform() -> list[float] | None:
        if transform_file is None or not transform_file.is_file():
            return None
        try:
            return parse_world_to_map_transform(transform_file)
        except (OSError, ValueError):
            return None

    def map_details(connection: sqlite3.Connection) -> dict[str, Any]:
        transform = map_transform()
        dimensions = None
        if map_file is not None and map_file.is_file():
            try:
                with map_file.open("rb") as image:
                    header = image.read(24)
                if header[:8] == b"\x89PNG\r\n\x1a\n":
                    dimensions = list(struct.unpack(">II", header[16:24]))
            except OSError:
                dimensions = None
        counts = {"inside_map": None, "outside_map": None, "missing_world_position": None}
        if transform is not None and dimensions is not None:
            width, height = dimensions
            inside = outside = missing = 0
            for row in connection.execute("SELECT world_x,world_y FROM observations"):
                world_x, world_y = row[0], row[1]
                if world_x is None or world_y is None:
                    missing += 1
                    continue
                x = transform[0] * world_x + transform[1] * world_y + transform[2]
                y = transform[3] * world_x + transform[4] * world_y + transform[5]
                w = transform[6] * world_x + transform[7] * world_y + transform[8]
                if w == 0:
                    outside += 1
                    continue
                x, y = x / w, y / w
                if 0 <= x < width and 0 <= y < height:
                    inside += 1
                else:
                    outside += 1
            counts = {"inside_map": inside, "outside_map": outside,
                      "missing_world_position": missing}
        return {"url": "/assets/map.png", "width": dimensions[0] if dimensions else None,
                "height": dimensions[1] if dimensions else None,
                "transform_world_to_pixel_row_major": transform,
                "scale_is_physically_verified": False,
                "projected_observation_counts": counts}

    def camera_configuration(meta: dict[str, str]) -> list[dict[str, Any]]:
        try:
            names = json.loads(meta.get("camera_mapping_json", "{}"))
            source_ids = json.loads(meta.get("camera_source_ids_json", "{}"))
            if not source_ids:
                source_ids = {sensor: source_id for sensor, (camera_id, source_id) in CAMERA_MAP.items()
                              if names.get(sensor) == camera_id}
            cameras = [{"mv3dt_sensor": sensor, "camera_id": camera,
                        "source_id": int(source_ids[sensor])}
                       for sensor, camera in names.items()]
        except (TypeError, ValueError, json.JSONDecodeError):
            cameras = []
        return sorted(cameras, key=lambda item: item["source_id"])

    @app.get("/health")
    def health() -> dict[str, Any]:
        try:
            with closing(connect_readonly()) as connection:
                messages = int(connection.execute("SELECT COUNT(*) FROM frame_messages").fetchone()[0])
                observations = int(connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0])
            return {"service": "mv3dt_offline_staging", "status": "ok", "mode": "offline_replay",
                    "event_messages": messages, "observations": observations,
                    "display_aliases_available": alias_index is not None,
                    "display_alias_count": len(alias_index.rules) if alias_index is not None else 0}
        except (HTTPException, sqlite3.Error) as exc:
            detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
            return {"service": "mv3dt_offline_staging", "status": "not_ready", "reason": detail}

    @app.get("/api/v1/summary")
    def summary() -> dict[str, Any]:
        with closing(connect_readonly()) as connection:
            meta = metadata(connection)
            cameras = []
            camera_config = camera_configuration(meta)
            expected_frames = int(meta.get("expected_frames", "3600"))
            for configured in camera_config:
                camera_id, source_id = configured["camera_id"], configured["source_id"]
                row = connection.execute(
                    "SELECT COUNT(*) AS messages, MIN(frame_id) AS first_frame, MAX(frame_id) AS last_frame "
                    "FROM frame_messages WHERE camera_id=?", (camera_id,),
                ).fetchone()
                count = int(row["messages"])
                within_range = int(connection.execute(
                    "SELECT COUNT(*) FROM frame_messages WHERE camera_id=? AND frame_id>=0 AND frame_id<?",
                    (camera_id, expected_frames),
                ).fetchone()[0])
                obs = connection.execute(
                    "SELECT COUNT(*) AS count, COUNT(DISTINCT mv3dt_global_id) AS ids, "
                    "SUM(CASE WHEN world_x IS NULL OR world_y IS NULL THEN 1 ELSE 0 END) AS missing_world, "
                    "SUM(CASE WHEN mv3dt_object_confidence IS NULL THEN 1 ELSE 0 END) AS missing_confidence "
                    "FROM observations WHERE camera_id=?", (camera_id,),
                ).fetchone()
                cameras.append({
                    "camera_id": camera_id,
                    "source_id": source_id,
                    "source_label": configured["mv3dt_sensor"],
                    "frame_messages": count,
                    "first_observed_frame_id": row["first_frame"],
                    "last_observed_frame_id": row["last_frame"],
                    "missing_frame_messages_in_0_to_expected_minus_1": expected_frames - within_range,
                    "observations": int(obs["count"]),
                    "distinct_mv3dt_output_ids": int(obs["ids"]),
                    "observations_missing_world_position": int(obs["missing_world"] or 0),
                    "observations_missing_mv3dt_confidence": int(obs["missing_confidence"] or 0),
                })
            total_messages = int(connection.execute("SELECT COUNT(*) FROM frame_messages").fetchone()[0])
            total_observations = int(connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0])
            ids_by_camera = {
                row["camera_id"]: set()
                for row in connection.execute("SELECT DISTINCT camera_id FROM observations")
            }
            for row in connection.execute("SELECT DISTINCT camera_id,mv3dt_global_id FROM observations"):
                ids_by_camera.setdefault(row["camera_id"], set()).add(str(row["mv3dt_global_id"]))
            configured_cameras = [item["camera_id"] for item in camera_config]
            shared_ids = (sorted(set.intersection(*(ids_by_camera.get(camera, set()) for camera in configured_cameras)))
                          if configured_cameras else [])
            min_frame, max_frame = connection.execute("SELECT MIN(frame_id),MAX(frame_id) FROM frame_messages").fetchone()
            return {
                "mode": "offline_replay",
                "experimental": True,
                "warning": (
                    "MV3DT output IDs and projected positions are experimental. Shared labels do not prove identity accuracy; "
                    "world scale and ground-plane accuracy are unverified. This is not connected to production identity logic."
                ),
                "camera_mapping": [
                    {**item, **({"location": "upper-right"} if item["camera_id"] == "CAM-01" else
                                {"location": "lower-left"} if item["camera_id"] == "CAM-04" else {})}
                    for item in camera_config
                ],
                "event_messages": total_messages,
                "observations": total_observations,
                "distinct_mv3dt_output_ids_union": len(set.union(*(
                    ids_by_camera.get(camera, set()) for camera in configured_cameras
                ))) if configured_cameras else 0,
                "shared_output_ids": shared_ids,
                "frame_id_range": [min_frame, max_frame],
                "nominal_fps": float(meta.get("nominal_fps", DEFAULT_FPS)),
                "recording_frame_count": expected_frames,
                "media_time_semantics": meta.get("media_time_semantics"),
                "event_timestamp_semantics": meta.get("timestamp_semantics"),
                "source_pts_in_event_log": False,
                "source_log_sha256": meta.get("source_log_sha256"),
                "display_aliases": {
                    "available": alias_index is not None,
                    "count": len(alias_index.rules) if alias_index is not None else 0,
                    "source_hash_bound": alias_index is not None,
                    "scope": "offline replay only" if alias_index is not None else None,
                    "note": ("Presentation-only ADOPTED_ID aliases; raw MV3DT IDs and source observations are retained."
                             if alias_index is not None else "No adoption alias manifest configured."),
                },
                "provenance": {key.removeprefix("provenance_"): value for key, value in meta.items()
                               if key.startswith("provenance_")},
                "map": map_details(connection),
                "cameras": cameras,
            }

    @app.get("/api/v1/observations")
    def observations(
        camera_id: str | None = None,
        frame_id: int | None = None,
        limit: int = Query(default=250, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        if camera_id is not None:
            with closing(connect_readonly()) as connection:
                allowed_camera_ids = {item["camera_id"] for item in camera_configuration(metadata(connection))}
            if camera_id not in allowed_camera_ids:
                raise HTTPException(status_code=400, detail="camera_id is not part of this offline dataset")
        where, params = [], []
        if camera_id:
            where.append("camera_id=?")
            params.append(camera_id)
        if frame_id is not None:
            where.append("frame_id=?")
            params.append(frame_id)
        clause = " WHERE " + " AND ".join(where) if where else ""
        with closing(connect_readonly()) as connection:
            rows = connection.execute(
                "SELECT camera_id,source_id,frame_id,mv3dt_global_id,event_timestamp,media_time_seconds_nominal,"
                "object_type,mv3dt_object_confidence,bbox_json,foot_x_px,foot_y_px,world_x,world_y FROM observations" +
                clause + " ORDER BY frame_id,source_id,mv3dt_global_id LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
            count = int(connection.execute("SELECT COUNT(*) FROM observations" + clause, params).fetchone()[0])
        return {"count": count, "limit": limit, "offset": offset,
                "observations": [_row_to_observation(row, alias_index) for row in rows]}

    @app.get("/api/v1/frames/{frame_id}")
    def frame(frame_id: int) -> dict[str, Any]:
        with closing(connect_readonly()) as connection:
            batch = _read_frame_batch(connection, frame_id)
            batch["visual_review"] = reviews.for_frame(metadata(connection), batch)
            batch["observations"] = [_row_to_observation(row, alias_index)
                                     for row in _frame_observation_rows(connection, frame_id)]
            batch["display_aliasing_available"] = alias_index is not None
        return batch

    @app.get("/assets/map.png")
    def map_image() -> FileResponse:
        if map_file is None or not map_file.is_file():
            raise HTTPException(status_code=404, detail="Dev Room map is unavailable")
        return FileResponse(map_file, media_type="image/png", filename="dev_room_map.png")

    @app.websocket("/ws/replay")
    async def replay(websocket: WebSocket) -> None:
        try:
            speed = float(websocket.query_params.get("speed", "5"))
            start_frame = max(0, int(websocket.query_params.get("start_frame", "0")))
            end_arg = websocket.query_params.get("end_frame")
            end_frame = int(end_arg) if end_arg is not None else None
            if not 0.1 <= speed <= 100.0 or (end_frame is not None and end_frame < start_frame):
                await websocket.close(code=1008, reason="invalid replay range/speed")
                return
            connection = connect_readonly()
            try:
                meta = metadata(connection)
                frame_ids = [int(row[0]) for row in connection.execute(
                    "SELECT DISTINCT frame_id FROM frame_messages WHERE frame_id>=? "
                    "AND (? IS NULL OR frame_id<=?) ORDER BY frame_id",
                    (start_frame, end_frame, end_frame),
                )]
                if not frame_ids:
                    await websocket.close(code=1008, reason="no events in selected range")
                    return
                await websocket.accept()
                await websocket.send_json({
                    "type": "start",
                    "mode": "offline_replay",
                    "experimental": True,
                    "warning": "MV3DT IDs/positions are experimental; no source PTS or verified metric scale is available.",
                    "nominal_fps": float(meta.get("nominal_fps", DEFAULT_FPS)),
                    "frame_id_range": [frame_ids[0], frame_ids[-1]],
                    "media_time_semantics": meta.get("media_time_semantics"),
                    "event_timestamp_semantics": meta.get("timestamp_semantics"),
                    "source_pts_in_event_log": False,
                    "display_aliasing_available": alias_index is not None,
                    "display_alias_count": len(alias_index.rules) if alias_index is not None else 0,
                    "display_alias_scope": "offline retrospective only" if alias_index is not None else None,
                    "map": map_details(connection),
                    "transform_world_to_pixel_row_major": map_transform(),
                })
                previous: int | None = None
                for frame_id in frame_ids:
                    if previous is not None:
                        await asyncio.sleep(max(0.0, (frame_id - previous) / DEFAULT_FPS / speed))
                    batch = _read_frame_batch(connection, frame_id)
                    batch["observations"] = [_row_to_observation(row, alias_index)
                                             for row in _frame_observation_rows(connection, frame_id)]
                    batch["display_aliasing_available"] = alias_index is not None
                    batch["visual_review"] = reviews.for_frame(meta, batch)
                    await websocket.send_json(batch)
                    previous = frame_id
                await websocket.send_json({"type": "complete", "last_frame_id": frame_ids[-1],
                                           "frames_sent": len(frame_ids)})
                await websocket.close(code=1000, reason="offline replay complete")
            finally:
                connection.close()
        except (HTTPException, sqlite3.Error, OSError, ValueError) as exc:
            if websocket.client_state.name == "CONNECTED":
                await websocket.close(code=1011, reason=str(exc)[:120])
        except (WebSocketDisconnect, RuntimeError):
            return

    return app


def _row_to_observation(row: sqlite3.Row, alias_index: AdoptionAliasIndex | None = None) -> dict[str, Any]:
    observation = {
        "camera_id": row["camera_id"],
        "source_id": int(row["source_id"]),
        "frame_id": int(row["frame_id"]),
        "mv3dt_global_id": str(row["mv3dt_global_id"]),
        "event_timestamp": row["event_timestamp"],
        "media_time_seconds_nominal": float(row["media_time_seconds_nominal"]),
        "object_type": row["object_type"],
        "mv3dt_object_confidence": row["mv3dt_object_confidence"],
        "bbox": json.loads(row["bbox_json"]) if row["bbox_json"] else None,
        "foot_location_2d_px": [row["foot_x_px"], row["foot_y_px"]]
        if row["foot_x_px"] is not None and row["foot_y_px"] is not None else None,
        "world_position_unverified_units": [row["world_x"], row["world_y"]]
        if row["world_x"] is not None and row["world_y"] is not None else None,
    }
    raw_id = observation["mv3dt_global_id"]
    resolved = (alias_index.resolve(camera_id=observation["camera_id"],
                                    source_id=observation["source_id"], raw_id=raw_id,
                                    frame_id=observation["frame_id"])
                if alias_index is not None else {"display_id": raw_id, "applied": False, "lineage": []})
    observation.update({
        "raw_global_id": raw_id,
        "display_global_id": resolved["display_id"],
        "display_alias_applied": resolved["applied"],
        "display_alias_lineage": resolved["lineage"],
    })
    return observation


def _frame_observation_rows(connection: sqlite3.Connection, frame_id: int) -> list[sqlite3.Row]:
    return connection.execute(
        "SELECT camera_id,source_id,frame_id,mv3dt_global_id,event_timestamp,media_time_seconds_nominal,"
        "object_type,mv3dt_object_confidence,bbox_json,foot_x_px,foot_y_px,world_x,world_y FROM observations "
        "WHERE frame_id=? ORDER BY source_id,mv3dt_global_id", (frame_id,),
    ).fetchall()


def _read_frame_batch(connection: sqlite3.Connection, frame_id: int) -> dict[str, Any]:
    messages = connection.execute(
        "SELECT camera_id,source_id,frame_id,event_timestamp,object_count FROM frame_messages "
        "WHERE frame_id=? ORDER BY source_id", (frame_id,),
    ).fetchall()
    rows = connection.execute(
        "SELECT camera_id,source_id,frame_id,mv3dt_global_id,event_timestamp,media_time_seconds_nominal,"
        "object_type,mv3dt_object_confidence,bbox_json,foot_x_px,foot_y_px,world_x,world_y FROM observations "
        "WHERE frame_id=? ORDER BY source_id,mv3dt_global_id", (frame_id,),
    ).fetchall()
    return {"type": "frame", "frame_id": frame_id,
            "nominal_media_time_seconds": frame_id / DEFAULT_FPS,
            "frame_messages": [dict(row) for row in messages],
            "observations": [_row_to_observation(row) for row in rows]}


def main() -> None:
    host = os.getenv("MV3DT_STAGING_HOST", "127.0.0.1")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("offline staging is loopback-only; MV3DT_STAGING_HOST must be localhost")
    port = int(os.getenv("MV3DT_STAGING_PORT", "18120"))
    uvicorn.run("services.mv3dt_staging.api:app", host=host, port=port,
                reload=False, access_log=False, log_level="info")


app = create_app()


if __name__ == "__main__":
    main()

