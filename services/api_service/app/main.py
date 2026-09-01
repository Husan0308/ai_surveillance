from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse

from services.api_service.app.config import load_settings
from services.api_service.app.ml_client import MLServiceClient, MLServiceUnavailable

settings = load_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = MLServiceClient(
        base_url=settings.ml_base_url,
        timeout_seconds=settings.ml_timeout_seconds,
    )
    app.state.ml_client = client
    try:
        yield
    finally:
        await client.close()


app = FastAPI(
    title="AI Surveillance API Service",
    version="0.2.0",
    lifespan=lifespan,
)


def get_ml_client(request: Request) -> MLServiceClient:
    return request.app.state.ml_client


def service_unavailable(exc: MLServiceUnavailable) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "service": "ml_service",
            "status": "unavailable",
            "reason": str(exc),
        },
    )


@app.get("/health")
async def health() -> dict:
    return {
        "service": "api_service",
        "status": "ok",
    }


@app.get("/api/v1/ml/health")
async def ml_health(request: Request) -> dict:
    try:
        return await get_ml_client(request).health()
    except MLServiceUnavailable as exc:
        raise service_unavailable(exc) from exc


@app.get("/api/v1/cameras")
async def cameras(request: Request) -> dict:
    try:
        payload = await get_ml_client(request).cameras()
    except MLServiceUnavailable as exc:
        raise service_unavailable(exc) from exc

    rows = payload.get("cameras", [])
    if not isinstance(rows, list):
        rows = []
    items: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        camera_id = str(item.get("id", "")).strip()
        if not camera_id:
            continue
        item["stream_url"] = f"/api/v1/cameras/{camera_id}/stream.mjpg"
        items.append(item)
    return {"count": len(items), "cameras": items}


@app.get("/api/v1/cameras/{camera_id}/stream.mjpg")
async def camera_stream(camera_id: str, request: Request):
    client = get_ml_client(request)

    async def stream():
        try:
            async for chunk in client.video_stream(camera_id):
                yield chunk
        except MLServiceUnavailable:
            return

    return StreamingResponse(
        stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _monitor_fingerprint(item: dict) -> tuple:
    tracks = item.get("tracks", [])
    track_fp = []
    if isinstance(tracks, list):
        for row in tracks:
            if not isinstance(row, dict):
                continue
            bbox = row.get("bbox_xyxy") or ()
            track_fp.append(
                (
                    str(row.get("track_id", "")),
                    str(row.get("state", "")),
                    tuple(round(float(v), 2) for v in bbox) if len(bbox) == 4 else (),
                )
            )
    return (
        int(item.get("frame_seq") or 0),
        bool(item.get("online", False)),
        round(float(item.get("fps") or 0.0), 1),
        tuple(track_fp),
    )


@app.websocket("/api/v1/ws/monitoring")
async def monitoring(websocket: WebSocket):
    await websocket.accept()
    client: MLServiceClient = websocket.app.state.ml_client
    previous: dict[str, tuple] = {}
    try:
        while True:
            try:
                payload = await client.tracks()
                items = payload.get("items", [])
                if not isinstance(items, list):
                    items = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    camera_id = str(item.get("camera_id", "")).strip()
                    if not camera_id:
                        continue
                    fingerprint = _monitor_fingerprint(item)
                    if previous.get(camera_id) == fingerprint:
                        continue
                    previous[camera_id] = fingerprint
                    await websocket.send_json({"type": "tracks", **item})
            except MLServiceUnavailable as exc:
                await websocket.send_json(
                    {"type": "service", "service": "ml_service", "status": "unavailable", "reason": str(exc)}
                )
            await asyncio.sleep(0.10)
    except (WebSocketDisconnect, RuntimeError):
        return


def main() -> None:
    uvicorn.run(
        "services.api_service.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
