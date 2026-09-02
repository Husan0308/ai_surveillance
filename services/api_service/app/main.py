from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status

from services.camera_v11.monitoring_telemetry_ipc_v1 import DEFAULT_CAMERA_IDS, offline_snapshot
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
    version="0.1.0",
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
        return await get_ml_client(request).cameras()
    except MLServiceUnavailable as exc:
        raise service_unavailable(exc) from exc


def monitoring_degraded(reason: str) -> dict:
    return offline_snapshot(
        DEFAULT_CAMERA_IDS, status="degraded", reason=f"ml_service_unavailable:{reason}"
    )


@app.get("/api/v1/monitoring/snapshot")
async def monitoring_snapshot(request: Request) -> dict:
    try:
        return await get_ml_client(request).monitoring_snapshot()
    except MLServiceUnavailable as exc:
        return monitoring_degraded(str(exc))


@app.websocket("/ws/v1/monitoring")
async def monitoring_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            try:
                payload = await websocket.app.state.ml_client.monitoring_snapshot()
            except MLServiceUnavailable as exc:
                payload = monitoring_degraded(str(exc))
            await websocket.send_json(payload)
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, RuntimeError):
        return


def main() -> None:
    uvicorn.run(
        "services.api_service.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        access_log=False,
        timeout_graceful_shutdown=2,
    )


if __name__ == "__main__":
    main()
