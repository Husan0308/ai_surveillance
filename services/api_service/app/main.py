from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status

from services.api_service.app.config import load_settings
from services.api_service.app.ml_client import (
    MLServiceClient,
    MLServiceNotFound,
    MLServiceUnavailable,
)

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
    version="0.3.0",
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


def resource_not_found(camera_id: str, exc: MLServiceNotFound) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "resource": "camera",
            "camera_id": camera_id,
            "reason": str(exc),
        },
    )


@app.get("/health")
async def health() -> dict:
    return {
        "service": "api_service",
        "status": "ok",
        "ml_base_url": settings.ml_base_url,
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


@app.get("/api/v1/tracks")
async def tracks_all(request: Request) -> dict:
    try:
        return await get_ml_client(request).tracks_all()
    except MLServiceUnavailable as exc:
        raise service_unavailable(exc) from exc


@app.get("/api/v1/cameras/{camera_id}/detections")
async def camera_detections(camera_id: str, request: Request) -> dict:
    try:
        return await get_ml_client(request).detections(camera_id)
    except MLServiceNotFound as exc:
        raise resource_not_found(camera_id, exc) from exc
    except MLServiceUnavailable as exc:
        raise service_unavailable(exc) from exc


@app.get("/api/v1/cameras/{camera_id}/tracks")
async def camera_tracks(camera_id: str, request: Request) -> dict:
    try:
        return await get_ml_client(request).tracks(camera_id)
    except MLServiceNotFound as exc:
        raise resource_not_found(camera_id, exc) from exc
    except MLServiceUnavailable as exc:
        raise service_unavailable(exc) from exc


def main() -> None:
    uvicorn.run(
        "services.api_service.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
