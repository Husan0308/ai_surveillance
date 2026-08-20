from __future__ import annotations

from typing import Any

import httpx


class MLServiceUnavailable(RuntimeError):
    pass


class MLServiceNotFound(RuntimeError):
    pass


class MLServiceClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        timeout = httpx.Timeout(timeout_seconds)
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers={"User-Agent": "ai-surveillance-api/1.0"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict[str, Any]:
        return await self._get_json("/health")

    async def cameras(self) -> dict[str, Any]:
        return await self._get_json("/cameras")

    async def detections(self, camera_id: str) -> dict[str, Any]:
        return await self._get_json(f"/detections/{camera_id}")

    async def tracks_all(self) -> dict[str, Any]:
        return await self._get_json("/tracks")

    async def tracks(self, camera_id: str) -> dict[str, Any]:
        return await self._get_json(f"/tracks/{camera_id}")

    async def _get_json(self, path: str) -> dict[str, Any]:
        try:
            response = await self._client.get(path)
        except httpx.RequestError as exc:
            raise MLServiceUnavailable(str(exc)) from exc

        if response.status_code == 404:
            detail = "resource not found"
            try:
                payload = response.json()
                if isinstance(payload, dict) and payload.get("detail"):
                    detail = str(payload["detail"])
            except ValueError:
                pass
            raise MLServiceNotFound(detail)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise MLServiceUnavailable(str(exc)) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise MLServiceUnavailable(
                f"Invalid JSON from ML service for {path}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise MLServiceUnavailable(
                f"Unexpected ML service response for {path}: expected JSON object"
            )
        return payload
