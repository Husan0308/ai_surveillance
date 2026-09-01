from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import quote

import httpx


class MLServiceUnavailable(RuntimeError):
    pass


class MLServiceClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self._timeout_seconds = float(timeout_seconds)
        timeout = httpx.Timeout(self._timeout_seconds)
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

    async def tracks(self) -> dict[str, Any]:
        return await self._get_json("/tracks")

    async def video_stream(self, camera_id: str) -> AsyncIterator[bytes]:
        path = f"/video/{quote(str(camera_id), safe='')}"
        timeout = httpx.Timeout(
            connect=self._timeout_seconds,
            read=None,
            write=self._timeout_seconds,
            pool=self._timeout_seconds,
        )
        try:
            async with self._client.stream("GET", path, timeout=timeout) as response:
                response.raise_for_status()
                async for chunk in response.aiter_raw():
                    if chunk:
                        yield chunk
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            raise MLServiceUnavailable(str(exc)) from exc

    async def _get_json(self, path: str) -> dict[str, Any]:
        try:
            response = await self._client.get(path)
            response.raise_for_status()
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            raise MLServiceUnavailable(str(exc)) from exc

        payload = response.json()
        if not isinstance(payload, dict):
            raise MLServiceUnavailable(
                f"Unexpected ML service response for {path}: expected JSON object"
            )
        return payload
