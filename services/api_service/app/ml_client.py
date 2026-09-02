from __future__ import annotations

from typing import Any

import httpx


class MLServiceUnavailable(RuntimeError):
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

    async def monitoring_snapshot(self) -> dict[str, Any]:
        return await self._get_json("/api/v1/monitoring/snapshot")

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
