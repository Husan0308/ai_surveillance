"""Publication-only adapter for the native room-pair readiness signal.

No video buffers, pipeline elements, identity decisions, or galleries enter
this module. Withheld observations are discarded, not held for later replay.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


# Match SOURCE_HEALTH_STALL_SEC in the native source-health implementation.
SOURCE_HEALTH_STALL_MS = 5_000


def read_readiness(root: Path, *, now_epoch_ms: float | None = None) -> dict[str, Any]:
    """Read the existing native signal, failing closed if its producer stopped.

    The native watchdog normally closes readiness itself. Its epoch/age fields
    also allow consumers to fail closed when the process is paused or crashes,
    without adding a second reconnect or source-readiness system.
    """
    now = time.time() * 1000.0 if now_epoch_ms is None else now_epoch_ms
    for path in (root / "run/logs/probe/readiness.json", root / "logs/probe/readiness.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("ready") is not True or payload.get("status") != "ready":
            return {**payload, "ready": False}
        try:
            elapsed_ms = max(0.0, now - float(payload["updated_epoch_ms"]))
            sources = payload["sources"]
            if {int(source["source_id"]) for source in sources} != {0, 1}:
                raise ValueError("required room-pair sources missing")
            fresh = elapsed_ms <= SOURCE_HEALTH_STALL_MS and all(
                source.get(f"{stage}_seen") is True
                and 0.0 <= float(source[f"last_{stage}_age_ms"]) + elapsed_ms <= SOURCE_HEALTH_STALL_MS
                for source in sources for stage in ("mux", "pgie", "tracker")
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return {**payload, "status": "not_ready", "ready": False,
                    "reason": "readiness_incomplete"}
        if not fresh:
            return {**payload, "status": "not_ready", "ready": False,
                    "reason": "readiness_progress_stale"}
        return payload
    return {"status": "not_ready", "ready": False, "reason": "readiness_missing"}


class PublicationBarrier:
    """A no-backlog WARMING/READY gate driven solely by native readiness."""

    def __init__(self) -> None:
        self.ready = False
        self.epoch = 0
        self.opened_monotonic_ns = 0
        self.readiness: dict[str, Any] = {"status": "not_ready", "ready": False}

    def update(self, readiness: dict[str, Any], now_ns: int) -> dict[str, Any] | None:
        self.readiness = readiness
        ready = readiness.get("ready") is True and readiness.get("status") == "ready"
        if ready == self.ready:
            return None
        previous = "READY" if self.ready else "WARMING"
        self.ready = ready
        self.epoch += 1
        # An opening always starts at current observations, including recovery.
        self.opened_monotonic_ns = int(now_ns) if ready else 0
        return {"previous": previous, **self.snapshot(), "monotonic_ns": int(now_ns),
                "native_readiness": readiness}

    def allows(self, observation: dict[str, Any]) -> bool:
        return self.ready and int(observation.get("receive_monotonic_ns") or 0) >= self.opened_monotonic_ns

    def snapshot(self) -> dict[str, Any]:
        return {"status": "READY" if self.ready else "WARMING", "ready": self.ready,
                "epoch": self.epoch, "opened_monotonic_ns": self.opened_monotonic_ns,
                "withheld_event_queue_depth": 0}
