from __future__ import annotations

import atexit
import http.client
import json
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from . import dashboard as ui
from .mmap_frame_reader import SmoothMmapFrameReader


ROOT = Path(__file__).resolve().parents[3]
ML_HOST = "127.0.0.1"
ML_PORT = 8001


class DashboardMmapFrameReader:
    """Dashboard-compatible wrapper around the local mmap latest-frame reader.

    The old dashboard reader polled /frame/CAM-XX JPEG endpoints. The camera
    baseline no longer exposes those endpoints; frames are published through
    shared mmap instead. This wrapper preserves the tiny interface expected by
    DashboardWindow while removing HTTP/JPEG from the video hot path.
    """

    def __init__(self, camera_id: str):
        self.camera_id = str(camera_id)
        self._reader = SmoothMmapFrameReader(self.camera_id)

    @property
    def frames(self) -> int:
        return int(self._reader.frames)

    @property
    def reconnects(self) -> int:
        return int(self._reader.reconnects)

    @property
    def last_frame_age_ms(self) -> float:
        return float(self._reader.last_frame_age_ms)

    def start(self):
        self._reader.start()

    def stop(self):
        self._reader.stop()

    def latest(self):
        return self._reader.latest()


class ResilientRealtimeState:
    """Keep the UI connected when optional Phase-2 endpoints are unavailable."""

    def __init__(self):
        import threading

        self._threading = threading
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self.state = {
            "connected": False,
            "health": {},
            "detections": {},
            "reid": {},
            "room_mapping": {},
        }
        self.recent = deque(maxlen=30)
        self.events = deque(maxlen=100)
        self._seen = {}

    def start(self):
        self._thread = self._threading.Thread(
            target=self._run,
            name="ui-state-resilient",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop.set()

    def snapshot(self):
        with self._lock:
            return dict(self.state), list(self.recent), list(self.events)

    @staticmethod
    def _json(connection, path):
        connection.request(
            "GET",
            path,
            headers={"Connection": "keep-alive", "Cache-Control": "no-cache"},
        )
        response = connection.getresponse()
        payload = response.read()
        if response.status != 200:
            raise RuntimeError(response.status)
        return json.loads(payload.decode("utf-8"))

    @staticmethod
    def _optional_json(connection, path, default):
        try:
            return ResilientRealtimeState._json(connection, path)
        except Exception:
            try:
                connection.close()
            except Exception:
                pass
            return default

    def _observe(self, reid_payload):
        cameras = ((reid_payload.get("state") or {}).get("cameras") or {})
        now = datetime.now()
        for camera_id, tracks in cameras.items():
            for track in tracks or []:
                gid = str(track.get("global_id") or "")
                if not gid:
                    continue
                local_id = int(track.get("local_id") or 0)
                key = (camera_id, local_id)
                old = self._seen.get(key)
                self._seen[key] = gid
                if old == gid:
                    continue
                entry = {
                    "time": now.strftime("%H:%M:%S"),
                    "camera": camera_id,
                    "global_id": gid,
                    "similarity": track.get("similarity"),
                    "reason": str(track.get("reason") or "detected"),
                }
                self.recent.appendleft(entry)
                self.events.appendleft(entry)

    def _run(self):
        while not self._stop.is_set():
            connection = None
            try:
                connection = http.client.HTTPConnection(ML_HOST, ML_PORT, timeout=1.0)
                health = self._json(connection, "/health")

                # Optional endpoints are queried with fresh connections so a
                # 404/close cannot poison the health connection or mark cameras
                # offline. Phase 1 legitimately has no ReID/room mapping.
                def optional(path, default):
                    conn = None
                    try:
                        conn = http.client.HTTPConnection(ML_HOST, ML_PORT, timeout=0.8)
                        return self._json(conn, path)
                    except Exception:
                        return default
                    finally:
                        if conn is not None:
                            try:
                                conn.close()
                            except Exception:
                                pass

                detections = optional("/detections", {"enabled": False, "cameras": {}})
                reid = optional("/reid", {})
                room_mapping = optional("/room-mapping", {})
                self._observe(reid)
                with self._lock:
                    self.state = {
                        "connected": True,
                        "health": health,
                        "detections": detections,
                        "reid": reid,
                        "room_mapping": room_mapping,
                    }
            except Exception:
                with self._lock:
                    self.state = {**self.state, "connected": False}
            finally:
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass
            self._stop.wait(0.5)


_backend_process: subprocess.Popen | None = None


def _backend_health() -> dict | None:
    conn = None
    try:
        conn = http.client.HTTPConnection(ML_HOST, ML_PORT, timeout=0.5)
        conn.request("GET", "/health", headers={"Connection": "close"})
        response = conn.getresponse()
        payload = response.read()
        if response.status != 200:
            return None
        return json.loads(payload.decode("utf-8"))
    except Exception:
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _stop_local_backend() -> None:
    global _backend_process
    process = _backend_process
    _backend_process = None
    if process is None or process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=3.0)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def ensure_mmap_backend() -> dict | None:
    """Start the local mmap publisher only when port 8001 has no backend."""
    global _backend_process

    health = _backend_health()
    if health is not None:
        print(
            f"DASHBOARD_MMAP backend already running mode={health.get('mode')} "
            f"online={health.get('online')}/{health.get('total')}",
            flush=True,
        )
        return health

    print("DASHBOARD_MMAP starting local DeepStream mmap backend...", flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    _backend_process = subprocess.Popen(
        [sys.executable, "-m", "services.ml_service.core_v1.main"],
        cwd=str(ROOT),
        env=env,
    )
    atexit.register(_stop_local_backend)

    deadline = time.monotonic() + 12.0
    while time.monotonic() < deadline:
        if _backend_process.poll() is not None:
            print(
                f"DASHBOARD_MMAP backend exited code={_backend_process.returncode}",
                file=sys.stderr,
                flush=True,
            )
            return None
        health = _backend_health()
        if health is not None:
            print(
                f"DASHBOARD_MMAP backend ready mode={health.get('mode')} "
                f"online={health.get('online')}/{health.get('total')}",
                flush=True,
            )
            return health
        time.sleep(0.15)

    print("DASHBOARD_MMAP backend health timeout; UI will keep reconnecting", flush=True)
    return None


def run() -> int:
    # Patch the dashboard's old HTTP/JPEG readers before DashboardWindow is
    # instantiated. No changes to the large UI layout file are required.
    ui.FrameReader = DashboardMmapFrameReader
    ui.RealtimeState = ResilientRealtimeState

    ensure_mmap_backend()
    try:
        return ui.run()
    finally:
        _stop_local_backend()


if __name__ == "__main__":
    raise SystemExit(run())
