"""Latest-only health adapter; observes producers without opening camera streams."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import time

from services.camera_v11.monitoring_telemetry_ipc_v1 import (
    DEFAULT_CAMERA_IDS, MonitoringTelemetryWriter, offline_snapshot,
)
from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameReader


class PreviewMonitoringBridge:
    def __init__(self, stats_path: Path, *, session_start_ns: int, session_id: str,
                 preview_paths: dict[str, str] | None = None):
        self.stats_path = stats_path
        self.session_start_ns = session_start_ns
        self.session_id = session_id
        self.readers = {c: PreviewFrameReader((preview_paths or {}).get(c,
            f"/dev/shm/v11_ui_preview_{c.lower().replace('-', '')}_v1.bin")) for c in DEFAULT_CAMERA_IDS}
        self.last: dict[str, tuple[int, int, int]] = {}

    def snapshot(self) -> dict:
        now = time.monotonic_ns()
        try:
            stats = json.loads(self.stats_path.read_text()) if time.time() - self.stats_path.stat().st_mtime < 2 else {}
        except (OSError, ValueError):
            stats = {}
        payload = offline_snapshot()
        payload["runtime"].update(status="live", session_id=self.session_id,
            producer_mode="preview-only", detector_enabled=False, shared_trt_workers=0,
            uptime_sec=(now - self.session_start_ns) / 1e9)
        for row in payload["cameras"]:
            camera = row["camera_id"]
            frame = self.readers[camera].read_latest(max_age_sec=1.0, metadata_only=True)
            current = stats.get(camera, {})
            if frame and frame.timestamp_ns >= self.session_start_ns:
                previous = self.last.get(camera)
                signature = (frame.sequence, frame.timestamp_ns)
                advancing = previous is not None and signature != previous[:2]
                changed_at = now if advancing or previous is None else previous[2]
                self.last[camera] = (*signature, changed_at)
                row.update(online=previous is not None and now - changed_at < 1e9,
                    preview_sequence=frame.sequence, preview_age_ms=(now - frame.timestamp_ns) / 1e6,
                    preview_fps=frame.fps, preview_exported=current.get("published_frames"),
                    source_fps=current.get("current_fps"), queue_depth=current.get("queue_buffers"),
                    queue_high_water=current.get("queue_high_water"), reconnects=current.get("reconnects"),
                    pipeline_errors=current.get("bus_errors"), warnings=current.get("bus_warnings"))
        online = sum(r["online"] for r in payload["cameras"])
        payload["runtime"]["online_camera_count"] = online
        payload["runtime"]["rtsp_source_count"] = (sum(
            current.get("runtime_graph", {}).get("factories", {}).get("rtspsrc", 0)
            for current in stats.values()) if stats else None)
        payload["runtime"]["errors_total"] = sum(r.get("pipeline_errors") or 0 for r in payload["cameras"])
        payload["runtime"]["warnings_total"] = sum(r.get("warnings") or 0 for r in payload["cameras"])
        return {"runtime": payload["runtime"], "cameras": payload["cameras"]}

    def close(self) -> None:
        for reader in self.readers.values():
            reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--session-start-ns", type=int, required=True)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    bridge = PreviewMonitoringBridge(args.stats, session_start_ns=args.session_start_ns, session_id=args.session_id)
    writer = MonitoringTelemetryWriter(args.output)
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while running:
            writer.publish(bridge.snapshot())
            time.sleep(0.25)
    finally:
        bridge.close()
        writer.publish({"runtime": dict(offline_snapshot()["runtime"], status="stopped"),
                        "cameras": offline_snapshot()["cameras"]})


if __name__ == "__main__":
    main()
