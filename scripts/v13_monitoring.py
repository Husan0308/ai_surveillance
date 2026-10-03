"""F5 opt-in telemetry adapter for two native and four preview-only owners.

Observes counters/SHM only; never reads RTSP or enters the preview critical path.
Frozen F3 freshness logic remains the authority for each preview's online state.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

from services.camera_v11.preview_monitoring_bridge import PreviewMonitoringBridge
from services.mv3dt_room.publication_readiness import read_readiness

PAIR = ("CAM-01", "CAM-04")
OTHER = ("CAM-02", "CAM-03", "CAM-05", "CAM-06")


def fresh_json(path: Path, max_age: float = 2) -> dict:
    try:
        if time.time() - path.stat().st_mtime > max_age:
            return {}
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


class MixedMonitoring:
    def __init__(self, output: Path, started: int):
        self.output = output
        self.started = started
        self.previous: dict[str, tuple[int, int]] = {}
        self.rates: dict[str, float] = {}
        self.published_previous: dict[str, tuple[int, int]] = {}
        self.published_rates: dict[str, float] = {}
        self.high = {c: 0 for c in PAIR}
        self.bridge = PreviewMonitoringBridge(output / "merged-stats.json",
            session_start_ns=started, session_id=output.name)

    def snapshot(self) -> dict:
        stats = fresh_json(self.output / "preview/camera-health.json")
        if any(c in stats for c in PAIR):
            raise ValueError("preview-only owner must never own the native pair")
        health = fresh_json(self.output / "room-pair/live-current/live-health.json")
        now = time.monotonic_ns()
        if health.get("mono_ns", 0) < self.started or now - health.get("mono_ns", 0) > 2e9:
            health = {}
        readiness = {s["source_id"]: s for s in (health.get("readiness") or {}).get("sources", [])}
        for decoder in health.get("decoder") or []:
            camera = decoder["camera_id"]
            if camera not in PAIR:
                raise ValueError("unexpected native source")
            count, stamp = decoder["decoder_output"], health["mono_ns"]
            previous = self.previous.get(camera)
            fps = ((count - previous[0]) * 1e9 / (stamp - previous[1])
                   if previous and stamp > previous[1] and count >= previous[0] else None)
            if previous is None or stamp > previous[1]:
                self.previous[camera] = count, stamp
                if fps is not None:
                    self.rates[camera] = fps
            self.high[camera] = max(self.high[camera], decoder["analytics_queue_level"])
            source = readiness.get(PAIR.index(camera), {})
            preview = health.get("preview", {}).get(camera, {})
            published = preview.get("sequence")
            old_published = self.published_previous.get(camera)
            if published is not None and (old_published is None or stamp > old_published[1]):
                if old_published and published >= old_published[0]:
                    self.published_rates[camera] = (published - old_published[0]) * 1e9 / (stamp - old_published[1])
                self.published_previous[camera] = published, stamp
            stats[camera] = {"current_fps": self.rates.get(camera), "published_frames": preview.get("sequence"),
                "queue_buffers": decoder["analytics_queue_level"], "queue_high_water": self.high[camera],
                "reconnects": source.get("reconnect_attempts"), "bus_errors": None,
                "bus_warnings": None, "runtime_graph": {"factories": {"rtspsrc": 1, "nvv4l2decoder": 1}}}
        merged = self.output / "merged-stats.json"
        temporary = merged.with_suffix(".tmp")
        temporary.write_text(json.dumps(stats))
        temporary.replace(merged)
        payload = self.bridge.snapshot()
        payload["runtime"].update(producer_mode="native-pair+four-preview-only", detector_enabled=True,
            native_analytics_sources=2,
            native_analytics_ready=read_readiness(self.output / "room-pair/live-current").get("ready", False),
            # No claim of a shared six-source TensorRT worker.
            shared_trt_workers=0)
        # Native log errors are checked by the acceptance auditor, not supplied
        # by these live counters. Never sum their unknown values into fake zero.
        payload["runtime"]["errors_total"] = None
        payload["runtime"]["warnings_total"] = None
        if set(stats) != set(PAIR + OTHER):
            payload["runtime"]["rtsp_source_count"] = None
        for row in payload["cameras"]:
            if row["camera_id"] in PAIR:
                # Instantaneous native header FPS can describe burst spacing,
                # not sustained publication. Use observed sequence progression.
                row["preview_fps"] = self.published_rates.get(row["camera_id"]) if health else None
            source = readiness.get(PAIR.index(row["camera_id"])) if row["camera_id"] in PAIR else None
            if source:
                row["analytics_pgie_frames"] = source["pgie_frames"]
                row["analytics_tracker_frames"] = source["tracker_frames"]
                row["analytics_result_age_ms"] = source["last_tracker_age_ms"]
        return payload

    def close(self):
        self.bridge.close()
