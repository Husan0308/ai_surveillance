from __future__ import annotations

import os
import threading
import time
from collections import deque

from .person_tracking_final import CameraPersonTrackingFinal
from .ui_bridge import NativeUIBridge


class CameraPersonHeatmap(CameraPersonTrackingFinal):
    """Camera + YOLO26m + NvDCF + movement heatmap + realtime UI snapshots."""

    def __init__(self) -> None:
        self.heatmap_updates = 0
        self.heatmap_points_now = 0
        self.qt_mode = os.environ.get("CAMERA_V2_QT_UI", "0").strip().lower() in {"1", "true", "yes", "on"}
        self.ui_lock = threading.RLock()
        self.live_tracks: dict[tuple[int, int], dict] = {}
        self.live_events: deque[dict] = deque(maxlen=600)
        self._last_ui_snapshot = 0.0
        self._snapshot_period = 0.10
        super().__init__()

        self.heat_deposit = float(os.environ.get("CAMERA_V2_HEATMAP_DEPOSIT", "0.0025"))
        self.heat_decay = float(os.environ.get("CAMERA_V2_HEATMAP_DECAY", "0.99992"))
        self.heat_low = float(os.environ.get("CAMERA_V2_HEATMAP_LOW", "0.003"))
        self.heat_yellow = float(os.environ.get("CAMERA_V2_HEATMAP_YELLOW", "0.070"))
        self.heat_red = float(os.environ.get("CAMERA_V2_HEATMAP_RED", "0.180"))
        self.heat_max_points = int(os.environ.get("CAMERA_V2_HEATMAP_MAX_POINTS", "18"))

        self.bridge.configure_heatmap(
            deposit=self.heat_deposit,
            decay=self.heat_decay,
            low_threshold=self.heat_low,
            yellow_threshold=self.heat_yellow,
            red_threshold=self.heat_red,
            max_points_per_source=self.heat_max_points,
        )
        self.bridge.reset_heatmap()
        self.ui_bridge = NativeUIBridge()

        # Sentinel monitoring page is 2 cameras per row, 3 rows. 1280x1080 keeps
        # every tile at 640x360 (16:9) and the same total pixels as 1920x720.
        if self.qt_mode:
            self.wall_width = int(os.environ.get("CAMERA_V2_QT_WALL_WIDTH", "1280"))
            self.wall_height = int(os.environ.get("CAMERA_V2_QT_WALL_HEIGHT", "1080"))
            self._set_if(self.tiler, "rows", 3)
            self._set_if(self.tiler, "columns", 2)
            self._set_if(self.tiler, "width", self.wall_width)
            self._set_if(self.tiler, "height", self.wall_height)

        osd_sink = self.osd.get_static_pad("sink")
        if osd_sink is None:
            raise RuntimeError("heatmap could not access nvdsosd sink pad")
        osd_sink.add_probe(self.Gst.PadProbeType.BUFFER, self._heatmap_render_probe)

        print(
            "CAMERA_HEATMAP enabled: mode=movement_only stationary_deposit=0 "
            "foot_point=bottom-center grid=32x18/camera motion_confirm=2-step "
            f"deposit={self.heat_deposit:.4f} decay={self.heat_decay:.5f} "
            f"yellow={self.heat_yellow:.3f} red={self.heat_red:.3f} "
            f"max_points={self.heat_max_points}/camera qt_mode={int(self.qt_mode)}",
            flush=True,
        )

    def _camera_name(self, source_id: int) -> str:
        if 0 <= source_id < len(self.cameras):
            return self.cameras[source_id].camera_id
        return f"CAM-{source_id + 1:02d}"

    def _update_ui_tracks(self, rows: list[dict], now: float) -> None:
        with self.ui_lock:
            for row in rows:
                source_id = int(row["source_id"])
                object_id = int(row["object_id"])
                key = (source_id, object_id)
                previous = self.live_tracks.get(key)
                item = dict(row)
                item["camera_id"] = self._camera_name(source_id)
                item["label"] = f"Unknown_{source_id + 1:02d}_{object_id}"
                item["first_seen"] = previous["first_seen"] if previous else now
                item["last_seen"] = now
                self.live_tracks[key] = item
                if previous is None:
                    self.live_events.appendleft({
                        "type": "entry",
                        "time": time.time(),
                        "camera_id": item["camera_id"],
                        "source_id": source_id,
                        "object_id": object_id,
                        "label": item["label"],
                        "message": f"{item['label']} {item['camera_id']} da paydo bo'ldi",
                    })

            expired = [key for key, item in self.live_tracks.items() if now - float(item["last_seen"]) > 0.85]
            for key in expired:
                item = self.live_tracks.pop(key)
                self.live_events.appendleft({
                    "type": "exit",
                    "time": time.time(),
                    "camera_id": item["camera_id"],
                    "source_id": item["source_id"],
                    "object_id": item["object_id"],
                    "label": item["label"],
                    "message": f"{item['label']} {item['camera_id']} dan chiqdi",
                })

    def _tracker_probe(self, pad, info):
        buffer = info.get_buffer()
        if buffer is not None:
            updated = self.bridge.heatmap_update(buffer)
            if updated > 0:
                self.heatmap_updates += updated

            now = time.monotonic()
            if now - self._last_ui_snapshot >= self._snapshot_period:
                rows = self.ui_bridge.snapshot_tracks(buffer)
                self._update_ui_tracks(rows, now)
                self._last_ui_snapshot = now
        return super()._tracker_probe(pad, info)

    def _heatmap_render_probe(self, _pad, info):
        # Qt mode uses transparent per-camera overlays. This lets every camera own
        # an independent Heatmap button while the native video path stays untouched.
        if self.qt_mode:
            self.heatmap_points_now = 0
            return self.Gst.PadProbeReturn.OK

        buffer = info.get_buffer()
        if buffer is not None:
            rendered = self.bridge.heatmap_render(
                buffer,
                wall_width=self.wall_width,
                wall_height=self.wall_height,
                rows=2,
                columns=3,
                source_count=len(self.cameras),
            )
            if rendered >= 0:
                self.heatmap_points_now = rendered
        return self.Gst.PadProbeReturn.OK

    def ui_snapshot(self) -> dict:
        now = time.monotonic()
        with self.ui_lock:
            expired = [key for key, item in self.live_tracks.items() if now - float(item["last_seen"]) > 0.85]
            for key in expired:
                item = self.live_tracks.pop(key)
                self.live_events.appendleft({
                    "type": "exit", "time": time.time(), "camera_id": item["camera_id"],
                    "source_id": item["source_id"], "object_id": item["object_id"],
                    "label": item["label"], "message": f"{item['label']} {item['camera_id']} dan chiqdi",
                })
            tracks = [dict(item) for item in self.live_tracks.values()]
            events = [dict(item) for item in self.live_events]

        cameras = []
        for source_id, camera in enumerate(self.cameras):
            runtime = self.stats[camera.camera_id]
            p50 = self._percentile(runtime.intervals_ms, 0.50)
            fps = 0.0 if p50 is None or p50 <= 0 else 1000.0 / p50
            count = sum(1 for row in tracks if row["source_id"] == source_id)
            cameras.append({
                "source_id": source_id,
                "camera_id": camera.camera_id,
                "fps": fps,
                "online": runtime.last_pts_ns is not None,
                "count": count,
            })

        room_counts = []
        for room_index, pair in enumerate(((0, 1), (2, 3), (4, 5)), start=1):
            per_camera = [sum(1 for row in tracks if row["source_id"] == sid) for sid in pair]
            room_counts.append({
                "room_id": room_index,
                "name": f"Room {room_index}",
                "count": max(per_camera) if per_camera else 0,
                "camera_counts": per_camera,
            })

        return {"cameras": cameras, "tracks": tracks, "events": events, "rooms": room_counts}

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        with self.ui_lock:
            active = len(self.live_tracks)
        print(
            "CAMERA_HEATMAP "
            f"movement_updates={self.heatmap_updates} points_now={self.heatmap_points_now} "
            f"points_total={self.bridge.heatmap_rendered_points_total()} active_tracks={active} "
            f"qt_mode={int(self.qt_mode)} mode=movement_only",
            flush=True,
        )
        return keep


def main() -> int:
    return CameraPersonHeatmap().run()


if __name__ == "__main__":
    raise SystemExit(main())
