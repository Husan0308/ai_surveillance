from __future__ import annotations

import os
import threading
import time
from collections import deque

from .person_heatmap import CameraPersonHeatmap
from .ui_bridge import NativeUIBridge


class SentinelLiveRuntime(CameraPersonHeatmap):
    """Realtime snapshot adapter around the proven Camera V2 pipeline.

    The camera/detector/tracker hot path is inherited unchanged:
      RTSP/NVDEC -> nvstreammux -> YOLO26m -> NvDCF -> tiler -> OSD/EGL

    This class only reads NvDsObjectMeta after NvDCF and exposes lightweight
    dictionaries for the Qt UI. No frame copies, JPEG, appsink or second decoder.
    """

    def __init__(self) -> None:
        self.ui_lock = threading.RLock()
        self.live_tracks: dict[tuple[int, int], dict] = {}
        self.live_events: deque[dict] = deque(maxlen=1200)
        self._last_ui_snapshot = 0.0
        self._snapshot_period = 0.10
        self._camera_last_frames: dict[str, int] = {}
        self._camera_last_change: dict[str, float] = {}
        self.ui_bridge = NativeUIBridge()

        os.environ.setdefault("CAMERA_V2_WALL_WIDTH", "1024")
        os.environ.setdefault("CAMERA_V2_WALL_HEIGHT", "864")
        super().__init__()

        self.wall_width = 1024
        self.wall_height = 864
        self._set_if(self.tiler, "rows", 3)
        self._set_if(self.tiler, "columns", 2)
        self._set_if(self.tiler, "width", self.wall_width)
        self._set_if(self.tiler, "height", self.wall_height)
        if self.tiler.find_property("show-source") is not None:
            self.tiler.set_property("show-source", -1)
        # The Qt grid owns the final visible geometry. Stretching only happens at
        # the presentation sink; detector/tracker coordinates remain source-space.
        self._set_if(self.sink, "force-aspect-ratio", False)

        now = time.monotonic()
        for camera in self.cameras:
            self._camera_last_frames[camera.camera_id] = int(self.stats[camera.camera_id].frames)
            self._camera_last_change[camera.camera_id] = now

    def _camera_name(self, source_id: int) -> str:
        if 0 <= source_id < len(self.cameras):
            return self.cameras[source_id].camera_id
        return f"CAM-{source_id + 1:02d}"

    def _update_ui_tracks(self, rows: list[dict], now: float) -> None:
        epoch = time.time()
        with self.ui_lock:
            for row in rows:
                source_id = int(row.get("source_id", -1))
                object_id = int(row.get("object_id", -1))
                if not 0 <= source_id < len(self.cameras) or object_id < 0:
                    continue

                key = (source_id, object_id)
                previous = self.live_tracks.get(key)
                item = dict(row)
                item["camera_id"] = self._camera_name(source_id)
                item["label"] = previous.get("label") if previous else f"Unknown_{source_id + 1:02d}_{object_id}"
                item["first_seen"] = previous["first_seen"] if previous else now
                item["last_seen"] = now
                item["first_seen_epoch"] = previous.get("first_seen_epoch", epoch) if previous else epoch
                item["last_seen_epoch"] = epoch
                item["room_id"] = source_id // 2 + 1
                item["cameras"] = [item["camera_id"]]
                self.live_tracks[key] = item

                if previous is None:
                    self.live_events.appendleft({
                        "type": "unknown",
                        "time": epoch,
                        "camera_id": item["camera_id"],
                        "source_id": source_id,
                        "room_id": item["room_id"],
                        "object_id": object_id,
                        "person_id": f"{source_id}:{object_id}",
                        "label": item["label"],
                        "message": f"{item['label']} {item['camera_id']} da aniqlandi",
                    })

            # nvstreammux may emit partial live batches; do not turn one missing
            # source in one batch into a false exit event.
            expired = [
                key for key, item in self.live_tracks.items()
                if now - float(item.get("last_seen", 0.0)) > 1.35
            ]
            for key in expired:
                item = self.live_tracks.pop(key)
                self.live_events.appendleft({
                    "type": "exit",
                    "time": epoch,
                    "camera_id": item["camera_id"],
                    "source_id": item["source_id"],
                    "room_id": item["room_id"],
                    "object_id": item["object_id"],
                    "person_id": f"{item['source_id']}:{item['object_id']}",
                    "label": item["label"],
                    "message": f"{item['label']} {item['camera_id']} dan chiqdi",
                })

    def _tracker_probe(self, pad, info):
        result = super()._tracker_probe(pad, info)
        buffer = info.get_buffer()
        if buffer is None:
            return result

        now = time.monotonic()
        if now - self._last_ui_snapshot >= self._snapshot_period:
            rows = self.ui_bridge.snapshot_tracks(buffer)
            self._update_ui_tracks(rows, now)
            self._last_ui_snapshot = now
        return result

    def _heatmap_render_probe(self, _pad, _info):
        # The supplied Sentinel reference UI intentionally has no Heatmap page.
        # Native movement accumulation still runs in the inherited tracker probe.
        self.heatmap_points_now = 0
        return self.Gst.PadProbeReturn.OK

    def _expire_tracks(self, now: float) -> None:
        epoch = time.time()
        with self.ui_lock:
            expired = [
                key for key, item in self.live_tracks.items()
                if now - float(item.get("last_seen", 0.0)) > 1.35
            ]
            for key in expired:
                item = self.live_tracks.pop(key)
                self.live_events.appendleft({
                    "type": "exit",
                    "time": epoch,
                    "camera_id": item["camera_id"],
                    "source_id": item["source_id"],
                    "room_id": item["room_id"],
                    "object_id": item["object_id"],
                    "person_id": f"{item['source_id']}:{item['object_id']}",
                    "label": item["label"],
                    "message": f"{item['label']} {item['camera_id']} dan chiqdi",
                })

    def ui_snapshot(self) -> dict:
        now = time.monotonic()
        self._expire_tracks(now)

        with self.ui_lock:
            tracks = [dict(item) for item in self.live_tracks.values()]
            events = [dict(item) for item in self.live_events]

        cameras = []
        for source_id, camera in enumerate(self.cameras):
            runtime = self.stats[camera.camera_id]
            frame_count = int(runtime.frames)
            previous_count = self._camera_last_frames.get(camera.camera_id, frame_count)
            if frame_count != previous_count:
                self._camera_last_frames[camera.camera_id] = frame_count
                self._camera_last_change[camera.camera_id] = now

            last_change = self._camera_last_change.get(camera.camera_id, 0.0)
            online = frame_count > 0 and (now - last_change) < 2.0
            p50 = self._percentile(runtime.intervals_ms, 0.50)
            fps = 0.0 if not online or p50 is None or p50 <= 0 else 1000.0 / p50
            count = sum(1 for row in tracks if int(row["source_id"]) == source_id)
            cameras.append({
                "source_id": source_id,
                "camera_id": camera.camera_id,
                "room_id": source_id // 2 + 1,
                "fps": fps,
                "online": online,
                "count": count,
                "frame_age_ms": max(0.0, (now - last_change) * 1000.0) if frame_count else 0.0,
            })

        rooms = []
        for room_id, pair in enumerate(((0, 1), (2, 3), (4, 5)), start=1):
            per_camera = [sum(1 for row in tracks if int(row["source_id"]) == sid) for sid in pair]
            rooms.append({
                "room_id": room_id,
                "name": ("Lobbi", "Ofis", "Ombor")[room_id - 1],
                # Camera V2 currently has no cross-camera ReID. max() prevents
                # obvious double-counting for the two views of the same room.
                "count": max(per_camera) if per_camera else 0,
                "camera_counts": per_camera,
                "camera_ids": [self._camera_name(sid) for sid in pair],
            })

        with self.det_lock:
            detector = {
                "ready": bool(self.det_ready),
                "error": str(self.det_error or ""),
                "batch_ms": float(self.det_batch_ms),
                "result_age_ms": float(getattr(self, "detector_result_age_ms", 0.0)),
                "tracked_now": int(self.tracked_now),
            }

        return {
            "timestamp": time.time(),
            "cameras": cameras,
            "tracks": tracks,
            "events": events,
            "rooms": rooms,
            "detector": detector,
        }
