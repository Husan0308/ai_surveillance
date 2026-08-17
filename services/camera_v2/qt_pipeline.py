from __future__ import annotations

import time

from .person_heatmap import CameraPersonHeatmap


class CameraPersonHeatmapQt(CameraPersonHeatmap):
    """Thin UI adapter; does not alter the DeepStream graph.

    CameraPersonHeatmap remains the complete camera/detection/tracking/heatmap
    implementation. This subclass only records realtime frame freshness so the Qt
    cards do not confuse an old PTS with a currently-online RTSP source.
    """

    def __init__(self) -> None:
        self._ui_last_frame_mono: dict[str, float] = {}
        super().__init__()
        self._ui_last_frame_mono = {camera.camera_id: 0.0 for camera in self.cameras}

    def _source_probe(self, pad, info, cid: str):
        result = super()._source_probe(pad, info, cid)
        if info.get_buffer() is not None:
            self._ui_last_frame_mono[cid] = time.monotonic()
        return result

    def ui_snapshot(self) -> dict:
        snapshot = super().ui_snapshot()
        now = time.monotonic()
        for camera in snapshot.get("cameras", []):
            cid = str(camera.get("camera_id", ""))
            last = float(self._ui_last_frame_mono.get(cid, 0.0))
            camera["online"] = bool(last > 0.0 and now - last <= 2.0)
            camera["frame_age_ms"] = None if last <= 0.0 else max(0.0, (now - last) * 1000.0)
        return snapshot
