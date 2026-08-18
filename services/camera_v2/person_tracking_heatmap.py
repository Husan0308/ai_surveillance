from __future__ import annotations

import os
import threading

from .heatmap_filter import NativeHeatmapFilter
from .person_tracking_final import CameraPersonTrackingFinal
from .pose_heatmap_bridge import PoseHeatmapBridge


class CameraPersonTrackingHeatmap(CameraPersonTrackingFinal):
    """Production camera-space heatmap driven only by tracked pose ankles.

    Bounding-box bottom-center is intentionally not accepted as a production heat
    anchor. A pose sidecar supplies left/right ankle keypoints matched to an active
    NvDCF track; if neither ankle is reliable, that sample contributes no heat.
    """

    def __init__(self) -> None:
        self.heatmap_enabled = os.environ.get("CAMERA_V2_HEATMAP", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        start_visible = os.environ.get("CAMERA_V2_HEATMAP_VISIBLE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.heatmap_render_enabled = self.heatmap_enabled and start_visible
        self.heatmap_updates = 0
        self.heatmap_render_frames = 0
        self.heatmap_visible_points = 0
        self.heatmap_filtered_points = 0
        self.heatmap_error = ""
        self.heatmap_filter: NativeHeatmapFilter | None = None
        self.pose_heatmap: PoseHeatmapBridge | None = None
        self._heatmap_sources: dict[int, bool] = {}
        self._heatmap_lock = threading.RLock()
        self.focus_source = -1
        super().__init__()

        self._heatmap_sources = {
            source_id: bool(self.heatmap_render_enabled)
            for source_id in range(len(self.cameras))
        }

        if not self.heatmap_enabled:
            print("CAMERA_HEATMAP disabled", flush=True)
            return

        cool_seconds = max(
            300.0,
            float(os.environ.get("CAMERA_V2_HEATMAP_COOL_SEC", "3600")),
        )
        hour_remaining = min(
            0.60,
            max(0.01, float(os.environ.get("CAMERA_V2_HEATMAP_REMAIN", "0.10"))),
        )
        decay = hour_remaining ** (
            1.0 / max(1.0, float(self.source_fps) * cool_seconds)
        )

        deposit = float(os.environ.get("CAMERA_V2_HEATMAP_DEPOSIT", "0.0028"))
        low = float(os.environ.get("CAMERA_V2_HEATMAP_LOW", "0.00028"))
        yellow = float(os.environ.get("CAMERA_V2_HEATMAP_YELLOW", "0.100"))
        red = float(os.environ.get("CAMERA_V2_HEATMAP_RED", "0.300"))
        max_points = max(
            12,
            min(96, int(os.environ.get("CAMERA_V2_HEATMAP_POINTS", "84"))),
        )

        self.pose_heatmap = PoseHeatmapBridge()
        self.pose_heatmap.configure(
            deposit=deposit,
            decay=decay,
            low_threshold=low,
            yellow_threshold=yellow,
            red_threshold=red,
            max_points_per_source=max_points,
        )
        self.pose_heatmap.reset()
        self.heatmap_filter = NativeHeatmapFilter()

        osd_sink = self.osd.get_static_pad("sink")
        if osd_sink is None:
            raise RuntimeError("CAMERA_HEATMAP could not get nvdsosd sink pad")
        osd_sink.add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._heatmap_render_probe,
        )

        print(
            "CAMERA_HEATMAP ready "
            f"anchor=pose-ankle-only bbox_anchor=disabled ankle_missing=skip "
            f"tracked_match=NvDCF grid=48x27 points={max_points}/cam "
            f"deposit={deposit:.5f} decay={decay:.8f} cool={cool_seconds:.0f}s "
            f"remain={hour_remaining:.2f} yellow={yellow:.3f} red={red:.3f} "
            "style=rolling-density fullscreen_focus_aware=1 per_camera_toggle=1",
            flush=True,
        )

    def set_heatmap_render_enabled(self, enabled: bool) -> None:
        value = bool(enabled) and self.heatmap_enabled
        for source_id in self._heatmap_sources:
            self._heatmap_sources[source_id] = value
        self.heatmap_render_enabled = value

    def set_heatmap_source_enabled(self, source_id: int, enabled: bool) -> None:
        source_id = int(source_id)
        if source_id not in self._heatmap_sources:
            return
        self._heatmap_sources[source_id] = bool(enabled) and self.heatmap_enabled
        self.heatmap_render_enabled = any(self._heatmap_sources.values())

    def heatmap_source_states(self) -> dict[int, bool]:
        return dict(self._heatmap_sources)

    def _enabled_mask(self) -> int:
        mask = 0
        for source_id, enabled in self._heatmap_sources.items():
            if enabled and 0 <= source_id < 32:
                mask |= 1 << source_id
        return mask

    def deposit_pose_ankle(
        self,
        *,
        source_id: int,
        object_id: int,
        captured_at: float,
        nx: float,
        ny: float,
        confidence: float,
    ) -> int:
        """Deposit one ankle observation; there is deliberately no bbox fallback."""
        if not self.heatmap_enabled or self.pose_heatmap is None:
            return 0
        tick = max(1, int(float(captured_at) * max(1.0, float(self.source_fps))))
        try:
            with self._heatmap_lock:
                updates = self.pose_heatmap.update_ankle(
                    source_id=int(source_id),
                    object_id=int(object_id),
                    tick=tick,
                    nx=float(nx),
                    ny=float(ny),
                    confidence=float(confidence),
                )
            if updates > 0:
                self.heatmap_updates += int(updates)
            self.heatmap_error = ""
            return max(0, int(updates))
        except Exception as exc:
            self.heatmap_error = f"ankle:{type(exc).__name__}:{exc}"
            return 0

    def _tracker_probe(self, pad, info):
        # Tracking remains authoritative for identity and bbox display, but heat is
        # NOT generated here. Pose ankle observations arrive asynchronously through
        # deposit_pose_ankle().
        return super()._tracker_probe(pad, info)

    def _current_focus_source(self) -> int:
        try:
            if self.tiler.find_property("show-source") is not None:
                value = int(self.tiler.get_property("show-source"))
                return value if 0 <= value < len(self.cameras) else -1
        except Exception:
            pass
        return -1

    def _heatmap_render_probe(self, _pad, info):
        buffer = info.get_buffer()
        enabled_mask = self._enabled_mask()
        if (
            not self.heatmap_enabled
            or enabled_mask == 0
            or buffer is None
            or self.pose_heatmap is None
        ):
            return self.Gst.PadProbeReturn.OK
        try:
            rows = int(getattr(self, "tiler_rows", 3))
            columns = int(getattr(self, "tiler_columns", 2))
            focus_source = self._current_focus_source()
            self.focus_source = focus_source
            if focus_source >= 0 and not self._heatmap_sources.get(focus_source, False):
                return self.Gst.PadProbeReturn.OK
            with self._heatmap_lock:
                rendered = self.pose_heatmap.render(
                    buffer,
                    wall_width=self.wall_width,
                    wall_height=self.wall_height,
                    rows=rows,
                    columns=columns,
                    source_count=len(self.cameras),
                    focus_source=focus_source,
                )
            filtered = 0
            if self.heatmap_filter is not None:
                filtered = self.heatmap_filter.apply(
                    buffer,
                    wall_width=self.wall_width,
                    wall_height=self.wall_height,
                    rows=1 if focus_source >= 0 else rows,
                    columns=1 if focus_source >= 0 else columns,
                    enabled_mask=(1 if focus_source >= 0 else enabled_mask),
                )
            if rendered >= 0:
                self.heatmap_render_frames += 1
                self.heatmap_filtered_points += max(0, int(filtered))
                self.heatmap_visible_points = max(0, int(rendered) - max(0, int(filtered)))
                self.heatmap_error = ""
        except Exception as exc:
            self.heatmap_error = f"render:{type(exc).__name__}:{exc}"
        return self.Gst.PadProbeReturn.OK

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        if self.heatmap_enabled:
            enabled = [sid for sid, state in self._heatmap_sources.items() if state]
            rendered_total = 0
            if self.pose_heatmap is not None:
                try:
                    rendered_total = self.pose_heatmap.rendered_points_total()
                except Exception:
                    rendered_total = 0
            print(
                "CAMERA_HEATMAP "
                f"anchor=ankle updates={self.heatmap_updates} "
                f"render_frames={self.heatmap_render_frames} "
                f"visible_points={self.heatmap_visible_points} "
                f"filtered_points={self.heatmap_filtered_points} "
                f"sources={enabled} focus={self.focus_source} "
                f"rendered_total={rendered_total} "
                f"error={self.heatmap_error or 'none'}",
                flush=True,
            )
        return keep


def main() -> int:
    return CameraPersonTrackingHeatmap().run()


if __name__ == "__main__":
    raise SystemExit(main())
