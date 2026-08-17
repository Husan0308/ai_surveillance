from __future__ import annotations

import os

from .person_tracking_final import CameraPersonTrackingFinal


class CameraPersonTrackingHeatmap(CameraPersonTrackingFinal):
    """Production local tracking runtime plus a camera-space occupancy heatmap.

    Cross-camera ReID is intentionally absent. Heatmap accumulation keeps running
    even when the UI hides the overlay, so reopening Heatmap shows the real recent
    history instead of starting from zero.
    """

    def __init__(self) -> None:
        self.heatmap_enabled = os.environ.get("CAMERA_V2_HEATMAP", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.heatmap_render_enabled = self.heatmap_enabled
        self.heatmap_updates = 0
        self.heatmap_render_frames = 0
        self.heatmap_visible_points = 0
        self.heatmap_error = ""
        super().__init__()

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

        deposit = float(os.environ.get("CAMERA_V2_HEATMAP_DEPOSIT", "0.0022"))
        low = float(os.environ.get("CAMERA_V2_HEATMAP_LOW", "0.00028"))
        yellow = float(os.environ.get("CAMERA_V2_HEATMAP_YELLOW", "0.100"))
        red = float(os.environ.get("CAMERA_V2_HEATMAP_RED", "0.300"))
        max_points = max(
            12,
            min(96, int(os.environ.get("CAMERA_V2_HEATMAP_POINTS", "84"))),
        )

        self.bridge.configure_heatmap(
            deposit=deposit,
            decay=decay,
            low_threshold=low,
            yellow_threshold=yellow,
            red_threshold=red,
            max_points_per_source=max_points,
        )
        self.bridge.reset_heatmap()

        osd_sink = self.osd.get_static_pad("sink")
        if osd_sink is None:
            raise RuntimeError("CAMERA_HEATMAP could not get nvdsosd sink pad")
        osd_sink.add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._heatmap_render_probe,
        )

        print(
            "CAMERA_HEATMAP ready "
            f"anchor=bottom-center-lift3pct probation=4 dwell_weighted=1 motion_trail=1 "
            f"perspective_splat=1 grid=48x27 points={max_points}/cam "
            f"deposit={deposit:.5f} decay={decay:.8f} cool={cool_seconds:.0f}s "
            f"remain={hour_remaining:.2f} yellow={yellow:.3f} red={red:.3f} "
            "style=rolling-density palette=blue-cyan-green-yellow-red "
            "overlay=post-tiler/pre-osd cross_camera=0 reid=0 ui_toggle=1",
            flush=True,
        )

    def set_heatmap_render_enabled(self, enabled: bool) -> None:
        self.heatmap_render_enabled = bool(enabled) and self.heatmap_enabled

    def _tracker_probe(self, pad, info):
        buffer = info.get_buffer()
        if self.heatmap_enabled and buffer is not None:
            try:
                updates = self.bridge.heatmap_update(buffer)
                if updates > 0:
                    self.heatmap_updates += int(updates)
                self.heatmap_error = ""
            except Exception as exc:
                self.heatmap_error = f"update:{type(exc).__name__}:{exc}"
        return super()._tracker_probe(pad, info)

    def _heatmap_render_probe(self, _pad, info):
        buffer = info.get_buffer()
        if (
            not self.heatmap_enabled
            or not self.heatmap_render_enabled
            or buffer is None
        ):
            return self.Gst.PadProbeReturn.OK
        try:
            rendered = self.bridge.heatmap_render(
                buffer,
                wall_width=self.wall_width,
                wall_height=self.wall_height,
                rows=int(getattr(self, "tiler_rows", 2)),
                columns=int(getattr(self, "tiler_columns", 3)),
                source_count=len(self.cameras),
            )
            if rendered >= 0:
                self.heatmap_render_frames += 1
                self.heatmap_visible_points = int(rendered)
                self.heatmap_error = ""
        except Exception as exc:
            self.heatmap_error = f"render:{type(exc).__name__}:{exc}"
        return self.Gst.PadProbeReturn.OK

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        if self.heatmap_enabled:
            print(
                "CAMERA_HEATMAP "
                f"updates={self.heatmap_updates} "
                f"render_frames={self.heatmap_render_frames} "
                f"visible_points={self.heatmap_visible_points} "
                f"visible={int(self.heatmap_render_enabled)} "
                f"rendered_total={self.bridge.heatmap_rendered_points_total()} "
                f"error={self.heatmap_error or 'none'}",
                flush=True,
            )
        return keep


def main() -> int:
    return CameraPersonTrackingHeatmap().run()


if __name__ == "__main__":
    raise SystemExit(main())
