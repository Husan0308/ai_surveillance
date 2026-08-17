from __future__ import annotations

import os

from .person_tracking_final import CameraPersonTrackingFinal


class CameraPersonTrackingHeatmap(CameraPersonTrackingFinal):
    """Production local tracking runtime plus a smooth camera-space heatmap.

    Cross-camera ReID is intentionally absent. Heatmap is isolated from detector
    scheduling and uses only current NvDCF boxes:
      * accumulation happens on raw current NvDCF boxes before display smoothing;
      * the anchor is lifted slightly above bbox bottom-center;
      * every real tracked person contributes a faint presence pulse;
      * confirmed motion adds a denser continuous trail;
      * rendering happens after the tiler and before nvdsosd;
      * the native renderer blends overlapping heat cells into a continuous field.
    """

    def __init__(self) -> None:
        self.heatmap_enabled = os.environ.get("CAMERA_V2_HEATMAP", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.heatmap_updates = 0
        self.heatmap_render_frames = 0
        self.heatmap_visible_points = 0
        self.heatmap_error = ""
        super().__init__()

        if not self.heatmap_enabled:
            print("CAMERA_HEATMAP disabled", flush=True)
            return

        # Exponential cooling. About 10% of the accumulated value remains after
        # one hour by default; old paths therefore cool gradually instead of reset.
        cool_seconds = max(300.0, float(os.environ.get("CAMERA_V2_HEATMAP_COOL_SEC", "3600")))
        hour_remaining = min(
            0.60,
            max(0.01, float(os.environ.get("CAMERA_V2_HEATMAP_REMAIN", "0.10"))),
        )
        decay = hour_remaining ** (1.0 / max(1.0, float(self.source_fps) * cool_seconds))

        # Reference-like traffic density palette:
        #   one/few people -> blue/cyan
        #   repeated path  -> green/yellow
        #   heavy traffic  -> orange/red
        deposit = float(os.environ.get("CAMERA_V2_HEATMAP_DEPOSIT", "0.0030"))
        low = float(os.environ.get("CAMERA_V2_HEATMAP_LOW", "0.00050"))
        yellow = float(os.environ.get("CAMERA_V2_HEATMAP_YELLOW", "0.070"))
        red = float(os.environ.get("CAMERA_V2_HEATMAP_RED", "0.200"))
        max_points = max(12, min(96, int(os.environ.get("CAMERA_V2_HEATMAP_POINTS", "72"))))

        self.bridge.configure_heatmap(
            deposit=deposit,
            decay=decay,
            low_threshold=low,
            yellow_threshold=yellow,
            red_threshold=red,
            max_points_per_source=max_points,
        )
        self.bridge.reset_heatmap()

        # Accumulator uses source-camera coordinates before tiling. Rendering runs
        # after the 3x2 tiler and immediately before nvdsosd so the translucent heat
        # field is painted directly on top of each live camera tile.
        osd_sink = self.osd.get_static_pad("sink")
        if osd_sink is None:
            raise RuntimeError("CAMERA_HEATMAP could not get nvdsosd sink pad")
        osd_sink.add_probe(self.Gst.PadProbeType.BUFFER, self._heatmap_render_probe)

        print(
            "CAMERA_HEATMAP ready "
            f"anchor=feet-lifted-8pct all_tracks=1 presence_pulse=1 motion_trail=1 "
            f"grid=48x27 points={max_points}/cam "
            f"deposit={deposit:.5f} decay={decay:.8f} cool={cool_seconds:.0f}s "
            f"remain={hour_remaining:.2f} yellow={yellow:.3f} red={red:.3f} "
            "style=continuous-field palette=blue-cyan-yellow-red overlay=post-tiler/pre-osd",
            flush=True,
        )

    def _tracker_probe(self, pad, info):
        buffer = info.get_buffer()
        if self.heatmap_enabled and buffer is not None:
            try:
                # Run BEFORE parent style/smoothing. Synthetic display-hold boxes
                # therefore never add fake heat; only real NvDCF tracks paint.
                updates = self.bridge.heatmap_update(buffer)
                if updates > 0:
                    self.heatmap_updates += int(updates)
                self.heatmap_error = ""
            except Exception as exc:
                self.heatmap_error = f"update:{type(exc).__name__}:{exc}"
        return super()._tracker_probe(pad, info)

    def _heatmap_render_probe(self, _pad, info):
        buffer = info.get_buffer()
        if not self.heatmap_enabled or buffer is None:
            return self.Gst.PadProbeReturn.OK
        try:
            rendered = self.bridge.heatmap_render(
                buffer,
                wall_width=self.wall_width,
                wall_height=self.wall_height,
                rows=2,
                columns=3,
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
                f"rendered_total={self.bridge.heatmap_rendered_points_total()} "
                f"error={self.heatmap_error or 'none'}",
                flush=True,
            )
        return keep


def main() -> int:
    return CameraPersonTrackingHeatmap().run()


if __name__ == "__main__":
    raise SystemExit(main())
