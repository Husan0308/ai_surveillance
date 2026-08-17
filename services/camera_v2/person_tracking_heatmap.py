from __future__ import annotations

import os

from .person_tracking_final import CameraPersonTrackingFinal


class CameraPersonTrackingHeatmap(CameraPersonTrackingFinal):
    """Production tracking runtime plus a lightweight camera-space foot heatmap.

    Heatmap is deliberately isolated from detector/tracker/ReID logic:
      * accumulation happens on raw current NvDCF boxes before display smoothing;
      * the anchor is near bbox bottom-center, lifted slightly above the feet line;
      * only confirmed motion deposits heat, so seated jitter does not paint;
      * rendering happens after the tiler and before nvdsosd;
      * all state is native metadata/grid state, never copied through NumPy/Python.
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

        # Exponential cooling: by default about 10% of a cell's heat remains after
        # one hour at the configured source FPS. A single pass stays blue/cyan and
        # fades; repeated traffic accumulates into yellow and eventually red.
        cool_seconds = max(300.0, float(os.environ.get("CAMERA_V2_HEATMAP_COOL_SEC", "3600")))
        hour_remaining = min(
            0.60,
            max(0.01, float(os.environ.get("CAMERA_V2_HEATMAP_REMAIN", "0.10"))),
        )
        decay = hour_remaining ** (1.0 / max(1.0, float(self.source_fps) * cool_seconds))

        deposit = float(os.environ.get("CAMERA_V2_HEATMAP_DEPOSIT", "0.0045"))
        low = float(os.environ.get("CAMERA_V2_HEATMAP_LOW", "0.00075"))
        # Keep normal one-person trails cool. Only repeatedly-used cells cross the
        # warmer thresholds, so density rather than one detection controls color.
        yellow = float(os.environ.get("CAMERA_V2_HEATMAP_YELLOW", "0.040"))
        red = float(os.environ.get("CAMERA_V2_HEATMAP_RED", "0.120"))
        max_points = max(8, min(48, int(os.environ.get("CAMERA_V2_HEATMAP_POINTS", "30"))))

        self.bridge.configure_heatmap(
            deposit=deposit,
            decay=decay,
            low_threshold=low,
            yellow_threshold=yellow,
            red_threshold=red,
            max_points_per_source=max_points,
        )
        self.bridge.reset_heatmap()

        # The native accumulator uses source-camera coordinates before tiling. The
        # renderer must run after nvmultistreamtiler has produced the 1920x720 wall,
        # immediately before nvdsosd consumes NvDsDisplayMeta circles.
        osd_sink = self.osd.get_static_pad("sink")
        if osd_sink is None:
            raise RuntimeError("CAMERA_HEATMAP could not get nvdsosd sink pad")
        osd_sink.add_probe(self.Gst.PadProbeType.BUFFER, self._heatmap_render_probe)

        print(
            "CAMERA_HEATMAP ready "
            f"anchor=feet-lifted-8pct motion_only=1 grid=48x27 points={max_points}/cam "
            f"deposit={deposit:.5f} decay={decay:.8f} cool={cool_seconds:.0f}s "
            f"remain={hour_remaining:.2f} yellow={yellow:.3f} red={red:.3f} "
            "palette=blue-cyan-yellow-red overlay=post-tiler/pre-osd",
            flush=True,
        )

    def _tracker_probe(self, pad, info):
        buffer = info.get_buffer()
        if self.heatmap_enabled and buffer is not None:
            try:
                # Run BEFORE parent style/smoothing. This guarantees synthetic
                # display-hold boxes never add fake heat and only real NvDCF tracks
                # can leave a trail.
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
