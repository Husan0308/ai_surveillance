from __future__ import annotations

import os

from .person_tracking_final import CameraPersonTrackingFinal


class CameraPersonHeatmap(CameraPersonTrackingFinal):
    """Camera + YOLO26m + NvDCF + lightweight movement-only heatmap.

    Heat is accumulated from the bottom-center of REAL current-frame NvDCF person
    tracks only after motion is confirmed. A seated/standing person therefore does
    not keep heating the same cell. The tiny native grid is rendered through
    NvDsDisplayMeta circles: no OpenCV frame copy/blend and no extra CUDA model.
    """

    def __init__(self) -> None:
        self.heatmap_updates = 0
        self.heatmap_points_now = 0
        super().__init__()

        # One normal walk should remain cool. Repeated traffic along the same path
        # gradually becomes yellow, then red. Keep the overlay transparent/light.
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

        osd_sink = self.osd.get_static_pad("sink")
        if osd_sink is None:
            raise RuntimeError("heatmap could not access nvdsosd sink pad")
        osd_sink.add_probe(self.Gst.PadProbeType.BUFFER, self._heatmap_render_probe)

        print(
            "CAMERA_HEATMAP enabled: mode=movement_only stationary_deposit=0 "
            "foot_point=bottom-center grid=32x18/camera motion_confirm=2-step "
            f"deposit={self.heat_deposit:.4f} decay={self.heat_decay:.5f} "
            f"yellow={self.heat_yellow:.3f} red={self.heat_red:.3f} "
            f"max_points={self.heat_max_points}/camera frame_copy=0 opencv_blend=0",
            flush=True,
        )

    def _tracker_probe(self, pad, info):
        buffer = info.get_buffer()
        if buffer is not None:
            # Update BEFORE display-only bbox enlargement/smoothing, so the floor
            # point comes from NvDCF's tight real tracker bbox.
            updated = self.bridge.heatmap_update(buffer)
            if updated > 0:
                self.heatmap_updates += updated
        return super()._tracker_probe(pad, info)

    def _heatmap_render_probe(self, _pad, info):
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

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        print(
            "CAMERA_HEATMAP "
            f"movement_updates={self.heatmap_updates} "
            f"points_now={self.heatmap_points_now} "
            f"points_total={self.bridge.heatmap_rendered_points_total()} "
            "mode=movement_only stationary_deposit=0 foot_point=bottom-center",
            flush=True,
        )
        return keep


def main() -> int:
    return CameraPersonHeatmap().run()


if __name__ == "__main__":
    raise SystemExit(main())
