from __future__ import annotations

import os

# Keep the mux at the known 4MP main-stream geometry. DeepStream recommends
# matching streammux to the input resolution to avoid an unnecessary scale before
# tracker/tiler. RF-DETR-S and NvDCF keep their own lightweight resolutions.
_CANONICAL_RUNTIME = {
    "CAMERA_V2_FRAME_WIDTH": "2560",
    "CAMERA_V2_FRAME_HEIGHT": "1440",
    "CAMERA_V2_DETECT_WIDTH": "672",
    "CAMERA_V2_DETECT_HEIGHT": "384",
    "CAMERA_V2_DETECT_CONF": "0.18",
    "CAMERA_V2_DETECT_IOU": "0.65",
    "CAMERA_V2_MAX_DET": "40",
    # GTX 1050 Ti hardware smoke proved batch 1 is faster than batch 2 for
    # RF-DETR-S. NvDCF carries continuity between sparse detector observations.
    "CAMERA_V2_MICRO_BATCH": "1",
    "CAMERA_V2_TRACKER_WIDTH": "512",
    "CAMERA_V2_TRACKER_HEIGHT": "288",
}
_stale_runtime_values: list[str] = []
for _key, _value in _CANONICAL_RUNTIME.items():
    _old = os.environ.get(_key)
    if _old is not None and _old != _value:
        _stale_runtime_values.append(f"{_key}={_old}")
    os.environ[_key] = _value

if _stale_runtime_values:
    print(
        "CAMERA_RUNTIME_PROFILE stale_env_overridden="
        + ",".join(_stale_runtime_values)
        + " canonical=mux:2560x1440,detector:RF-DETR-S@672x384,tracker:512x288",
        flush=True,
    )

# The active core uses RF-DETR-S as the mandatory sparse person detector. Install
# the spawn-safe worker before importing CameraPersonTrackingFinal, which imports
# the detector module and captures its worker function for process creation.
from .rfdetr_backend import install as _install_rfdetr_backend

_install_rfdetr_backend()

from .heatmap_filter import NativeHeatmapFilter
from .person_tracking_final import CameraPersonTrackingFinal


class CameraPersonTrackingHeatmap(CameraPersonTrackingFinal):
    """Camera-local RF-DETR-S + NvDCF tracking with floor-contact heatmap."""

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
        self._heatmap_sources: dict[int, bool] = {}
        self.focus_source = -1

        # Production display padding is intentionally separate from NvDCF truth.
        # Count + floor-contact heatmap sample the original current tracker bbox;
        # only the rectangle sent to OSD is enlarged afterwards so hands, elbows
        # and feet are not visually clipped by a tight person detector box.
        self.display_box_side_margin = max(
            0.0, min(0.18, float(os.environ.get("CAMERA_V2_DISPLAY_BOX_SIDE_MARGIN", "0.08")))
        )
        self.display_box_top_margin = max(
            0.0, min(0.12, float(os.environ.get("CAMERA_V2_DISPLAY_BOX_TOP_MARGIN", "0.04")))
        )
        self.display_box_bottom_margin = max(
            0.0, min(0.18, float(os.environ.get("CAMERA_V2_DISPLAY_BOX_BOTTOM_MARGIN", "0.10")))
        )
        self.display_boxes_expanded = 0
        self.display_box_error = ""

        super().__init__()

        self._heatmap_sources = {
            source_id: bool(self.heatmap_render_enabled)
            for source_id in range(len(self.cameras))
        }

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

        if self.heatmap_enabled:
            self.bridge.configure_heatmap(
                deposit=deposit,
                decay=decay,
                low_threshold=low,
                yellow_threshold=yellow,
                red_threshold=red,
                max_points_per_source=max_points,
            )
            self.bridge.reset_heatmap()
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
                "anchor=tracked-floor-point "
                f"grid=48x27 points={max_points}/cam "
                f"deposit={deposit:.5f} decay={decay:.8f} cool={cool_seconds:.0f}s "
                f"remain={hour_remaining:.2f} yellow={yellow:.3f} red={red:.3f} "
                "style=rolling-density per_camera_toggle=1 fullscreen_heatmap=hidden",
                flush=True,
            )
        else:
            print("CAMERA_HEATMAP disabled", flush=True)

        print(
            "CAMERA_DISPLAY_BOX ready "
            f"side={self.display_box_side_margin:.3f} "
            f"top={self.display_box_top_margin:.3f} "
            f"bottom={self.display_box_bottom_margin:.3f} "
            "mode=display-only adaptive-wide=1 tracker_truth=unchanged heatmap_truth=unchanged",
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

    def _tracker_probe(self, pad, info):
        result = super()._tracker_probe(pad, info)
        buffer = info.get_buffer()
        if buffer is None:
            return result

        # Heatmap samples the unpadded current tracker rectangle first. This keeps
        # the bottom-center floor point physically meaningful even though the OSD
        # rectangle is deliberately made more forgiving for limbs afterwards.
        if self.heatmap_enabled:
            try:
                updates = self.bridge.heatmap_update(buffer)
                if updates > 0:
                    self.heatmap_updates += int(updates)
                self.heatmap_error = ""
            except Exception as exc:
                self.heatmap_error = f"update:{type(exc).__name__}:{exc}"

        try:
            expanded = self.bridge.expand_display_boxes(
                buffer,
                side_margin=self.display_box_side_margin,
                top_margin=self.display_box_top_margin,
                bottom_margin=self.display_box_bottom_margin,
            )
            if expanded > 0:
                self.display_boxes_expanded += int(expanded)
                # Re-run styling so the text origin follows the padded display box.
                # Geometry is not read back into tracking/count/heatmap state.
                self.bridge.apply_local_track_style(buffer)
            self.display_box_error = ""
        except Exception as exc:
            self.display_box_error = f"{type(exc).__name__}:{exc}"

        return result

    def _current_focus_source(self) -> int:
        try:
            value = int(getattr(self, "focus_source", -1))
            if 0 <= value < len(self.cameras):
                return value
            if self.tiler.find_property("show-source") is not None:
                value = int(self.tiler.get_property("show-source"))
                return value if 0 <= value < len(self.cameras) else -1
        except Exception:
            pass
        return -1

    def _heatmap_render_probe(self, _pad, info):
        buffer = info.get_buffer()
        enabled_mask = self._enabled_mask()
        if not self.heatmap_enabled or enabled_mask == 0 or buffer is None:
            return self.Gst.PadProbeReturn.OK

        focus_source = self._current_focus_source()
        self.focus_source = focus_source
        if focus_source >= 0:
            return self.Gst.PadProbeReturn.OK

        try:
            rows = int(getattr(self, "tiler_rows", 3))
            columns = int(getattr(self, "tiler_columns", 2))
            rendered = self.bridge.heatmap_render(
                buffer,
                wall_width=int(self.wall_width),
                wall_height=int(self.wall_height),
                rows=rows,
                columns=columns,
                source_count=len(self.cameras),
            )
            filtered = 0
            if self.heatmap_filter is not None and rendered >= 0:
                filtered = self.heatmap_filter.apply(
                    buffer,
                    wall_width=int(self.wall_width),
                    wall_height=int(self.wall_height),
                    rows=rows,
                    columns=columns,
                    enabled_mask=enabled_mask,
                )
            if rendered >= 0:
                self.heatmap_render_frames += 1
                self.heatmap_filtered_points += max(0, int(filtered))
                self.heatmap_visible_points = max(
                    0,
                    int(rendered) - max(0, int(filtered)),
                )
                self.heatmap_error = ""
        except Exception as exc:
            self.heatmap_error = f"render:{type(exc).__name__}:{exc}"
        return self.Gst.PadProbeReturn.OK

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        if self.heatmap_enabled:
            enabled = [sid for sid, state in self._heatmap_sources.items() if state]
            rendered_total = 0
            try:
                rendered_total = self.bridge.heatmap_rendered_points_total()
            except Exception:
                pass
            print(
                "CAMERA_HEATMAP "
                f"updates={self.heatmap_updates} "
                f"render_frames={self.heatmap_render_frames} "
                f"visible_points={self.heatmap_visible_points} "
                f"filtered_points={self.heatmap_filtered_points} "
                f"sources={enabled} focus={self.focus_source} "
                f"rendered_total={rendered_total} "
                f"error={self.heatmap_error or 'none'}",
                flush=True,
            )
        print(
            "CAMERA_DISPLAY_BOX "
            f"expanded_total={self.display_boxes_expanded} "
            f"side={self.display_box_side_margin:.3f} "
            f"top={self.display_box_top_margin:.3f} "
            f"bottom={self.display_box_bottom_margin:.3f} "
            f"error={self.display_box_error or 'none'}",
            flush=True,
        )
        return keep


def main() -> int:
    return CameraPersonTrackingHeatmap().run()


if __name__ == "__main__":
    raise SystemExit(main())