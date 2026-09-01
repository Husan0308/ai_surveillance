from __future__ import annotations

# The Sentinel UI source is stored in line-preserving parts so the exact supplied
# interface can be carried without dropping pages/controls while CAM-01 is wired
# incrementally. The parts are concatenated and executed as this module.
from pathlib import Path as _Path

_parts_dir = _Path(__file__).with_name("ui_parts")
_source = "".join(path.read_text(encoding="utf-8") for path in sorted(_parts_dir.glob("part_*.pyfrag")))
exec(compile(_source, str(_parts_dir / "sentinel_ui_combined.py"), "exec"), globals(), globals())

# Keep the supplied page/widget construction untouched.  CAM-01 becomes real at
# the CameraView boundary itself, so every place that already creates a CAM-01
# view (Monitoring, fullscreen and expand dialog) consumes the same latest-only
# preview without changing any page layout or opening another RTSP session.
_camera_view_init = CameraView.__init__

def _camera_view_init_cam01_live(
    self,
    camera,
    people=None,
    show_room=True,
    occupancy=None,
    display_name=None,
    heatmap_enabled=False,
    realtime_camera_id=None,
    parent=None,
):
    if realtime_camera_id is None and getattr(camera, "id", None) == LIVE_PREVIEW_CAMERA:
        realtime_camera_id = LIVE_PREVIEW_CAMERA
    _camera_view_init(
        self,
        camera,
        people=people,
        show_room=show_room,
        occupancy=occupancy,
        display_name=display_name,
        heatmap_enabled=heatmap_enabled,
        realtime_camera_id=realtime_camera_id,
        parent=parent,
    )

CameraView.__init__ = _camera_view_init_cam01_live

del _source, _parts_dir, _Path
