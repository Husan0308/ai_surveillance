from __future__ import annotations

"""Native-X11-safe MonitoringPage for the GstVideoOverlay wall."""

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QWidget

from .sentinel_ui_monitoring import MonitoringPage as _MonitoringPage


class MonitoringPage(_MonitoringPage):
    """Keep EGL pixels on a dedicated native child, never on the Qt wall itself.

    The camera wall is a normal Qt overlay/container that owns labels, borders and
    hover controls. GstVideoOverlay renders into one separate native QWidget below
    those overlays. This prevents Qt backing-store repaints from painting the dark
    panel background over an otherwise-live EGL image.
    """

    def __init__(self) -> None:
        super().__init__()

        self._video_surface = QWidget(self.wall)
        self._video_surface.setObjectName("nativeVideoSurface")
        self._video_surface.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self._video_surface.setAttribute(
            Qt.WidgetAttribute.WA_DontCreateNativeAncestors, False
        )
        self._video_surface.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self._video_surface.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._video_surface.setAutoFillBackground(False)
        self._video_surface.setGeometry(self.wall.rect())
        self._video_surface.lower()
        self._video_surface.installEventFilter(self)
        _ = int(self._video_surface.winId())

        # The historical wall itself is still a native QFrame. It is now only the
        # stable parent/overlay host; EGL is never bound to this paintable QFrame.
        self.wall.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors, False)
        self._ensure_native_ancestor_chain()
        self._raise_wall_overlays()

    def _raise_wall_overlays(self) -> None:
        """Keep every Qt HUD widget above the dedicated EGL surface."""
        surface = getattr(self, "_video_surface", None)
        if surface is not None:
            surface.lower()

        for name in (
            "camera_labels",
            "status_labels",
            "action_frames",
            "heatmap_buttons",
            "fullscreen_buttons",
            "fullscreen_camera_label",
            "fullscreen_fps_label",
        ):
            value = getattr(self.wall, name, None)
            if isinstance(value, (list, tuple)):
                for widget in value:
                    try:
                        widget.raise_()
                    except Exception:
                        pass
            elif value is not None:
                try:
                    value.raise_()
                except Exception:
                    pass

        for borders in getattr(self.wall, "tile_borders", []):
            for widget in borders:
                try:
                    widget.raise_()
                except Exception:
                    pass

    def _sync_video_surface_geometry(self) -> None:
        surface = getattr(self, "_video_surface", None)
        if surface is None:
            return
        rect = self.wall.rect()
        if surface.geometry() != rect:
            surface.setGeometry(rect)
        self._raise_wall_overlays()

    def _ensure_native_ancestor_chain(self) -> None:
        wall = getattr(self, "wall", None)
        if wall is None:
            return

        wall.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors, False)

        widget = wall
        seen: set[int] = set()
        while widget is not None:
            marker = id(widget)
            if marker in seen:
                break
            seen.add(marker)
            widget.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
            try:
                _ = int(widget.winId())
            except Exception:
                pass
            widget = widget.parentWidget()

    def _native_video_xid(self) -> int:
        surface = getattr(self, "_video_surface", None)
        if surface is None:
            return 0
        try:
            return int(surface.winId())
        except Exception:
            return 0

    def _ensure_video_binding(self, reason: str) -> None:
        self._ensure_native_ancestor_chain()
        self._sync_video_surface_geometry()
        xid = self._native_video_xid()
        if xid <= 0:
            print(f"SENTINEL_UI_BIND_SKIP reason={reason} error=no-video-surface-xid", flush=True)
            return
        self._bind_window_id(xid, reason=reason)

    def _start_or_bind(self, _xid: int) -> None:
        # Base LiveVideoWall emits its own QFrame XID. Ignore it intentionally and
        # bind GstVideoOverlay only to the dedicated non-painting child surface.
        self._ensure_video_binding("native-ready-dedicated-surface")

    def eventFilter(self, watched, event):
        event_type = event.type()
        surface = getattr(self, "_video_surface", None)

        if watched is self.wall:
            if event_type in (QEvent.Type.Resize, QEvent.Type.Show):
                self._sync_video_surface_geometry()
            elif event_type in (QEvent.Type.ParentChange, QEvent.Type.WinIdChange):
                self._ensure_native_ancestor_chain()
        elif surface is not None and watched is surface:
            if event_type == QEvent.Type.WinIdChange:
                self._schedule_binding_check("video-surface-winid-change")
            elif event_type == QEvent.Type.Show:
                self._schedule_binding_check("video-surface-show")

        return super().eventFilter(watched, event)

    def showEvent(self, event) -> None:
        self._ensure_native_ancestor_chain()
        self._sync_video_surface_geometry()
        super().showEvent(event)
