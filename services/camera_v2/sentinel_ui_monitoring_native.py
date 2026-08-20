from __future__ import annotations

"""Native-X11-safe MonitoringPage for the GstVideoOverlay wall."""

from PySide6.QtCore import Qt

from .sentinel_ui_monitoring import MonitoringPage as _MonitoringPage


class MonitoringPage(_MonitoringPage):
    """Keep the EGL wall inside a complete native Qt ancestor chain.

    The wall itself is a native X11 drawable. A native child whose intermediate
    ancestors remain alien widgets can escape Qt's normal sibling clipping/stacking
    under X11/XWayland. Promote the already-constructed hierarchy before the first
    GstVideoOverlay bind and allow the wall to participate in that hierarchy.
    """

    def _ensure_native_ancestor_chain(self) -> None:
        wall = getattr(self, "wall", None)
        if wall is None:
            return

        # The historical wall explicitly prevented native ancestors. That was useful
        # for an earlier layout, but the current sidebar/header/rail shell needs a
        # real native parent chain so EGL cannot paint outside the camera panel.
        wall.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors, False)

        widget = wall.parentWidget()
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

    def showEvent(self, event) -> None:
        self._ensure_native_ancestor_chain()
        super().showEvent(event)
