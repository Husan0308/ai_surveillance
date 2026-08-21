from __future__ import annotations

"""Compatibility selector for the exact old Apsidal UI detection stack.

The production launcher selects ``stable-yolo26m``.  On this branch that name
means the detector/tracker policy from ``ui-aspect-ratio-final@865bfedf`` rather
than the later tracker-free truth diagnostic.
"""


def install() -> None:
    from .old_ui_detection_backend import install as install_old_ui_detection

    install_old_ui_detection()


__all__ = ["install"]
