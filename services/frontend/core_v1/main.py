from __future__ import annotations

import os
import sys
import traceback

from . import dashboard


def _enabled(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _install_optional(name: str, installer) -> None:
    try:
        installer(dashboard)
    except Exception as exc:
        print(
            f"[frontend] optional {name} disabled after install error: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()


if _enabled("AI_SURVEILLANCE_UI_POLISH", True):
    from .ui_polish import install as install_polish

    _install_optional("ui_polish", install_polish)

# Heatmap/Pose controls are presentation-only. They do not stop analytics.
if _enabled("AI_SURVEILLANCE_UI_OVERLAY_CONTROLS", True):
    from .overlay_controls_ui import install as install_overlay_controls

    _install_optional("overlay_controls", install_overlay_controls)

# Legacy room-floor Heatmap page remains opt-in; live camera heat is rendered
# directly into the monitoring stream.
if _enabled("AI_SURVEILLANCE_UI_HEATMAP", False):
    from .heatmap_ui import install as install_heatmap

    _install_optional("legacy_floor_heatmap_ui", install_heatmap)


if __name__ == "__main__":
    sys.exit(dashboard.run())
