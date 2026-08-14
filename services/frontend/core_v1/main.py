from __future__ import annotations

import os
import sys

from . import dashboard


def _enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Stable baseline: run the real dashboard without presentation monkey-patches.
# Optional UI extensions are opt-in so a heatmap/polish regression can never
# make the live camera page unusable.
if _enabled("AI_SURVEILLANCE_UI_POLISH", False):
    from .ui_polish import install as install_polish

    install_polish(dashboard)

if _enabled("AI_SURVEILLANCE_UI_HEATMAP", False):
    from .heatmap_ui import install as install_heatmap

    install_heatmap(dashboard)


if __name__ == "__main__":
    sys.exit(dashboard.run())
