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
        # Presentation extensions must never stop the live camera dashboard from
        # launching. Print diagnostics and continue with the plain dashboard.
        print(f"[frontend] optional {name} disabled after install error: {exc}", file=sys.stderr)
        traceback.print_exc()


if _enabled("AI_SURVEILLANCE_UI_POLISH", True):
    from .ui_polish import install as install_polish

    _install_optional("ui_polish", install_polish)

if _enabled("AI_SURVEILLANCE_UI_HEATMAP", True):
    from .heatmap_ui import install as install_heatmap

    _install_optional("heatmap_ui", install_heatmap)


if __name__ == "__main__":
    sys.exit(dashboard.run())
