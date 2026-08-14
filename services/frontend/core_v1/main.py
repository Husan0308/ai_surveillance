from __future__ import annotations

import sys

from . import dashboard
from .ui_polish import install as install_polish
from .heatmap_ui import install as install_heatmap


# Presentation layers are installed before DashboardWindow/LivePage instances
# are created. The heatmap extension adds a separate low-rate floor page and
# never changes the realtime camera rendering hot path.
install_polish(dashboard)
install_heatmap(dashboard)


if __name__ == "__main__":
    sys.exit(dashboard.run())
