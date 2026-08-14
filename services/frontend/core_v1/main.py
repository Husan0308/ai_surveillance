from __future__ import annotations

import sys

from . import dashboard
from .ui_polish import install


# Install the production UI layer before any DashboardWindow/LivePage instances
# are created. This keeps the existing realtime backend wiring intact while
# replacing only the presentation/layout behavior.
install(dashboard)


if __name__ == "__main__":
    sys.exit(dashboard.run())
