from __future__ import annotations

from . import operator_dashboard as base


class DashboardPage(base.DashboardPage):
    """Reference dashboard uses three camera columns by two rows."""

    def __init__(self, toggle_callback):
        super().__init__(toggle_callback)
        # The first draft reused the newer 2x3 layout. Re-place the same camera
        # widgets into the exact old Operator Console 3x2 arrangement.
        for card in self.cards.values():
            self.grid.removeWidget(card)
        for index, camera_id in enumerate(base.CAMERAS):
            self.grid.addWidget(self.cards[camera_id], index // 3, index % 3)
        for row in range(3):
            self.grid.setRowStretch(row, 0)
        for col in range(3):
            self.grid.setColumnStretch(col, 1)
        self.grid.setRowStretch(0, 1)
        self.grid.setRowStretch(1, 1)
        self.grid_select.setCurrentText("3 × 2")


class RightRail(base.RightRail):
    def update_state(self, state, events):
        super().update_state(state, events)
        health = state.get("health") or {}
        publishers = health.get("publishers") or {}
        rates = [float(value.get("publish_rate") or 0.0) for value in publishers.values()]
        fps = sum(rates) / len(rates) if rates else 0.0
        self.fps[0].setText(f"{fps:.1f}")
        self.fps[1].setValue(max(0, min(100, int(round(fps / 30.0 * 100.0)))))


# OperatorWindow resolves these names from its defining module at runtime.
base.DashboardPage = DashboardPage
base.RightRail = RightRail


def run():
    return base.run()
