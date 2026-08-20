from __future__ import annotations

import multiprocessing as mp
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .data import EVENTS, PEOPLE
from .sentinel_ui_base import APP_QSS, C, label, make_button
from .sentinel_ui_enrollment import EnrollmentPage
from .sentinel_ui_monitoring_native import MonitoringPage
from .sentinel_ui_pages import EventsPage, PeoplePage, RoomsPage
from .sentinel_ui_settings import SettingsPage


BUILD_TAG = "2026.08.20-r12-rfdetr"


class MainWindow(QMainWindow):
    NAV = [
        ("▣", "Monitoring", "Live cameras · RF-DETR-S · NvDCF", MonitoringPage),
        ("♙", "People", f"{len(PEOPLE)} ta identity", PeoplePage),
        ("⌁", "Events", f"{len(EVENTS)} ta hodisa", EventsPage),
        ("▥", "Rooms", "Xonalar va camera topology", RoomsPage),
        ("♙+", "Enrollment", "10 ta yuz rasmi orqali ro'yxatga olish", EnrollmentPage),
        ("⚙", "Settings", "Camera management", SettingsPage),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentinel VMS")
        self.resize(1500, 920)
        self.setMinimumSize(1180, 720)
        self._monitoring_fullscreen = False

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(198)
        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(0)

        brand = QFrame()
        brand.setFixedHeight(64)
        brand.setStyleSheet(f"border-bottom:1px solid {C['border']};")
        bl = QHBoxLayout(brand)
        bl.setContentsMargins(15, 0, 12, 0)
        shield = label("◇", color=C["primary"])
        shield.setStyleSheet(f"color:{C['primary']};font-size:22px;")
        bl.addWidget(shield)
        bt = QVBoxLayout()
        bt.setSpacing(0)
        bt.addWidget(label("SENTINEL VMS", "brand"))
        bt.addWidget(label("video monitoring", "mono"))
        bl.addLayout(bt)
        bl.addStretch()
        side.addWidget(brand)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons = []
        navwrap = QWidget()
        nl = QVBoxLayout(navwrap)
        nl.setContentsMargins(8, 9, 8, 8)
        nl.setSpacing(3)
        for i, (icon, title, _, _) in enumerate(self.NAV):
            button = make_button(f"{icon}   {title}")
            button.setObjectName("nav")
            button.setCheckable(True)
            button.setFixedHeight(38)
            button.clicked.connect(lambda _, i=i: self.switch_page(i))
            self.nav_group.addButton(button)
            self.nav_buttons.append(button)
            nl.addWidget(button)
        nl.addStretch()
        side.addWidget(navwrap, 1)

        build = label(f"build {BUILD_TAG}", "mono")
        build.setStyleSheet(
            f"border-top:1px solid {C['border']};padding:12px 14px;color:{C['muted']};"
        )
        side.addWidget(build)
        main.addWidget(self.sidebar)

        self.content = QWidget()
        content_l = QVBoxLayout(self.content)
        content_l.setContentsMargins(0, 0, 0, 0)
        content_l.setSpacing(0)

        self.header = QFrame()
        self.header.setObjectName("header")
        self.header.setFixedHeight(64)
        hl = QHBoxLayout(self.header)
        hl.setContentsMargins(20, 0, 18, 0)
        titles = QVBoxLayout()
        titles.setSpacing(1)
        self.title = label("Monitoring", "title")
        self.subtitle = label(self.NAV[0][2], "subtitle")
        titles.addWidget(self.title)
        titles.addWidget(self.subtitle)
        hl.addLayout(titles)
        hl.addStretch()

        self.camera_fullscreen = QToolButton()
        self.camera_fullscreen.setText("⛶  Wall fullscreen")
        self.camera_fullscreen.setToolTip("6-camera wallni fullscreen ko'rish")
        self.camera_fullscreen.setCursor(Qt.PointingHandCursor)
        self.camera_fullscreen.clicked.connect(self.open_camera_fullscreen)
        hl.addWidget(self.camera_fullscreen)
        content_l.addWidget(self.header)

        self.stack = QStackedWidget()
        self.pages = []
        for _, _, _, klass in self.NAV:
            page = klass()
            self.pages.append(page)
            self.stack.addWidget(page)
            if isinstance(page, SettingsPage):
                page.applyRequested.connect(self.apply_camera_settings)
        content_l.addWidget(self.stack, 1)
        main.addWidget(self.content, 1)

        self.nav_buttons[0].setChecked(True)

    def switch_page(self, index: int) -> None:
        if not (0 <= index < len(self.NAV)):
            return
        if self._monitoring_fullscreen:
            self.pages[0].exit_fullscreen()
        self.stack.setCurrentIndex(index)
        _, title, subtitle, _ = self.NAV[index]
        self.title.setText(title)
        self.subtitle.setText(subtitle)
        self.camera_fullscreen.setVisible(index == 0)

    def set_monitoring_fullscreen(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._monitoring_fullscreen:
            return
        self._monitoring_fullscreen = enabled

        # Keep one top-level X11 window state for the lifetime of the EGL child.
        # Toggling showFullScreen()/showNormal() can recreate/reparent native child
        # windows under X11/XWayland and makes GstVideoOverlay paint outside the
        # camera panel. The app is maximized from startup; fullscreen monitoring is
        # therefore just a shell/layout change, not a window-manager mode change.
        self.sidebar.setVisible(not enabled)
        self.header.setVisible(not enabled)

    def open_camera_fullscreen(self) -> None:
        self.pages[0].open_fullscreen_grid()

    def apply_camera_settings(self) -> None:
        settings_page = next(
            (page for page in self.pages if isinstance(page, SettingsPage)),
            None,
        )
        current = self.stack.currentWidget()
        old_monitoring = self.pages[0]
        old_monitoring.shutdown()
        self.stack.removeWidget(old_monitoring)
        old_monitoring.deleteLater()

        new_monitoring = MonitoringPage()
        self.stack.insertWidget(0, new_monitoring)
        self.pages[0] = new_monitoring
        if current is old_monitoring:
            self.stack.setCurrentWidget(new_monitoring)
        else:
            self.stack.setCurrentWidget(current)

        if settings_page is not None:
            settings_page.mark_applied()

    def keyPressEvent(self, event) -> None:
        if self._monitoring_fullscreen and event.key() in (Qt.Key_Escape, Qt.Key_F11):
            self.pages[0].exit_fullscreen()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        monitoring = self.pages[0] if self.pages else None
        if monitoring is not None and hasattr(monitoring, "shutdown"):
            monitoring.shutdown()
        super().closeEvent(event)


def run():
    if os.environ.get("DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"

    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Sentinel VMS")
    app.setOrganizationName("Sentinel")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_QSS)
    print(
        f"SENTINEL_QT platform={app.platformName()} display={os.environ.get('DISPLAY', 'unset')}",
        flush=True,
    )
    window = MainWindow()
    # Settle the top-level X11 geometry before the MonitoringPage binds EGL.
    window.showMaximized()
    return app.exec()


def main() -> int:
    return run()


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
