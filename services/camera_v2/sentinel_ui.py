from __future__ import annotations

import multiprocessing as mp

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
from .sentinel_ui_monitoring import MonitoringPage
from .sentinel_ui_pages import EventsPage, PeoplePage, RoomsPage
from .sentinel_ui_settings import SettingsPage


class MainWindow(QMainWindow):
    NAV = [
        ("▣", "Monitoring", "Jonli DeepStream camera wall · tracking · heatmap", MonitoringPage),
        ("♙", "People", f"{len(PEOPLE)} ta global ID", PeoplePage),
        ("⌁", "Events", f"{len(EVENTS)} ta hodisa", EventsPage),
        ("▥", "Rooms", "Kameralar orasidagi global identity holati", RoomsPage),
        ("♙+", "Enrollment", "10 ta yuz rasmi va profile photo bilan ro'yxatga olish", EnrollmentPage),
        ("⚙", "Settings", "Camera sources · RTSP · enable/disable · add/edit/delete", SettingsPage),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SENTINEL VMS")
        self.resize(1440, 900)
        self.setMinimumSize(1180, 720)
        self._monitoring_fullscreen = False
        self._restore_maximized = False

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(224)
        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(0)

        brand = QFrame()
        brand.setFixedHeight(70)
        brand.setStyleSheet(f"border-bottom:1px solid {C['border']};")
        bl = QHBoxLayout(brand)
        bl.setContentsMargins(16, 0, 12, 0)
        shield = label("◇", color=C["primary"])
        shield.setStyleSheet(f"color:{C['primary']};font-size:24px;")
        bl.addWidget(shield)
        bt = QVBoxLayout()
        bt.setSpacing(1)
        bt.addWidget(label("SENTINEL VMS", "brand"))
        bt.addWidget(label("edge ai · deepstream", "mono"))
        bl.addLayout(bt)
        bl.addStretch()
        side.addWidget(brand)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons = []
        navwrap = QWidget()
        nl = QVBoxLayout(navwrap)
        nl.setContentsMargins(8, 8, 8, 8)
        nl.setSpacing(2)
        for i, (icon, title, _, _) in enumerate(self.NAV):
            button = make_button(f"{icon:>2}   {title}")
            button.setObjectName("nav")
            button.setCheckable(True)
            button.setFixedHeight(38)
            button.clicked.connect(lambda _, i=i: self.switch_page(i))
            self.nav_group.addButton(button)
            self.nav_buttons.append(button)
            nl.addWidget(button)
        nl.addStretch()
        side.addWidget(navwrap, 1)
        build = label("build 2026.08 · edge worker", "mono")
        build.setStyleSheet(
            f"border-top:1px solid {C['border']};padding:14px;color:{C['muted']};"
        )
        side.addWidget(build)
        main.addWidget(self.sidebar)

        self.content = QWidget()
        content_l = QVBoxLayout(self.content)
        content_l.setContentsMargins(0, 0, 0, 0)
        content_l.setSpacing(0)

        self.header = QFrame()
        self.header.setObjectName("header")
        self.header.setFixedHeight(70)
        hl = QHBoxLayout(self.header)
        hl.setContentsMargins(24, 0, 24, 0)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self.title = label("Monitoring", "title")
        self.subtitle = label(self.NAV[0][2], "subtitle")
        titles.addWidget(self.title)
        titles.addWidget(self.subtitle)
        hl.addLayout(titles)
        hl.addStretch()

        self.camera_fullscreen = QToolButton()
        self.camera_fullscreen.setText("⛶  Fullscreen")
        self.camera_fullscreen.setToolTip("Camera wallni fullscreen ko'rish")
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
        if self._monitoring_fullscreen:
            monitoring = self.pages[0]
            monitoring.exit_fullscreen()
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

        if enabled:
            self._restore_maximized = self.isMaximized()
            self.sidebar.hide()
            self.header.hide()
            self.showFullScreen()
        else:
            self.sidebar.show()
            self.header.show()
            if self._restore_maximized:
                self.showMaximized()
            else:
                self.showNormal()

    def open_camera_fullscreen(self) -> None:
        monitoring = self.pages[0]
        monitoring.open_fullscreen_grid()

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
        if current is not old_monitoring:
            self.stack.setCurrentWidget(current)
        else:
            self.stack.setCurrentWidget(new_monitoring)

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
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Sentinel VMS")
    app.setOrganizationName("Sentinel")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_QSS)
    window = MainWindow()
    window.show()
    return app.exec()


def main() -> int:
    return run()


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
