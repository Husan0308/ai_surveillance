from __future__ import annotations

import multiprocessing as mp

from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QFrame, QHBoxLayout, QMainWindow, QStackedWidget,
    QToolButton, QVBoxLayout, QWidget,
)

from .data import EVENTS, PEOPLE
from .sentinel_ui_base import APP_QSS, C, label, make_button
from .sentinel_ui_enrollment import EnrollmentPage, ReportsPage
from .sentinel_ui_monitoring import MonitoringPage
from .sentinel_ui_pages import EventsPage, PeoplePage, RoomsPage


class MainWindow(QMainWindow):
    NAV = [
        ("▣","Monitoring","6 ta jonli kamera · bir ekranda · so'nggi kuzatuvlar",MonitoringPage),
        ("♙","People",f"{len(PEOPLE)} ta global ID",PeoplePage),
        ("⌁","Events",f"{len(EVENTS)} ta hodisa",EventsPage),
        ("▥","Rooms","Kameralar orasidagi bir xil odam bir marta hisoblanadi (global ID bo'yicha)",RoomsPage),
        ("♙+","Enrollment","10 ta yuz rasmi va profile photo bilan ro'yxatga olish",EnrollmentPage),
        ("▤","Reports","Kunlik va haftalik hisobotlar",ReportsPage),
    ]

    def __init__(self):
        super().__init__(); self.setWindowTitle("SENTINEL VMS"); self.resize(1440,900); self.setMinimumSize(1180,720)
        root=QWidget(); root.setObjectName("root"); self.setCentralWidget(root); main=QHBoxLayout(root); main.setContentsMargins(0,0,0,0); main.setSpacing(0)
        sidebar=QFrame(); sidebar.setObjectName("sidebar"); sidebar.setFixedWidth(224); side=QVBoxLayout(sidebar); side.setContentsMargins(0,0,0,0); side.setSpacing(0)
        brand=QFrame(); brand.setFixedHeight(70); brand.setStyleSheet(f"border-bottom:1px solid {C['border']};"); bl=QHBoxLayout(brand); bl.setContentsMargins(16,0,12,0); shield=label("◇",color=C['primary']); shield.setStyleSheet(f"color:{C['primary']};font-size:24px;"); bl.addWidget(shield); bt=QVBoxLayout(); bt.setSpacing(1); bt.addWidget(label("SENTINEL VMS","brand")); bt.addWidget(label("face re-id · 6 cam","mono")); bl.addLayout(bt); bl.addStretch(); side.addWidget(brand)
        self.nav_group=QButtonGroup(self); self.nav_group.setExclusive(True); self.nav_buttons=[]
        navwrap=QWidget(); nl=QVBoxLayout(navwrap); nl.setContentsMargins(8,8,8,8); nl.setSpacing(2)
        for i,(icon,title,_,_) in enumerate(self.NAV):
            b=make_button(f"{icon:>2}   {title}"); b.setObjectName("nav"); b.setCheckable(True); b.setFixedHeight(38); b.clicked.connect(lambda _,i=i:self.switch_page(i)); self.nav_group.addButton(b); self.nav_buttons.append(b); nl.addWidget(b)
        nl.addStretch(); side.addWidget(navwrap,1); build=label("build 2026.08 · edge worker","mono"); build.setStyleSheet(f"border-top:1px solid {C['border']};padding:14px;color:{C['muted']};"); side.addWidget(build); main.addWidget(sidebar)
        content=QWidget(); content_l=QVBoxLayout(content); content_l.setContentsMargins(0,0,0,0); content_l.setSpacing(0); header=QFrame(); header.setObjectName("header"); header.setFixedHeight(70); hl=QHBoxLayout(header); hl.setContentsMargins(24,0,24,0); titles=QVBoxLayout(); titles.setSpacing(2); self.title=label("Monitoring","title"); self.subtitle=label(self.NAV[0][2],"subtitle"); titles.addWidget(self.title); titles.addWidget(self.subtitle); hl.addLayout(titles); hl.addStretch()
        self.camera_fullscreen = QToolButton()
        self.camera_fullscreen.setText("⛶  Fullscreen")
        self.camera_fullscreen.setToolTip("Barcha kameralarni fullscreen ko'rish")
        self.camera_fullscreen.clicked.connect(self.open_camera_fullscreen)
        hl.addWidget(self.camera_fullscreen)
        content_l.addWidget(header)
        self.stack=QStackedWidget(); self.pages=[]
        for _,_,_,klass in self.NAV: page=klass(); self.pages.append(page); self.stack.addWidget(page)
        content_l.addWidget(self.stack,1); main.addWidget(content,1); self.nav_buttons[0].setChecked(True)

    def switch_page(self,index):
        self.stack.setCurrentIndex(index); _,title,subtitle,_=self.NAV[index]; self.title.setText(title); self.subtitle.setText(subtitle)
        self.camera_fullscreen.setVisible(index == 0)

    def open_camera_fullscreen(self):
        monitoring = self.pages[0]
        monitoring.open_fullscreen_grid()

    def closeEvent(self, event):
        monitoring = self.pages[0] if self.pages else None
        if monitoring is not None and hasattr(monitoring, "shutdown"):
            monitoring.shutdown()
        super().closeEvent(event)


def run():
    app=QApplication.instance() or QApplication([]); app.setApplicationName("Sentinel VMS"); app.setOrganizationName("Sentinel"); app.setStyle("Fusion"); app.setStyleSheet(APP_QSS)
    window=MainWindow(); window.show(); return app.exec()


def main() -> int:
    return run()


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
