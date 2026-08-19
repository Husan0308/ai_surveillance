from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

C = {
    "bg": "#070b10",
    "sidebar": "#0a1016",
    "panel": "#0d141c",
    "panel2": "#111b25",
    "border": "#22313f",
    "text": "#e8eef4",
    "muted": "#7f8e9c",
    "primary": "#35d5c0",
    "known": "#39d995",
    "unknown": "#f2b84b",
    "offline": "#ef6666",
    "blue": "#68aaff",
    "violet": "#a78bfa",
    "field": "#091018",
}

APP_QSS = f"""
* {{ color: {C['text']}; font-family: 'DejaVu Sans'; font-size: 12px; }}
QMainWindow, QWidget#root, QWidget#pageRoot {{ background: {C['bg']}; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: 0; }}
QFrame#sidebar {{ background: {C['sidebar']}; border-right: 1px solid {C['border']}; }}
QFrame#header {{ background: {C['bg']}; border-bottom: 1px solid {C['border']}; }}
QFrame#panel {{ background: {C['panel']}; border: 1px solid {C['border']}; border-radius: 7px; }}
QLabel#title {{ font-size: 18px; font-weight: 800; }}
QLabel#subtitle, QLabel#muted {{ color: {C['muted']}; }}
QLabel#eyebrow {{ color: {C['muted']}; font-family: 'DejaVu Sans Mono'; font-size: 9px; letter-spacing: 1px; }}
QLabel#brand {{ font-size: 13px; font-weight: 800; letter-spacing: 1px; }}
QLabel#metric {{ font-size: 29px; font-weight: 800; }}
QLabel#sectionTitle {{ font-size: 13px; font-weight: 750; }}
QLabel#mono {{ color: {C['muted']}; font-family: 'DejaVu Sans Mono'; font-size: 9px; }}
QPushButton {{ background: transparent; border: 1px solid {C['border']}; border-radius: 5px; padding: 7px 11px; min-height: 18px; }}
QPushButton:hover {{ background: {C['panel2']}; border-color: #304457; }}
QPushButton:pressed {{ background: #16222d; }}
QPushButton:disabled {{ color: #53616d; border-color: #19242d; background: #0a1016; }}
QPushButton:checked {{ background: {C['panel2']}; color: {C['text']}; border-color: #34495b; }}
QPushButton#primary {{ background: {C['primary']}; border-color: {C['primary']}; color: #06110f; font-weight: 800; }}
QPushButton#primary:hover {{ background: #50e3d1; border-color: #50e3d1; }}
QPushButton#primary:disabled {{ background: #17342f; border-color: #17342f; color: #64827d; }}
QPushButton#secondary {{ background: {C['panel2']}; }}
QPushButton#ghost {{ border-color: transparent; color: {C['muted']}; }}
QPushButton#nav {{ text-align: left; border: 0; border-radius: 5px; padding: 9px 11px; color: #96a4b0; }}
QPushButton#nav:hover {{ background: #101a23; color: {C['text']}; }}
QPushButton#nav:checked {{ background: #12242c; color: {C['primary']}; }}
QToolButton {{ background: transparent; border: 1px solid {C['border']}; border-radius: 5px; padding: 7px; }}
QToolButton:hover {{ background: {C['panel2']}; border-color: #304457; }}
QToolButton:pressed {{ background: #16222d; }}
QLineEdit, QComboBox, QTextEdit {{ background: {C['field']}; border: 1px solid {C['border']}; border-radius: 5px; padding: 7px 9px; selection-background-color: {C['primary']}; selection-color: #07110f; }}
QLineEdit:focus, QComboBox:focus, QTextEdit:focus {{ border-color: {C['primary']}; }}
QLineEdit:disabled, QComboBox:disabled {{ color:#5b6974; background:#080d12; }}
QComboBox::drop-down {{ border: 0; width: 22px; }}
QComboBox QAbstractItemView {{ background: {C['panel']}; border: 1px solid {C['border']}; selection-background-color: {C['panel2']}; }}
QCheckBox {{ spacing: 7px; color: #b4c0ca; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid #3a4b5b; border-radius: 3px; background: #091018; }}
QCheckBox::indicator:hover {{ border-color: {C['primary']}; }}
QCheckBox::indicator:checked {{ background: {C['primary']}; border-color: {C['primary']}; }}
QScrollBar:vertical {{ background: transparent; width: 9px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #273642; border-radius: 4px; min-height: 35px; }}
QScrollBar::handle:vertical:hover {{ background: #344858; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QToolTip {{ background:#111a23; color:{C['text']}; border:1px solid #304253; padding:5px 7px; }}
"""


def clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            clear_layout(item.layout())


def label(text: str, role: str | None = None, color: str | None = None) -> QLabel:
    widget = QLabel(str(text))
    if role:
        widget.setObjectName(role)
    if color:
        widget.setStyleSheet(f"color:{color};")
    return widget


class Panel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")


class ScrollPage(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.body = QWidget()
        self.body.setObjectName("pageRoot")
        self.layout = QVBoxLayout(self.body)
        self.layout.setContentsMargins(20, 18, 20, 22)
        self.layout.setSpacing(14)
        self.setWidget(self.body)


def panel_layout(panel: Panel, margins=(16, 15, 16, 15), spacing=8):
    lay = QVBoxLayout(panel)
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    return lay


def make_button(text: str, role: str = "") -> QPushButton:
    button = QPushButton(text)
    if role:
        button.setObjectName(role)
    button.setCursor(Qt.PointingHandCursor)
    return button
