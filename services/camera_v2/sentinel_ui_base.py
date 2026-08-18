from __future__ import annotations

from PySide6.QtCore import Qt, QRectF, QSize, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap, QRadialGradient
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QStackedWidget, QTextEdit, QToolButton,
    QVBoxLayout, QWidget,
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
QLineEdit, QComboBox, QPlainTextEdit, QTextEdit {{ background: {C['field']}; border: 1px solid {C['border']}; border-radius: 5px; padding: 7px 9px; selection-background-color: {C['primary']}; selection-color: #07110f; }}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{ border-color: {C['primary']}; }}
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
QScrollBar:horizontal {{ background: transparent; height: 9px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #273642; border-radius: 4px; min-width: 35px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QProgressBar {{ background: {C['panel2']}; border: 0; border-radius: 3px; height: 6px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {C['primary']}; border-radius: 3px; }}
QTableWidget {{ background: {C['panel']}; alternate-background-color: #101923; border: 1px solid {C['border']}; gridline-color: {C['border']}; selection-background-color: #17313a; }}
QHeaderView::section {{ background: {C['panel2']}; color: {C['muted']}; border: 0; border-bottom: 1px solid {C['border']}; padding: 9px; font-family: 'DejaVu Sans Mono'; font-size: 10px; }}
QDialog {{ background: {C['bg']}; }}
QToolTip {{ background:#111a23; color:{C['text']}; border:1px solid #304253; padding:5px 7px; }}
"""


def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            clear_layout(item.layout())


def label(text: str, role: str | None = None, color: str | None = None) -> QLabel:
    widget = QLabel(text)
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


class StatCard(Panel):
    def __init__(self, heading: str, value: str, tone: str = "text", hint: str = ""):
        super().__init__()
        self.setMinimumHeight(116)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(15, 14, 15, 13)
        lay.setSpacing(5)
        lay.addWidget(label(heading.upper(), "eyebrow"))
        lay.addWidget(label(str(value), "metric", C[tone]))
        if hint:
            lay.addWidget(label(hint, "muted"))
        lay.addStretch()


class FaceAvatar(QWidget):
    def __init__(self, person, size=64):
        super().__init__()
        self.person = person
        self.setFixedSize(size, size)

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        c1 = QColor.fromHsl(self.person.hue, 115, 90)
        c2 = QColor.fromHsl((self.person.hue + 60) % 360, 100, 35)
        gradient = QRadialGradient(self.width() * .3, self.height() * .2, self.width())
        gradient.setColorAt(0, c1)
        gradient.setColorAt(1, c2)
        painter.setBrush(gradient)
        painter.setPen(QPen(QColor(C['border']), 1))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 6, 6)
        initials = "".join(x[0] for x in self.person.label.replace("_", " ").split()[:2]).upper()
        painter.setPen(QColor(235, 240, 245, 220))
        painter.setFont(QFont("DejaVu Sans", max(10, self.width() // 4), QFont.Bold))
        painter.drawText(self.rect(), Qt.AlignCenter, initials)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(C['known'] if self.person.known else C['unknown']))
        painter.drawRect(0, self.height() - 4, self.width(), 4)


class BarChart(QWidget):
    def __init__(self, series: list[tuple[str, list[int], str]], labels: list[str], parent=None):
        super().__init__(parent)
        self.series = series
        self.labels = labels
        self.setMinimumHeight(220)

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        area = self.rect().adjusted(44, 16, -16, -28)
        maxv = max([1] + [v for _, vals, _ in self.series for v in vals])
        painter.setFont(QFont("DejaVu Sans Mono", 7))
        for i in range(5):
            y = area.bottom() - i * area.height() / 4
            painter.setPen(QColor(C['border']))
            painter.drawLine(area.left(), int(y), area.right(), int(y))
            painter.setPen(QColor(C['muted']))
            painter.drawText(2, int(y - 8), 36, 16, Qt.AlignRight | Qt.AlignVCenter, str(round(maxv * i / 4)))
        count = max(1, len(self.labels))
        group = area.width() / count
        bw = min(16, group / (len(self.series) + 1))
        for j, text in enumerate(self.labels):
            painter.setPen(QColor(C['muted']))
            painter.drawText(QRectF(area.left() + j * group, area.bottom() + 5, group, 18), Qt.AlignCenter, text)
            for k, (_, vals, color) in enumerate(self.series):
                h = vals[j] / maxv * area.height()
                x = area.left() + j * group + (group - len(self.series) * bw) / 2 + k * bw
                painter.fillRect(QRectF(x, area.bottom() - h, bw - 3, h), QColor(color))


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
