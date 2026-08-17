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
    "bg": "#090d12", "sidebar": "#0b1118", "panel": "#0e151d", "panel2": "#111b25",
    "border": "#22303e", "text": "#e7edf3", "muted": "#7e8c99", "primary": "#39d9c5",
    "known": "#3ddc97", "unknown": "#f6b94b", "offline": "#f06464", "blue": "#65a8ff",
    "violet": "#a78bfa", "field": "#0b1219",
}


APP_QSS = f"""
* {{ color: {C['text']}; font-family: 'DejaVu Sans'; font-size: 12px; }}
QMainWindow, QWidget#root, QWidget#pageRoot, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {C['bg']}; }}
QFrame#sidebar {{ background: {C['sidebar']}; border-right: 1px solid {C['border']}; }}
QFrame#header {{ background: {C['bg']}; border-bottom: 1px solid {C['border']}; }}
QFrame#panel {{ background: {C['panel']}; border: 1px solid {C['border']}; border-radius: 6px; }}
QLabel#title {{ font-size: 18px; font-weight: 700; }}
QLabel#subtitle, QLabel#muted {{ color: {C['muted']}; }}
QLabel#eyebrow {{ color: {C['muted']}; font-family: 'DejaVu Sans Mono'; font-size: 10px; letter-spacing: 1px; }}
QLabel#brand {{ font-size: 14px; font-weight: 700; letter-spacing: 1px; }}
QLabel#metric {{ font-size: 29px; font-weight: 700; }}
QLabel#sectionTitle {{ font-size: 14px; font-weight: 700; }}
QLabel#mono {{ color: {C['muted']}; font-family: 'DejaVu Sans Mono'; font-size: 10px; }}
QPushButton {{ background: transparent; border: 1px solid {C['border']}; border-radius: 5px; padding: 7px 11px; }}
QPushButton:hover {{ background: {C['panel2']}; }}
QPushButton:checked {{ background: {C['panel2']}; color: {C['text']}; }}
QPushButton#primary {{ background: {C['primary']}; border-color: {C['primary']}; color: #07110f; font-weight: 700; }}
QPushButton#primary:hover {{ background: #52e5d3; }}
QPushButton#secondary {{ background: {C['panel2']}; }}
QPushButton#ghost {{ border-color: transparent; color: {C['muted']}; }}
QPushButton#nav {{ text-align: left; border: 0; border-radius: 5px; padding: 9px 12px; color: #9ca9b4; }}
QPushButton#nav:hover {{ background: #111c26; color: {C['text']}; }}
QPushButton#nav:checked {{ background: #14242d; color: {C['primary']}; }}
QToolButton {{ background: transparent; border: 1px solid {C['border']}; border-radius: 5px; padding: 7px; }}
QToolButton:hover {{ background: {C['panel2']}; }}
QLineEdit, QComboBox, QPlainTextEdit, QTextEdit {{ background: {C['field']}; border: 1px solid {C['border']}; border-radius: 5px; padding: 7px 9px; selection-background-color: {C['primary']}; selection-color: #07110f; }}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{ border-color: {C['primary']}; }}
QComboBox::drop-down {{ border: 0; width: 22px; }}
QComboBox QAbstractItemView {{ background: {C['panel']}; border: 1px solid {C['border']}; selection-background-color: {C['panel2']}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #283743; border-radius: 4px; min-height: 35px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QProgressBar {{ background: {C['panel2']}; border: 0; border-radius: 3px; height: 6px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {C['primary']}; border-radius: 3px; }}
QTableWidget {{ background: {C['panel']}; alternate-background-color: #101923; border: 1px solid {C['border']}; gridline-color: {C['border']}; selection-background-color: #17313a; }}
QHeaderView::section {{ background: {C['panel2']}; color: {C['muted']}; border: 0; border-bottom: 1px solid {C['border']}; padding: 9px; font-family: 'DejaVu Sans Mono'; font-size: 10px; }}
QDialog {{ background: {C['bg']}; }}
"""


def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            clear_layout(item.layout())


def label(text: str, role: str | None = None, color: str | None = None) -> QLabel:
    w = QLabel(text)
    if role:
        w.setObjectName(role)
    if color:
        w.setStyleSheet(f"color:{color};")
    return w


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
        self.layout.setContentsMargins(24, 22, 24, 28)
        self.layout.setSpacing(16)
        self.setWidget(self.body)


class StatCard(Panel):
    def __init__(self, heading: str, value: str, tone: str = "text", hint: str = ""):
        super().__init__()
        self.setMinimumHeight(120)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 15, 16, 14)
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
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c1 = QColor.fromHsl(self.person.hue, 115, 90)
        c2 = QColor.fromHsl((self.person.hue + 60) % 360, 100, 35)
        g = QRadialGradient(self.width() * .3, self.height() * .2, self.width())
        g.setColorAt(0, c1); g.setColorAt(1, c2)
        p.setBrush(g); p.setPen(QPen(QColor(C['border']), 1)); p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 6, 6)
        initials = "".join(x[0] for x in self.person.label.replace("_", " ").split()[:2]).upper()
        p.setPen(QColor(235, 240, 245, 220)); p.setFont(QFont("DejaVu Sans", max(10, self.width() // 4), QFont.Bold))
        p.drawText(self.rect(), Qt.AlignCenter, initials)
        p.setPen(Qt.NoPen); p.setBrush(QColor(C['known'] if self.person.known else C['unknown']))
        p.drawRect(0, self.height()-4, self.width(), 4)


class BarChart(QWidget):
    def __init__(self, series: list[tuple[str, list[int], str]], labels: list[str], parent=None):
        super().__init__(parent); self.series = series; self.labels = labels; self.setMinimumHeight(220)

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        area = self.rect().adjusted(44, 16, -16, -28); maxv = max([1] + [v for _, vals, _ in self.series for v in vals])
        p.setFont(QFont("DejaVu Sans Mono", 7));
        for i in range(5):
            y = area.bottom() - i*area.height()/4; p.setPen(QColor(C['border'])); p.drawLine(area.left(), int(y), area.right(), int(y))
            p.setPen(QColor(C['muted'])); p.drawText(2, int(y-8), 36, 16, Qt.AlignRight|Qt.AlignVCenter, str(round(maxv*i/4)))
        count = max(1, len(self.labels)); group = area.width()/count; bw = min(16, group/(len(self.series)+1))
        for j, text in enumerate(self.labels):
            p.setPen(QColor(C['muted'])); p.drawText(QRectF(area.left()+j*group, area.bottom()+5, group, 18), Qt.AlignCenter, text)
            for k, (_, vals, color) in enumerate(self.series):
                h = vals[j]/maxv*area.height(); x = area.left()+j*group+(group-len(self.series)*bw)/2+k*bw
                p.fillRect(QRectF(x, area.bottom()-h, bw-3, h), QColor(color))


def panel_layout(panel: Panel, margins=(16, 15, 16, 15), spacing=8):
    lay = QVBoxLayout(panel); lay.setContentsMargins(*margins); lay.setSpacing(spacing); return lay


def make_button(text: str, role: str = "") -> QPushButton:
    b = QPushButton(text)
    if role: b.setObjectName(role)
    b.setCursor(Qt.PointingHandCursor)
    return b
