from __future__ import annotations

from PySide6.QtGui import QFont

BG = "#020b18"
SIDEBAR_BG = "#061426"
PANEL = "#07192d"
CARD = "#0a2038"
CARD_ALT = "#0d2744"
BORDER = "#143a60"
TEXT = "#f5f7fb"
MUTED = "#9baabd"
BLUE = "#1368ff"
BLUE_DARK = "#0b4ed8"
GREEN = "#00e47a"
ORANGE = "#ff8a00"
RED = "#ff4057"
CYAN = "#12b8ff"


def font(px: int, weight=QFont.Weight.Normal) -> QFont:
    f = QFont("Inter")
    f.setPixelSize(px)
    f.setWeight(weight)
    return f


def stylesheet() -> str:
    return f"""
    QMainWindow, QWidget {{
        background: {BG};
        color: {TEXT};
        font-family: Inter, 'Noto Sans', sans-serif;
    }}

    #sidebar {{
        background: {SIDEBAR_BG};
        border-right: 1px solid #102e4c;
    }}

    #topbar {{
        background: {BG};
        border-bottom: 1px solid #0c2138;
    }}

    QPushButton {{
        color: {TEXT};
        border: 0;
        outline: none;
        background: transparent;
    }}

    #navButton {{
        text-align: left;
        padding-left: 16px;
        border-radius: 8px;
        font-size: 16px;
    }}

    #navButton:hover {{ background: #0b223c; }}
    #navButton:checked {{
        background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
            stop:0 #0c66ff, stop:1 #0948d6);
        border: 1px solid #1a73ff;
    }}

    #panel, #statusCard, #cameraTile, #tableCard, #placeholder {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 10px;
    }}

    #statCard {{
        background: {CARD};
        border: 1px solid #0f2b4b;
        border-radius: 9px;
    }}

    #cameraHeader, #cameraFooter {{ background: {CARD}; }}

    #squareButton, #topButton {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 7px;
    }}
    #squareButton:hover, #topButton:hover {{ background: {CARD_ALT}; }}

    QProgressBar {{
        background: #0b355e;
        border: 0;
        border-radius: 4px;
    }}
    QProgressBar::chunk {{
        background: {BLUE};
        border-radius: 4px;
    }}

    QTableWidget#dataTable {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 9px;
        gridline-color: transparent;
        color: {TEXT};
    }}
    QTableWidget#dataTable::item {{
        border-bottom: 1px solid #10304f;
        padding: 8px;
    }}
    QTableWidget#dataTable QHeaderView::section {{
        background: #0a2038;
        color: #c7d1de;
        border: 0;
        border-bottom: 1px solid {BORDER};
        padding: 10px;
        font-size: 12px;
        font-weight: 600;
    }}

    QScrollArea {{ background: transparent; border: 0; }}
    QScrollArea > QWidget > QWidget {{ background: transparent; }}
    QScrollBar:vertical {{
        background: #07192d;
        width: 7px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: #24527a;
        min-height: 30px;
        border-radius: 3px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    """
