"""Canonical PySide6 frontend entry point."""
from __future__ import annotations

import signal
import sys

from PySide6.QtGui import QColor,QPalette
from PySide6.QtWidgets import QApplication,QDialog

from services.frontend import ui
from shared.logging import configure_logging,get_logger
from shared.settings import ServiceSettings


def _palette():
    palette=QPalette()
    palette.setColor(QPalette.Window,QColor(ui.TH.PANEL))
    palette.setColor(QPalette.WindowText,QColor(ui.TH.TXT))
    palette.setColor(QPalette.Base,QColor(ui.TH.CARD))
    palette.setColor(QPalette.Text,QColor(ui.TH.TXT))
    palette.setColor(QPalette.Button,QColor(ui.TH.CARD2))
    palette.setColor(QPalette.ButtonText,QColor(ui.TH.TXT))
    palette.setColor(QPalette.Highlight,QColor(ui.TH.ACCENT))
    palette.setColor(QPalette.HighlightedText,QColor("white"))
    return palette


def main():
    settings=ServiceSettings.from_env();configure_logging(settings.log_level,"frontend");log=get_logger(__name__)
    app=QApplication(sys.argv);app.setStyle("Fusion");app.setPalette(_palette());app.setStyleSheet(ui.STYLE)
    signal.signal(signal.SIGINT,lambda *_:app.quit())
    signal.signal(signal.SIGTERM,lambda *_:app.quit())
    log.info("Frontend starting independently (API and WebSocket clients only)")
    splash=ui.SplashScreen()
    if splash.exec()!=QDialog.Accepted:return 0
    window=ui.MainWindow();window.show()
    app.aboutToQuit.connect(window.sys.shutdown)
    return app.exec()


if __name__=="__main__":
    raise SystemExit(main())
