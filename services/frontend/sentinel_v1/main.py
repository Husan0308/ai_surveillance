import sys
import signal

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from .ui import APP_QSS, MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Sentinel VMS")
    app.setOrganizationName("Sentinel")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_QSS)
    window = MainWindow()
    window.show()
    signal_timer = QTimer()
    signal_timer.setInterval(200)
    signal_timer.timeout.connect(lambda: None)
    signal_timer.start()
    signal.signal(signal.SIGINT, lambda _signum, _frame: app.quit())
    signal.signal(signal.SIGTERM, lambda _signum, _frame: app.quit())
    try:
        return app.exec()
    finally:
        window.close()


if __name__ == "__main__":
    raise SystemExit(main())
