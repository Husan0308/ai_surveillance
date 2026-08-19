from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from services.frontend.app.operator_window import OperatorWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Apsidal AI Surveillance")
    window = OperatorWindow()
    window.showMaximized()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
