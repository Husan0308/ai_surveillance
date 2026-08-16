from __future__ import annotations

# Stable public entrypoint. The previous in-file implementation is replaced by
# qt_app.py so the UI shell can be shown before any DeepStream runtime is built.
from .qt_app import main


if __name__ == "__main__":
    raise SystemExit(main())
