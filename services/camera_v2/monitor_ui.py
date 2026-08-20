from __future__ import annotations

"""Production entry point for the camera-only Sentinel wall."""

import multiprocessing as mp

from .sentinel_ui import main


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
