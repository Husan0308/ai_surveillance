from __future__ import annotations

"""Canonical entry point for the Sentinel VMS desktop shell."""

import multiprocessing as mp

from .sentinel_ui import main


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
