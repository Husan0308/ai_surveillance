from __future__ import annotations

"""Production camera wall entrypoint with the Stage-22-proven grid alignment.

This is intentionally a validation shim: it changes only the production grid
height from 1350 to 1352 before importing the normal Sentinel UI. All controller,
Qt/XID, DeepStream, RF-DETR, analysis, metadata and focus code remains production.
"""

import multiprocessing as mp

from . import camera_wall_runtime as _wall_runtime

_wall_runtime.WALL_HEIGHT = 1352

from .sentinel_ui import main


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
