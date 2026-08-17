from __future__ import annotations

"""Reference-scale adapter for the stable Camera V2 monitoring UI.

The proven monitor_ui implementation stays untouched. This adapter changes only
presentation geometry to match the supplied screenshot: two cameras per row and
about 500x280 pixels per feed. It also fixes QScrollArea's auto-fill side effect
without changing RTSP/NVDEC/YOLO/NvDCF pipeline ownership.
"""

import multiprocessing as _mp

from . import monitor_ui as _base

TILE_WIDTH = 512
TILE_HEIGHT = 288
GRID_COLUMNS = 2
GRID_ROWS = 3
WALL_WIDTH = TILE_WIDTH * GRID_COLUMNS   # 1024
WALL_HEIGHT = TILE_HEIGHT * GRID_ROWS    # 864
CAMERA_COUNT = 6

# Save the original worker before replacing the symbol used by PipelineController.
_ORIGINAL_PIPELINE_PROCESS = _base._pipeline_process


def _apply_geometry() -> None:
    _base.TILE_WIDTH = TILE_WIDTH
    _base.TILE_HEIGHT = TILE_HEIGHT
    _base.GRID_COLUMNS = GRID_COLUMNS
    _base.GRID_ROWS = GRID_ROWS
    _base.WALL_WIDTH = WALL_WIDTH
    _base.WALL_HEIGHT = WALL_HEIGHT
    _base.CAMERA_COUNT = CAMERA_COUNT


def _reference_pipeline_process(window_id: int, command_q, status_q) -> None:
    """Spawn-safe worker wrapper.

    multiprocessing uses spawn, so parent-side module mutations do not propagate
    automatically. Reapply the exact geometry inside the child before entering the
    original proven pipeline worker.
    """
    _apply_geometry()
    _ORIGINAL_PIPELINE_PROCESS(window_id, command_q, status_q)


def _patch_qscrollarea() -> None:
    """Keep Qt's backing store from painting over the native EGL video window."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QScrollArea

    if getattr(QScrollArea, "_camera_v2_reference_patch", False):
        return

    original_set_widget = QScrollArea.setWidget

    def safe_set_widget(self, widget):
        original_set_widget(self, widget)
        # Qt documents that QScrollArea.setWidget() enables autoFillBackground.
        # nveglglessink owns this native window, so immediately disable it again.
        widget.setAutoFillBackground(False)
        widget.setAttribute(Qt.WA_NoSystemBackground, True)
        widget.setAttribute(Qt.WA_PaintOnScreen, True)

    QScrollArea.setWidget = safe_set_widget
    QScrollArea._camera_v2_reference_patch = True


def main() -> int:
    _apply_geometry()
    _patch_qscrollarea()

    # PipelineController resolves this module-global symbol at construction time.
    # The wrapper is importable by multiprocessing.spawn and reapplies geometry in
    # the child, while the underlying pipeline implementation remains unchanged.
    _base._pipeline_process = _reference_pipeline_process

    print(
        "CAMERA_UI_REFERENCE layout=2x3 "
        f"tile={TILE_WIDTH}x{TILE_HEIGHT} wall={WALL_WIDTH}x{WALL_HEIGHT} "
        "pipeline_architecture=unchanged",
        flush=True,
    )
    return _base.main()


if __name__ == "__main__":
    _mp.freeze_support()
    raise SystemExit(main())
