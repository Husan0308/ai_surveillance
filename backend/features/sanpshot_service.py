import os
from datetime import datetime

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QPixmap, QPainter, QColor

from backend.core.logger import get_logger

log = get_logger("features.snapshot")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SnapshotService(QObject):
    """
    Snapshot service.

    - camera snapshot
    - event snapshot
    - wall snapshot
    """

    snapshot_taken = Signal(dict)

    def __init__(self, config):
        super().__init__()

        self.config = config

        snapshots_dir = config.get("storage.snapshots_dir", "snapshots")

        if not os.path.isabs(snapshots_dir):
            snapshots_dir = os.path.join(BASE_DIR, snapshots_dir)

        self.dir = snapshots_dir
        os.makedirs(self.dir, exist_ok=True)

        log.info("SnapshotService started: %s", self.dir)

    def _timestamp(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # ---------------- QImage snapshot ----------------
    def take_snapshot_qimage(
        self,
        camera_id: str,
        qimage,
        person_name: str = None,
        emit_signal: bool = True,
    ):
        if qimage is None:
            return None

        try:
            safe_cam = "".join(
                c for c in str(camera_id) if c.isalnum() or c in ("-", "_")
            )

            filename = f"{safe_cam}_{self._timestamp()}.png"
            path = os.path.join(self.dir, filename)

            ok = qimage.save(path, "PNG")

            if not ok:
                log.error("Snapshot save failed: %s", path)
                return None

            info = {
                "camera_id": camera_id,
                "path": path,
                "person_name": person_name,
                "time": datetime.now().isoformat(),
            }

            if emit_signal:
                self.snapshot_taken.emit(info)

            log.info("Snapshot saved: %s", path)

            return path

        except Exception as e:
            log.error("take_snapshot_qimage error: %s", e)
            return None

    # ---------------- BGR snapshot ----------------
    def take_snapshot_bgr(
        self,
        camera_id: str,
        bgr,
        person_name: str = None,
        emit_signal: bool = True,
    ):
        if bgr is None:
            return None

        try:
            import cv2

            safe_cam = "".join(
                c for c in str(camera_id) if c.isalnum() or c in ("-", "_")
            )

            filename = f"{safe_cam}_{self._timestamp()}.jpg"
            path = os.path.join(self.dir, filename)

            ok = cv2.imwrite(path, bgr)

            if not ok:
                log.error("Snapshot BGR save failed: %s", path)
                return None

            info = {
                "camera_id": camera_id,
                "path": path,
                "person_name": person_name,
                "time": datetime.now().isoformat(),
            }

            if emit_signal:
                self.snapshot_taken.emit(info)

            return path

        except Exception as e:
            log.error("take_snapshot_bgr error: %s", e)
            return None

    # ---------------- wall snapshot ----------------
    def take_wall_snapshot(self, frames: dict, columns: int = 3):
        """
        frames:
            {
                "CAM-01": QImage,
                "CAM-02": QImage,
                ...
            }
        """

        try:
            if not frames:
                return None

            items = list(frames.items())

            cell_w = 640
            cell_h = 360

            count = len(items)
            cols = max(1, int(columns))
            rows = (count + cols - 1) // cols

            grid = QPixmap(cols * cell_w, rows * cell_h)
            grid.fill(QColor("#000000"))

            p = QPainter(grid)

            for i, (camera_id, img) in enumerate(items):
                if img is None:
                    continue

                x = (i % cols) * cell_w
                y = (i // cols) * cell_h

                p.drawImage(x, y, img.scaled(cell_w, cell_h))

                p.setPen(QColor(255, 255, 255, 200))
                p.drawText(x + 10, y + 24, str(camera_id))

            p.end()

            filename = f"wall_{self._timestamp()}.png"
            path = os.path.join(self.dir, filename)

            ok = grid.save(path, "PNG")

            if not ok:
                return None

            info = {
                "camera_id": "WALL",
                "path": path,
                "person_name": None,
                "time": datetime.now().isoformat(),
            }

            self.snapshot_taken.emit(info)

            log.info("Wall snapshot saved: %s", path)

            return path

        except Exception as e:
            log.error("take_wall_snapshot error: %s", e)
            return None