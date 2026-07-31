import cv2

from PySide6.QtWidgets import (
    QMenu, QMessageBox, QFileDialog, QPushButton, QLabel,
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

from backend.bridge.widgets import (
    FlashOverlay,
    CamerasDialog,
    PersonFormDialog,
    UnknownFacesDialog,
)

from backend.core.logger import get_logger

log = get_logger("bridge.ui_patches")


def apply_ui_patches(ui, sm, real_system, sound_player):
    """
    ui.py ga tegmasdan UI ni backend ga ulaydi.
    """
    # ✅ EnrollmentService ga PersonService referensini berish
    try:
        sm.enrollment_service._person_service = sm.person_service
        print("[Patch] EnrollmentService._person_service linked", flush=True)
    except Exception as e:
        print(f"[Patch] Link error: {e}", flush=True)

    # ---------------- System factory ----------------
    def system_factory():
        return real_system

    ui.System = system_factory

    # ---------------- PasswordDialog ----------------
    def patched_password_check(self):
        try:
            if sm.settings_service.verify_password(self.edit.text()):
                self.accept()
            else:
                self.err.show()
                self.edit.clear()
                self.edit.setFocus()
        except Exception as e:
            log.error("password check error: %s", e)

    ui.PasswordDialog.check = patched_password_check

    # ---------------- MainWindow ----------------
    def patched_slow_tick(self):
        try:
            self.header.update_stats()
            self.header.tick_clock()
            # self.analytics.push()  # ✅ Analytics removed
        except Exception as e:
            log.error("slow_tick error: %s", e)

    ui.MainWindow.slow_tick = patched_slow_tick

    def patched_on_event(self, e):
        try:
            self.right.add_event(e)
            self.events_pg.add_event(e)
            self.notif.add_alert(e)

            if e.get("level") in ("warn", "err"):
                self.header.bump()
        except Exception as ex:
            log.error("on_event error: %s", ex)

    ui.MainWindow.on_event = patched_on_event

    def patched_snapshot(self, sim):
        try:
            if not sim.online or sim.frame is None:
                self.toast("⚠ Camera is offline")
                return

            path = sm.snapshot_service.take_snapshot_qimage(
                camera_id=sim.id,
                qimage=sim.frame,
            )

            if path:
                self.toast(f"📸 Snapshot saved — {path}")

        except Exception as e:
            log.error("snapshot error: %s", e)

    ui.MainWindow.snapshot = patched_snapshot

    # ---------------- Dashboard wall snapshot ----------------
    def patched_wall_snapshot(self):
        try:
            frames = {}

            for sim in self.hub.sys.sims[:6]:
                if sim.frame is not None:
                    frames[sim.id] = sim.frame

            path = sm.snapshot_service.take_wall_snapshot(frames)

            if path:
                self.hub.toast(f"📸 Wall snapshot saved — {path}")

        except Exception as e:
            log.error("wall_snapshot error: %s", e)

    ui.DashboardPage.wall_snapshot = patched_wall_snapshot

