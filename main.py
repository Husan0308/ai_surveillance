# =====================================================================
#  AI SURVEILLANCE SYSTEM — PRODUCTION ENTRY
# =====================================================================

import signal
import sys
import os
import os
import sys
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor

import ui

from backend.core.service_manager import ServiceManager
from backend.bridge.system_bridge import build_real_system
from backend.bridge.sound_player import SoundPlayer
from backend.bridge.ui_patches import apply_ui_patches

import atexit


def patch_login(ui_module, sm):
    """
    LoginScreen parolni hash orqali tekshiradi.
    """

    def patched_login_check(self):
        try:
            username = self.user.text().strip()
            password = self.pwd.text()

            expected_username = sm.config.get("security.login_username", "admin")

            if username == expected_username and sm.settings_service.verify_password(password):
                self.accept()
            else:
                self.err.show()
                self.pwd.clear()
                self.pwd.setFocus()

        except Exception as e:
            print("[login] error:", e)

    ui_module.LoginScreen.check = patched_login_check


def build_palette():
    pal = QPalette()

    pal.setColor(QPalette.Window, QColor(ui.TH.PANEL))
    pal.setColor(QPalette.WindowText, QColor(ui.TH.TXT))
    pal.setColor(QPalette.Base, QColor(ui.TH.CARD))
    pal.setColor(QPalette.Text, QColor(ui.TH.TXT))
    pal.setColor(QPalette.Button, QColor(ui.TH.CARD2))
    pal.setColor(QPalette.ButtonText, QColor(ui.TH.TXT))
    pal.setColor(QPalette.Highlight, QColor(ui.TH.ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor("white"))

    return pal

def main():
    # ═══ GRACEFUL SHUTDOWN ═══
    def _shutdown(signum=None, frame=None):
        print("\n[Shutdown] Dastur to'xtatilmoqda...", flush=True)
        try:
            # Barcha thread larni to'xtatish
            import threading
            for t in threading.enumerate():
                if t != threading.main_thread() and t.is_alive():
                    print(f"  Stopping thread: {t.name}", flush=True)
            
            # Qt app ni to'xtatish
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                # Barcha oynalarni yopish
                for w in app.allWindows():
                    w.close()
                app.quit()
        except Exception as e:
            print(f"  Shutdown error: {e}", flush=True)
        
        print("[Shutdown] ✅ To'liq to'xtadi", flush=True)
        os._exit(0)  # Majburiy chiqish — barcha thread/process larni o'ldirish
    
    signal.signal(signal.SIGINT, _shutdown)   # Ctrl+C
    signal.signal(signal.SIGTERM, _shutdown)  # kill command

    import sys
    WEBCAM_DEMO = "--webcam" in sys.argv
    app = QApplication(sys.argv)

    app.setStyle("Fusion")
    app.setPalette(build_palette())
    app.setStyleSheet(ui.STYLE)

    splash = ui.SplashScreen()

    if splash.exec() != ui.QDialog.Accepted:
        sys.exit(0)

    sm = None

    try:
        sm = ServiceManager()
        sm.start()

        real_system = build_real_system(sm)

        sound_player = SoundPlayer(sm.config)

        # MUHIM: thread'larni saqlab qolish uchun app ga biriktiramiz
        app._service_manager = sm
        app._real_system = real_system
        app._sound_player = sound_player

        # DEBUG
        print(f"[DEBUG] AI workers: {len(sm.ai_workers)}", flush=True)
        print(f"[DEBUG] Camera IDs: {list(sm.ai_workers.keys())}", flush=True)
        print(f"[DEBUG] Camera sims: {list(sm.camera_manager.workers.keys())}", flush=True)
        print(f"[DEBUG] RealSystem sims: {[c.id for c in real_system.sims]}", flush=True)

        apply_ui_patches(
            ui=ui,
            sm=sm,
            real_system=real_system,
            sound_player=sound_player,
        )

        patch_login(ui, sm)

    except Exception as e:
        traceback.print_exc()

        if sm is not None:
            sm.shutdown()

        raise

    # login = ui.LoginScreen()

    # if login.exec() != ui.QDialog.Accepted:
    #     sm.shutdown()
    #     sys.exit(0)

    win = ui.MainWindow()
    win.show()

    # MainWindow ga ham saqlaymiz
    win._service_manager = sm
    # Events → UI signal zanjiri
    try:
        sys_obj = getattr(win, 'sys', None)
        if sys_obj and hasattr(sys_obj, 'new_event'):
            def _on_event_to_ui(evt):
                print(f"[MAIN] 🎯 UI EVENT: type={evt.get('type')} person={evt.get('person_name')}", flush=True)
                sys_obj.new_event.emit(evt)
            # sm.events_service.event_added.connect(_on_event_to_ui)  # system_bridge.py:714 allaqachon ulagan
            print("[MAIN] ✅ event_added → sys.new_event ulandi", flush=True)
        else:
            print("[MAIN] ⚠ sys.new_event topilmadi", flush=True)
    except Exception as e:
        print(f"[MAIN] ⚠ Event connect error: {e}", flush=True)
    win._real_system = real_system

    def on_quit():
        try:
            try:
                real_system.enroll_sim.stop()
                print("[shutdown] Enrollment kamera yopildi ⏹", flush=True)
            except Exception as e:
                print("[shutdown] enroll_sim stop error:", e)
            sm.shutdown()
        except Exception as e:
            print("[shutdown] error:", e)

    app.aboutToQuit.connect(on_quit)

    # ✅ Qo'shimcha kafolat — dastur qanday yopilsa ham kamera bo'shaydi
    def _force_stop():
        try:
            real_system.enroll_sim.stop()
        except Exception:
            pass
    atexit.register(_force_stop)

    sys.exit(app.exec())
    
if __name__ == "__main__":
    main()