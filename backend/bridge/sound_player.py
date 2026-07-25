import os

from PySide6.QtCore import QObject, QUrl
from PySide6.QtWidgets import QApplication

from backend.core.logger import get_logger

log = get_logger("bridge.sound")


try:
    from PySide6.QtMultimedia import QSoundEffect
except Exception:
    QSoundEffect = None


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SoundPlayer(QObject):
    def __init__(self, config):
        super().__init__()

        self.enabled = bool(config.get("alerts.sound_enabled", True))

        self.effect = None

        sound_path = os.path.join(BASE_DIR, "assets", "sounds", "shutter.wav")

        if QSoundEffect is not None and os.path.exists(sound_path):
            try:
                self.effect = QSoundEffect()
                self.effect.setSource(QUrl.fromLocalFile(sound_path))
                self.effect.setVolume(0.7)
            except Exception as e:
                log.error("QSoundEffect init error: %s", e)

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled)

    def play_shutter(self):
        if not self.enabled:
            return

        try:
            if self.effect is not None:
                self.effect.play()
            else:
                QApplication.beep()
        except Exception:
            QApplication.beep()

    def play_alert(self, event_type: str = None):
        if not self.enabled:
            return

        try:
            QApplication.beep()
        except Exception:
            pass