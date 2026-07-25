from PySide6.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox, QSpinBox, QComboBox,
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox,
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve

from PySide6.QtWidgets import QGraphicsOpacityEffect


# ============================ FLASH OVERLAY ==========================
class FlashOverlay(QWidget):
    """
    Enrollment capture paytida qisqa oq flash.
    Mavjud video surface ustida overlay.
    """

    def __init__(self, parent):
        super().__init__(parent)

        self.setStyleSheet("background: white;")
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        self.effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.effect)
        self.effect.setOpacity(0)

        self._anim = None

        self.hide()

    def flash(self):
        try:
            if self.parentWidget() is not None:
                self.setGeometry(self.parentWidget().rect())

            self.show()
            self.raise_()

            self._anim = QPropertyAnimation(self.effect, b"opacity")
            self._anim.setDuration(220)
            self._anim.setStartValue(0.85)
            self._anim.setEndValue(0.0)
            self._anim.setEasingCurve(QEasingCurve.OutQuad)
            self._anim.finished.connect(self.hide)
            self._anim.start()

        except Exception:
            pass


# ============================ CAMERA FORM ============================
class CameraFormDialog(QDialog):
    def __init__(self, cam=None, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Camera")
        self.setModal(True)
        self.setMinimumWidth(420)

        self.cam = cam or {}

        v = QVBoxLayout(self)
        f = QFormLayout()

        self.id_edit = QLineEdit(self.cam.get("id", ""))
        self.id_edit.setPlaceholderText("CAM-09 (auto if empty)")

        self.name_edit = QLineEdit(self.cam.get("name", ""))
        self.location_edit = QLineEdit(self.cam.get("location", ""))

        self.source_edit = QLineEdit(self.cam.get("source", ""))
        self.source_edit.setPlaceholderText("rtsp://... yoki 0 (webcam)")

        self.username_edit = QLineEdit(self.cam.get("username", ""))
        self.password_edit = QLineEdit(self.cam.get("password", ""))
        self.password_edit.setEchoMode(QLineEdit.Password)

        self.online_check = QCheckBox("Online")
        self.online_check.setChecked(bool(self.cam.get("online", False)))

        self.ai_check = QCheckBox("AI enabled")
        self.ai_check.setChecked(bool(self.cam.get("ai_enabled", True)))

        self.heatmap_check = QCheckBox("Heatmap enabled")
        self.heatmap_check.setChecked(bool(self.cam.get("heatmap_enabled", False)))

        self.recording_check = QCheckBox("Recording enabled")
        self.recording_check.setChecked(bool(self.cam.get("recording_enabled", False)))

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(int(self.cam.get("fps", 25)))

        self.res_combo = QComboBox()
        self.res_combo.addItems(["1920x1080", "1280x720", "2560x1440"])
        self.res_combo.setCurrentText(self.cam.get("resolution", "1920x1080"))

        f.addRow("ID", self.id_edit)
        f.addRow("Name", self.name_edit)
        f.addRow("Location", self.location_edit)
        f.addRow("Source", self.source_edit)
        f.addRow("Username", self.username_edit)
        f.addRow("Password", self.password_edit)
        f.addRow("", self.online_check)
        f.addRow("", self.ai_check)
        f.addRow("", self.heatmap_check)
        f.addRow("", self.recording_check)
        f.addRow("FPS", self.fps_spin)
        f.addRow("Resolution", self.res_combo)

        v.addLayout(f)

        row = QHBoxLayout()

        self.test_btn = QPushButton("Connection Test")
        self.test_btn.setObjectName("btnGhost")

        self.save_btn = QPushButton("Save")
        self.save_btn.setObjectName("btnPrimary")

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("btnGhost")

        row.addWidget(self.test_btn)
        row.addStretch(1)
        row.addWidget(self.cancel_btn)
        row.addWidget(self.save_btn)

        v.addLayout(row)

        self.status_label = QLabel("")
        v.addWidget(self.status_label)

        self.cancel_btn.clicked.connect(self.reject)
        self.save_btn.clicked.connect(self.accept)

    def get_data(self):
        return {
            "id": self.id_edit.text().strip(),
            "name": self.name_edit.text().strip(),
            "location": self.location_edit.text().strip(),
            "source": self.source_edit.text().strip(),
            "username": self.username_edit.text().strip(),
            "password": self.password_edit.text(),
            "online": self.online_check.isChecked(),
            "ai_enabled": self.ai_check.isChecked(),
            "heatmap_enabled": self.heatmap_check.isChecked(),
            "recording_enabled": self.recording_check.isChecked(),
            "fps": self.fps_spin.value(),
            "resolution": self.res_combo.currentText(),
        }


# ============================ CAMERAS DIALOG =========================
class CamerasDialog(QDialog):
    def __init__(self, settings_service, parent=None):
        super().__init__(parent)

        self.settings_service = settings_service

        self.setWindowTitle("Camera Management")
        self.setModal(True)
        self.resize(560, 460)

        v = QVBoxLayout(self)

        self.list = QListWidget()
        v.addWidget(self.list, 1)

        row = QHBoxLayout()

        self.add_btn = QPushButton("＋ Add")
        self.edit_btn = QPushButton("✏️ Edit")
        self.delete_btn = QPushButton("🗑 Delete")
        self.test_btn = QPushButton("🔌 Test")
        self.close_btn = QPushButton("Close")

        for b in (self.add_btn, self.edit_btn, self.delete_btn, self.test_btn):
            b.setObjectName("btnGhost")

        self.close_btn.setObjectName("btnPrimary")

        row.addWidget(self.add_btn)
        row.addWidget(self.edit_btn)
        row.addWidget(self.delete_btn)
        row.addWidget(self.test_btn)
        row.addStretch(1)
        row.addWidget(self.close_btn)

        v.addLayout(row)

        self.status_label = QLabel("")
        v.addWidget(self.status_label)

        self.add_btn.clicked.connect(self.add_camera)
        self.edit_btn.clicked.connect(self.edit_camera)
        self.delete_btn.clicked.connect(self.delete_camera)
        self.test_btn.clicked.connect(self.test_camera)
        self.close_btn.clicked.connect(self.accept)

        self.refresh()

    def refresh(self):
        self.list.clear()

        cams = self.settings_service.get_cameras()

        for cam in cams:
            online = "🟢" if cam.get("online") else "🔴"
            text = f"{online} {cam.get('id')} — {cam.get('name')} — {cam.get('source')}"

            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, cam)

            self.list.addItem(item)

    def selected_camera(self):
        item = self.list.currentItem()

        if item is None:
            return None

        return item.data(Qt.UserRole)

    def add_camera(self):
        dlg = CameraFormDialog(None, self)

        if dlg.exec() != QDialog.Accepted:
            return

        data = dlg.get_data()

        if dlg.test_btn.isEnabled():
            pass

        self.settings_service.add_camera(data)
        self.refresh()

    def edit_camera(self):
        cam = self.selected_camera()

        if cam is None:
            return

        dlg = CameraFormDialog(cam, self)

        if dlg.exec() != QDialog.Accepted:
            return

        data = dlg.get_data()
        data["id"] = cam.get("id")

        self.settings_service.update_camera(data)
        self.refresh()

    def delete_camera(self):
        cam = self.selected_camera()

        if cam is None:
            return

        reply = QMessageBox.question(
            self,
            "Delete camera",
            f"Delete {cam.get('id')}?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.settings_service.delete_camera(cam.get("id"))
            self.refresh()

    def test_camera(self):
        cam = self.selected_camera()

        if cam is None:
            dlg = CameraFormDialog(None, self)
            self.status_label.setText("Fill source and press Test inside form")
            return

        result = self.settings_service.test_camera(
            cam.get("source"),
            cam.get("username"),
            cam.get("password"),
            timeout=6,
        )

        if result.get("ok"):
            self.status_label.setText(
                f"✅ OK — {result.get('resolution')} — {result.get('latency_ms')} ms"
            )
        else:
            self.status_label.setText(f"❌ {result.get('message')}")


# ============================ PERSON FORM ============================
class PersonFormDialog(QDialog):
    def __init__(self, person=None, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Employee")
        self.setModal(True)
        self.setMinimumWidth(380)

        self.person = person or {}

        v = QVBoxLayout(self)
        f = QFormLayout()

        self.name_edit = QLineEdit(self.person.get("name", ""))
        self.dept_edit = QLineEdit(self.person.get("dept", self.person.get("department", "")))
        self.emp_edit = QLineEdit(self.person.get("emp_id", self.person.get("employee_id", "")))

        self.status_combo = QComboBox()
        self.status_combo.addItems(["Active", "Inactive"])
        self.status_combo.setCurrentText(self.person.get("status", "Active"))

        f.addRow("Name *", self.name_edit)
        f.addRow("Department", self.dept_edit)
        f.addRow("Employee ID", self.emp_edit)
        f.addRow("Status", self.status_combo)

        v.addLayout(f)

        row = QHBoxLayout()

        self.save_btn = QPushButton("Save")
        self.save_btn.setObjectName("btnPrimary")

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("btnGhost")

        row.addStretch(1)
        row.addWidget(self.cancel_btn)
        row.addWidget(self.save_btn)

        v.addLayout(row)

        self.cancel_btn.clicked.connect(self.reject)
        self.save_btn.clicked.connect(self.accept)

    def get_data(self):
        return {
            "name": self.name_edit.text().strip(),
            "department": self.dept_edit.text().strip(),
            "employee_id": self.emp_edit.text().strip(),
            "status": self.status_combo.currentText(),
        }


# ============================ UNKNOWN FACES ==========================
class UnknownFacesDialog(QDialog):
    def __init__(self, unknown_service, parent=None):
        super().__init__(parent)

        self.unknown_service = unknown_service

        self.setWindowTitle("Unknown Faces")
        self.setModal(True)
        self.resize(560, 460)

        v = QVBoxLayout(self)

        self.list = QListWidget()
        self.list.setIconSize(self.list.iconSize())
        v.addWidget(self.list, 1)

        row = QHBoxLayout()

        self.convert_btn = QPushButton("👤 Convert to Employee")
        self.delete_btn = QPushButton("🗑 Delete")
        self.close_btn = QPushButton("Close")

        self.convert_btn.setObjectName("btnPrimary")
        self.delete_btn.setObjectName("btnGhost")
        self.close_btn.setObjectName("btnGhost")

        row.addWidget(self.convert_btn)
        row.addWidget(self.delete_btn)
        row.addStretch(1)
        row.addWidget(self.close_btn)

        v.addLayout(row)

        self.convert_btn.clicked.connect(self.convert_selected)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.close_btn.clicked.connect(self.accept)

        self.refresh()

    def refresh(self):
        self.list.clear()

        rows = self.unknown_service.get_unknown_faces(200)

        for r in rows:
            text = (
                f"Unknown #{r.get('id')} | "
                f"{r.get('camera_id')} | "
                f"count {r.get('count')} | "
                f"{r.get('last_seen')}"
            )

            item = QListWidgetItem(text)

            pm = r.get("image_pm")

            if pm is not None and not pm.isNull():
                item.setIcon(pm.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation))

            item.setData(Qt.UserRole, r)
            self.list.addItem(item)

    def selected(self):
        item = self.list.currentItem()

        if item is None:
            return None

        return item.data(Qt.UserRole)

    def convert_selected(self):
        row = self.selected()

        if row is None:
            return

        dlg = PersonFormDialog(None, self)

        if dlg.exec() != QDialog.Accepted:
            return

        data = dlg.get_data()

        self.unknown_service.convert_unknown_to_person(
            unknown_id=row.get("id"),
            name=data.get("name"),
            department=data.get("department"),
            employee_id=data.get("employee_id"),
        )

        self.refresh()

    def delete_selected(self):
        row = self.selected()

        if row is None:
            return

        reply = QMessageBox.question(
            self,
            "Delete unknown face",
            "Delete this unknown face?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.unknown_service.delete_unknown_face(row.get("id"))
            self.refresh()