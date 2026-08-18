from __future__ import annotations

import os
from pathlib import Path

import yaml
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .sentinel_ui_base import C, Panel, label, make_button

ROOT = Path(__file__).resolve().parents[2]
CAMERA_CONFIG_PATH = ROOT / "config" / "cameras.yaml"
MAX_CONFIGURED_CAMERAS = 16
MAX_ACTIVE_CAMERAS = 6


class CameraConfigStore:
    def __init__(self, path: Path = CAMERA_CONFIG_PATH) -> None:
        self.path = Path(path)

    def load(self) -> dict:
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        raw.setdefault("cameras", [])
        return raw

    def save(self, payload: dict) -> None:
        cameras = list(payload.get("cameras") or [])
        if not cameras:
            raise ValueError("Kamida bitta camera config bo'lishi kerak")
        if len(cameras) > MAX_CONFIGURED_CAMERAS:
            raise ValueError(f"Maksimum {MAX_CONFIGURED_CAMERAS} ta camera config mumkin")

        ids: set[str] = set()
        active = 0
        cleaned: list[dict] = []
        for row in cameras:
            camera_id = str(row.get("id", "")).strip().upper()
            if not camera_id:
                raise ValueError("Camera ID bo'sh bo'lmasligi kerak")
            if camera_id in ids:
                raise ValueError(f"Duplicate Camera ID: {camera_id}")
            ids.add(camera_id)

            uri = str(row.get("uri", "")).strip()
            if not uri.lower().startswith("rtsp://"):
                raise ValueError(f"{camera_id}: RTSP URL rtsp:// bilan boshlanishi kerak")

            enabled = bool(row.get("enabled", True))
            active += int(enabled)
            cleaned.append(
                {
                    "id": camera_id,
                    "name": str(row.get("name", camera_id)).strip() or camera_id,
                    "room": str(row.get("room", "")).strip(),
                    "enabled": enabled,
                    "uri": uri,
                }
            )

        if active < 1:
            raise ValueError("Kamida bitta camera yoqilgan bo'lishi kerak")
        if active > MAX_ACTIVE_CAMERAS:
            raise ValueError(
                f"Monitoring 2x3 wall maksimum {MAX_ACTIVE_CAMERAS} ta active camera qo'llaydi"
            )

        # Canonical source schema. Codec/decoder and any environment-URI selector
        # are intentionally absent; DeepStream negotiates the RTSP stream itself.
        payload["cameras"] = cleaned
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)


class CameraEditorDialog(QDialog):
    """Dark in-app camera editor with only the fields a user should choose."""

    def __init__(
        self,
        camera: dict | None = None,
        *,
        suggested_id: str = "CAM-01",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.original = dict(camera or {})
        self.setModal(True)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMinimumWidth(520)

        shell = QFrame(self)
        shell.setObjectName("cameraDialog")
        shell.setStyleSheet(
            "QFrame#cameraDialog{background:#0d141c;border:1px solid #2a3a49;"
            "border-radius:10px;}"
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(shell)

        root = QVBoxLayout(shell)
        root.setContentsMargins(18, 14, 18, 18)
        root.setSpacing(14)

        header = QHBoxLayout()
        header_text = QVBoxLayout()
        header_text.setSpacing(2)
        header_text.addWidget(
            label("Yangi camera" if camera is None else "Camerani tahrirlash", "sectionTitle")
        )
        header_text.addWidget(label("RTSP source", "mono"))
        header.addLayout(header_text)
        header.addStretch()
        close = QToolButton()
        close.setText("×")
        close.setCursor(Qt.PointingHandCursor)
        close.setFixedSize(30, 30)
        close.clicked.connect(self.reject)
        header.addWidget(close)
        root.addLayout(header)

        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background:{C['border']};border:0;")
        root.addWidget(divider)

        self.camera_id = self._field(root, "Camera ID", str(self.original.get("id", suggested_id)))
        self.name = self._field(root, "Name", str(self.original.get("name", "")), "Masalan: Entrance 1")
        self.room = self._field(root, "Room", str(self.original.get("room", "")), "Masalan: Entrance")
        self.uri = self._field(
            root,
            "RTSP URL",
            str(self.original.get("uri", "rtsp://")),
            "rtsp://192.168.1.210:554/Streaming/Channels/101",
        )

        status_row = QHBoxLayout()
        status_text = QVBoxLayout()
        status_text.setSpacing(1)
        status_text.addWidget(label("Camera status", "sectionTitle"))
        status_text.addWidget(label("Monitoring wallga source qo'shilsin", "muted"))
        status_row.addLayout(status_text)
        status_row.addStretch()
        self.enabled = QCheckBox("Enabled")
        self.enabled.setChecked(bool(self.original.get("enabled", True)))
        status_row.addWidget(self.enabled)
        root.addLayout(status_row)

        note = QLabel(
            "Codec va decoder tanlanmaydi. DeepStream RTSP streamni avtomatik aniqlaydi; "
            "login/parol .env dagi SURVEILLANCE_RTSP_USERNAME va "
            "SURVEILLANCE_RTSP_PASSWORD dan olinadi."
        )
        note.setWordWrap(True)
        note.setStyleSheet(
            f"color:{C['muted']};background:#091017;border:1px solid {C['border']};"
            "border-radius:6px;padding:9px 10px;font-size:10px;"
        )
        root.addWidget(note)

        actions = QHBoxLayout()
        actions.addStretch()
        cancel = make_button("Cancel")
        cancel.clicked.connect(self.reject)
        actions.addWidget(cancel)
        save = make_button("Save camera", "primary")
        save.clicked.connect(self._validate_and_accept)
        actions.addWidget(save)
        root.addLayout(actions)

    @staticmethod
    def _field(layout: QVBoxLayout, title: str, value: str, placeholder: str = "") -> QLineEdit:
        layout.addWidget(label(title, "muted"))
        field = QLineEdit(value)
        if placeholder:
            field.setPlaceholderText(placeholder)
        field.setMinimumHeight(36)
        layout.addWidget(field)
        return field

    def _validate_and_accept(self) -> None:
        camera_id = self.camera_id.text().strip()
        uri = self.uri.text().strip()
        if not camera_id:
            QMessageBox.warning(self, "Camera", "Camera ID kiriting.")
            self.camera_id.setFocus()
            return
        if not uri.lower().startswith("rtsp://"):
            QMessageBox.warning(self, "Camera", "RTSP URL rtsp:// bilan boshlanishi kerak.")
            self.uri.setFocus()
            return
        self.accept()

    def camera_row(self) -> dict:
        camera_id = self.camera_id.text().strip().upper()
        return {
            "id": camera_id,
            "name": self.name.text().strip() or camera_id,
            "room": self.room.text().strip(),
            "enabled": self.enabled.isChecked(),
            "uri": self.uri.text().strip(),
        }


class CameraSettingsRow(Panel):
    toggled = Signal(int, bool)
    editRequested = Signal(int)
    deleteRequested = Signal(int)

    def __init__(self, index: int, camera: dict, parent=None) -> None:
        super().__init__(parent)
        self.index = int(index)
        self.camera = dict(camera)
        self.setMinimumHeight(94)

        row = QHBoxLayout(self)
        row.setContentsMargins(14, 11, 12, 11)
        row.setSpacing(12)

        enabled_state = bool(camera.get("enabled", True))
        dot = QLabel("●")
        dot.setFixedWidth(14)
        dot.setStyleSheet(
            f"color:{C['known'] if enabled_state else C['muted']};font-size:14px;"
        )
        row.addWidget(dot)

        info = QVBoxLayout()
        info.setSpacing(4)
        heading = QHBoxLayout()
        cid = QLabel(str(camera.get("id", "CAM")))
        cid.setStyleSheet("font-weight:800;font-size:12px;")
        heading.addWidget(cid)
        name = QLabel(str(camera.get("name", "")))
        name.setStyleSheet(f"color:{C['muted']};font-size:11px;")
        heading.addWidget(name)
        heading.addStretch()
        room = QLabel(str(camera.get("room", "") or "No room"))
        room.setStyleSheet(
            f"color:{C['blue']};background:#0c1823;border-radius:4px;padding:2px 6px;font-size:9px;"
        )
        heading.addWidget(room)
        info.addLayout(heading)

        uri = QLabel(str(camera.get("uri", "")))
        uri.setTextInteractionFlags(Qt.TextSelectableByMouse)
        uri.setStyleSheet(f"color:{C['muted']};font:10px 'DejaVu Sans Mono';")
        info.addWidget(uri)

        source_note = QLabel("DeepStream · auto codec · RTSP")
        source_note.setStyleSheet(f"color:#5f7080;font:9px 'DejaVu Sans Mono';")
        info.addWidget(source_note)
        row.addLayout(info, 1)

        enabled = QCheckBox("Enabled")
        enabled.setChecked(enabled_state)
        enabled.toggled.connect(lambda checked: self.toggled.emit(self.index, checked))
        row.addWidget(enabled)

        edit = make_button("Edit")
        edit.clicked.connect(lambda: self.editRequested.emit(self.index))
        row.addWidget(edit)

        delete = make_button("Delete")
        delete.setStyleSheet(
            f"QPushButton{{color:{C['offline']};border-color:#4a2529;}}"
            "QPushButton:hover{background:#281519;}"
        )
        delete.clicked.connect(lambda: self.deleteRequested.emit(self.index))
        row.addWidget(delete)


class SettingsPage(QWidget):
    applyRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pageRoot")
        self.store = CameraConfigStore()
        self.payload = self.store.load()
        self.dirty = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 18, 20, 20)
        outer.setSpacing(12)

        controls = QHBoxLayout()
        controls.addWidget(label("Camera sources", "sectionTitle"))
        controls.addStretch()
        self.apply_button = make_button("Apply changes", "primary")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self._apply)
        controls.addWidget(self.apply_button)
        add = make_button("+ Add camera")
        add.clicked.connect(self.add_camera)
        controls.addWidget(add)
        outer.addLayout(controls)

        self.banner = QLabel(
            "Camera ID, name, room va RTSP URL yetarli. Codec va decoder DeepStream tomonidan avtomatik tanlanadi."
        )
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet(
            f"color:{C['muted']};background:#0b1219;border:1px solid {C['border']};"
            "border-radius:6px;padding:9px 12px;"
        )
        outer.addWidget(self.banner)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        body.setObjectName("pageRoot")
        self.rows_layout = QVBoxLayout(body)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(9)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        self.refresh()

    def _next_camera_id(self) -> str:
        used = {str(row.get("id", "")) for row in self.payload.get("cameras", [])}
        for i in range(1, MAX_CONFIGURED_CAMERAS + 1):
            candidate = f"CAM-{i:02d}"
            if candidate not in used:
                return candidate
        return "CAM-NEW"

    def _set_dirty(self, message: str) -> None:
        self.dirty = True
        self.apply_button.setEnabled(True)
        self.banner.setText(
            message + " · Apply changes bosilganda live camera pipeline qayta ishga tushadi."
        )
        self.banner.setStyleSheet(
            "color:#f6d98a;background:#211c0e;border:1px solid #5b4921;"
            "border-radius:6px;padding:9px 12px;"
        )

    def _save(self, message: str) -> bool:
        try:
            self.store.save(self.payload)
        except Exception as exc:
            QMessageBox.critical(self, "Camera Settings", str(exc))
            self.payload = self.store.load()
            self.refresh()
            return False
        self.payload = self.store.load()
        self._set_dirty(message)
        self.refresh()
        return True

    def refresh(self) -> None:
        while self.rows_layout.count():
            item = self.rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        cameras = list(self.payload.get("cameras") or [])
        active = sum(1 for row in cameras if bool(row.get("enabled", True)))
        summary = QLabel(
            f"{len(cameras)} configured  ·  {active}/{MAX_ACTIVE_CAMERAS} active"
        )
        summary.setStyleSheet(f"color:{C['muted']};font:10px 'DejaVu Sans Mono';")
        self.rows_layout.addWidget(summary)

        for index, camera in enumerate(cameras):
            row = CameraSettingsRow(index, camera)
            row.toggled.connect(self.toggle_camera)
            row.editRequested.connect(self.edit_camera)
            row.deleteRequested.connect(self.delete_camera)
            self.rows_layout.addWidget(row)
        self.rows_layout.addStretch()

    def toggle_camera(self, index: int, enabled: bool) -> None:
        cameras = self.payload.get("cameras") or []
        if not (0 <= index < len(cameras)):
            return
        previous = bool(cameras[index].get("enabled", True))
        cameras[index]["enabled"] = bool(enabled)
        if not self._save(
            f"{cameras[index].get('id')} {'enabled' if enabled else 'disabled'}"
        ):
            cameras[index]["enabled"] = previous

    def edit_camera(self, index: int) -> None:
        cameras = self.payload.get("cameras") or []
        if not (0 <= index < len(cameras)):
            return
        dialog = CameraEditorDialog(cameras[index], parent=self)
        if dialog.exec() != QDialog.Accepted:
            return
        cameras[index] = dialog.camera_row()
        self._save(f"{cameras[index].get('id')} updated")

    def add_camera(self) -> None:
        cameras = self.payload.get("cameras") or []
        if len(cameras) >= MAX_CONFIGURED_CAMERAS:
            QMessageBox.warning(self, "Camera Settings", "Camera config limiti to'lgan")
            return
        dialog = CameraEditorDialog(
            None,
            suggested_id=self._next_camera_id(),
            parent=self,
        )
        if sum(1 for row in cameras if bool(row.get("enabled", True))) >= MAX_ACTIVE_CAMERAS:
            dialog.enabled.setChecked(False)
        if dialog.exec() != QDialog.Accepted:
            return
        cameras.append(dialog.camera_row())
        self._save(f"{cameras[-1].get('id')} added")

    def delete_camera(self, index: int) -> None:
        cameras = self.payload.get("cameras") or []
        if not (0 <= index < len(cameras)):
            return
        camera_id = str(cameras[index].get("id", "camera"))
        answer = QMessageBox.question(
            self,
            "Delete camera",
            f"{camera_id} ni configdan butunlay o'chiraymi?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        removed = cameras.pop(index)
        if not self._save(f"{camera_id} deleted"):
            cameras.insert(index, removed)

    def _apply(self) -> None:
        if self.dirty:
            self.applyRequested.emit()

    def mark_applied(self) -> None:
        self.dirty = False
        self.apply_button.setEnabled(False)
        self.payload = self.store.load()
        self.banner.setText(
            "Camera config applied. Monitoring pipeline yangi source list bilan qayta ishga tushdi."
        )
        self.banner.setStyleSheet(
            f"color:{C['known']};background:#0b1c19;border:1px solid #174238;"
            "border-radius:6px;padding:9px 12px;"
        )
        self.refresh()
