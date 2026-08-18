from __future__ import annotations

import os
from pathlib import Path

import yaml
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QScrollArea,
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
            camera_id = str(row.get("id", "")).strip()
            if not camera_id:
                raise ValueError("Camera ID bo'sh bo'lmasligi kerak")
            if camera_id in ids:
                raise ValueError(f"Duplicate Camera ID: {camera_id}")
            ids.add(camera_id)

            uri = str(row.get("uri", "")).strip()
            if not uri.startswith("rtsp://"):
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

        # Canonical camera schema: no codec and no environment-URI fields.
        payload["cameras"] = cleaned
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)


class CameraEditorDialog(QDialog):
    def __init__(
        self,
        camera: dict | None = None,
        *,
        suggested_id: str = "CAM-01",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Camera qo'shish" if camera is None else "Camerani tahrirlash")
        self.setMinimumWidth(540)
        self.original = dict(camera or {})

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        form = QFormLayout()
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(11)

        self.camera_id = QLineEdit(str(self.original.get("id", suggested_id)))
        self.name = QLineEdit(str(self.original.get("name", "")))
        self.room = QLineEdit(str(self.original.get("room", "")))
        self.uri = QLineEdit(str(self.original.get("uri", "rtsp://")))
        self.enabled = QCheckBox("Camera active")
        self.enabled.setChecked(bool(self.original.get("enabled", True)))

        form.addRow("Camera ID", self.camera_id)
        form.addRow("Name", self.name)
        form.addRow("Room", self.room)
        form.addRow("RTSP URL", self.uri)
        form.addRow("Status", self.enabled)
        root.addLayout(form)

        note = QLabel(
            "Faqat RTSP URL yetarli. Stream formati va decoder DeepStream tomonidan "
            "avtomatik aniqlanadi. Login/parol .env dagi "
            "SURVEILLANCE_RTSP_USERNAME / SURVEILLANCE_RTSP_PASSWORD dan olinadi."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{C['muted']};font-size:10px;")
        root.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def camera_row(self) -> dict:
        return {
            "id": self.camera_id.text().strip(),
            "name": self.name.text().strip() or self.camera_id.text().strip(),
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
        self.setMinimumHeight(90)

        row = QHBoxLayout(self)
        row.setContentsMargins(14, 11, 12, 11)
        row.setSpacing(12)

        dot = QLabel("●")
        dot.setStyleSheet(
            f"color:{C['known'] if camera.get('enabled', True) else C['muted']};font-size:14px;"
        )
        row.addWidget(dot)

        info = QVBoxLayout()
        info.setSpacing(3)
        heading = QHBoxLayout()
        cid = QLabel(str(camera.get("id", "CAM")))
        cid.setStyleSheet("font-weight:800;font-size:13px;")
        heading.addWidget(cid)
        name = QLabel(str(camera.get("name", "")))
        name.setStyleSheet(f"color:{C['muted']};")
        heading.addWidget(name)
        heading.addStretch()
        info.addLayout(heading)

        uri = QLabel(str(camera.get("uri", "")))
        uri.setTextInteractionFlags(Qt.TextSelectableByMouse)
        uri.setStyleSheet(f"color:{C['muted']};font:10px 'DejaVu Sans Mono';")
        info.addWidget(uri)

        room = QLabel(str(camera.get("room", "") or "No room"))
        room.setStyleSheet(f"color:{C['muted']};font-size:10px;")
        info.addWidget(room)
        row.addLayout(info, 1)

        enabled = QCheckBox("Enabled")
        enabled.setChecked(bool(camera.get("enabled", True)))
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
        outer.setContentsMargins(24, 20, 24, 22)
        outer.setSpacing(14)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(label("Camera Settings", "title"))
        titles.addWidget(
            label(
                "RTSP cameras · enable/disable · edit · delete · add",
                "subtitle",
            )
        )
        header.addLayout(titles)
        header.addStretch()

        self.apply_button = make_button("Apply cameras", "primary")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self._apply)
        header.addWidget(self.apply_button)

        add = make_button("+ Add camera")
        add.clicked.connect(self.add_camera)
        header.addWidget(add)
        outer.addLayout(header)

        self.banner = QLabel(
            "Camera ID, name, room va RTSP URL yetarli. Qolgan stream parametrlarini pipeline avtomatik aniqlaydi."
        )
        self.banner.setStyleSheet(
            f"color:{C['muted']};background:#0b1219;border:1px solid {C['border']};"
            "border-radius:6px;padding:9px 12px;"
        )
        outer.addWidget(self.banner)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
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
            message + "  ·  Apply cameras bosilganda live pipeline restart bo'ladi."
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
            "Camera config applied. Monitoring pipeline yangi config bilan qayta ishga tushadi."
        )
        self.banner.setStyleSheet(
            f"color:{C['known']};background:#0b1c19;border:1px solid #174238;"
            "border-radius:6px;padding:9px 12px;"
        )
        self.refresh()
