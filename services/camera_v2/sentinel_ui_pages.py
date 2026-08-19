from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

from .config import load_settings
from .sentinel_store import SentinelStore
from .sentinel_ui_base import C, Panel, ScrollPage, clear_layout, label, make_button, panel_layout


class PeoplePage(ScrollPage):
    def __init__(self):
        super().__init__()
        self.store = SentinelStore()
        top = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Ism yoki ID qidirish")
        self.search.setMaximumWidth(340)
        self.search.textChanged.connect(self.rebuild)
        top.addWidget(self.search)
        top.addStretch()
        self.layout.addLayout(top)

        self.grid = QGridLayout()
        self.grid.setSpacing(12)
        self.layout.addLayout(self.grid)
        self.layout.addStretch()
        self.rebuild()

    def refresh(self) -> None:
        self.rebuild()

    def rebuild(self) -> None:
        clear_layout(self.grid)
        query = self.search.text().strip().lower()
        people = self.store.list_people()
        if query:
            people = [
                person
                for person in people
                if query in str(person.get("name", "")).lower()
                or query in str(person.get("id", "")).lower()
            ]

        if not people:
            empty = label("Saqlangan profil topilmadi.", "muted")
            self.grid.addWidget(empty, 0, 0)
            return

        for index, person in enumerate(people):
            self.grid.addWidget(self._person_card(person), index // 3, index % 3)
        for column in range(3):
            self.grid.setColumnStretch(column, 1)

    def _person_card(self, person: dict) -> Panel:
        card = Panel()
        card.setMinimumWidth(280)
        layout = panel_layout(card, (12, 12, 12, 12), 8)

        top = QHBoxLayout()
        photo = QLabel()
        photo.setFixedSize(64, 64)
        photo.setAlignment(Qt.AlignCenter)
        photo.setStyleSheet(
            f"background:{C['field']};border:1px solid {C['border']};border-radius:6px;color:{C['muted']};"
        )
        profile_path = Path(str(person.get("profile_photo", "")))
        pixmap = QPixmap(str(profile_path)) if profile_path.is_file() else QPixmap()
        if pixmap.isNull():
            name = str(person.get("name", "?"))
            photo.setText("".join(part[:1] for part in name.split()[:2]).upper() or "?")
        else:
            photo.setPixmap(
                pixmap.scaled(photo.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            )
        top.addWidget(photo)

        info = QVBoxLayout()
        info.addWidget(label(str(person.get("name", "")), "sectionTitle"))
        info.addWidget(label(str(person.get("id", "")), "mono"))
        details = " · ".join(
            value
            for value in (
                str(person.get("role", "")).strip(),
                str(person.get("department", "")).strip(),
            )
            if value
        )
        if details:
            info.addWidget(label(details, "muted"))
        info.addStretch()
        top.addLayout(info, 1)
        layout.addLayout(top)

        notes = str(person.get("notes", "")).strip()
        if notes:
            note = label(notes, "muted")
            note.setWordWrap(True)
            layout.addWidget(note)

        actions = QHBoxLayout()
        actions.addStretch()
        remove = make_button("Deactivate", "ghost")
        remove.clicked.connect(lambda _=False, pid=str(person.get("id", "")): self._deactivate(pid))
        actions.addWidget(remove)
        layout.addLayout(actions)
        return card

    def _deactivate(self, person_id: str) -> None:
        answer = QMessageBox.question(
            self,
            "Profilni o'chirish",
            f"{person_id} profilini active ro'yxatdan olib tashlaysizmi?",
        )
        if answer != QMessageBox.Yes:
            return
        self.store.deactivate_person(person_id)
        self.rebuild()


class EventsPage(ScrollPage):
    TYPE_LABELS = {
        "entry": "Kirish",
        "exit": "Chiqish",
        "transition": "Xonalar orasida",
        "unknown": "Unknown",
        "restricted": "Restricted zone",
        "camera_offline": "Camera offline",
        "service": "Service",
    }

    def __init__(self):
        super().__init__()
        self.store = SentinelStore()

        filters = QHBoxLayout()
        self.kind = QComboBox()
        self.kind.addItem("Barcha turlar", "all")
        for key, title in self.TYPE_LABELS.items():
            self.kind.addItem(title, key)
        self.kind.currentIndexChanged.connect(self.rebuild)
        filters.addWidget(self.kind)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Person ID, camera yoki room")
        self.search.textChanged.connect(self.rebuild)
        filters.addWidget(self.search, 1)
        self.layout.addLayout(filters)

        self.rows = QVBoxLayout()
        self.rows.setSpacing(8)
        self.layout.addLayout(self.rows)
        self.layout.addStretch()
        self.rebuild()

    def refresh(self) -> None:
        self.rebuild()

    @staticmethod
    def _fmt(timestamp: float) -> str:
        try:
            return datetime.fromtimestamp(float(timestamp)).astimezone().strftime("%d.%m.%Y, %H:%M:%S")
        except Exception:
            return "—"

    def rebuild(self) -> None:
        clear_layout(self.rows)
        kind = self.kind.currentData()
        query = self.search.text().strip().lower()
        events = self.store.list_events(limit=500)
        filtered = []
        for event in events:
            if kind != "all" and str(event.get("event_type", "")) != kind:
                continue
            haystack = " ".join(
                str(event.get(key, ""))
                for key in ("person_id", "person_name", "local_id", "camera_id", "room")
            ).lower()
            if query and query not in haystack:
                continue
            filtered.append(event)

        if not filtered:
            self.rows.addWidget(label("Real event yozuvlari hali yo'q.", "muted"))
            return

        for event in filtered:
            card = Panel()
            layout = QHBoxLayout(card)
            layout.setContentsMargins(12, 10, 12, 10)
            layout.setSpacing(12)

            snapshot = QLabel()
            snapshot.setFixedSize(72, 54)
            snapshot.setAlignment(Qt.AlignCenter)
            snapshot.setStyleSheet(
                f"background:{C['field']};border:1px solid {C['border']};border-radius:4px;color:{C['muted']};"
            )
            snapshot_path = Path(str(event.get("snapshot_path", "")))
            pixmap = QPixmap(str(snapshot_path)) if snapshot_path.is_file() else QPixmap()
            if pixmap.isNull():
                snapshot.setText("No image")
            else:
                snapshot.setPixmap(
                    pixmap.scaled(snapshot.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                )
            layout.addWidget(snapshot)

            info = QVBoxLayout()
            event_type = str(event.get("event_type", ""))
            title = self.TYPE_LABELS.get(event_type, event_type or "Event")
            info.addWidget(label(f"{title} · {self._fmt(event.get('created_at', 0))}", "sectionTitle"))
            identity = str(event.get("person_name") or event.get("person_id") or event.get("local_id") or "Unknown")
            info.addWidget(label(identity))
            location = " · ".join(
                value
                for value in (
                    str(event.get("camera_id", "")).strip(),
                    str(event.get("room", "")).strip(),
                )
                if value
            )
            if location:
                info.addWidget(label(location, "mono"))
            layout.addLayout(info, 1)
            self.rows.addWidget(card)


class RoomsPage(ScrollPage):
    def __init__(self):
        super().__init__()
        self.grid = QGridLayout()
        self.grid.setSpacing(16)
        self.layout.addLayout(self.grid)
        self.layout.addStretch()
        self.refresh()

    def refresh(self) -> None:
        clear_layout(self.grid)
        settings = load_settings()
        rooms: dict[str, list] = {}
        for camera in settings.cameras:
            room = str(camera.room or "Unassigned").strip() or "Unassigned"
            rooms.setdefault(room, []).append(camera)

        for index, (room_name, cameras) in enumerate(sorted(rooms.items())):
            card = Panel()
            layout = panel_layout(card)
            header = QHBoxLayout()
            header.addWidget(label(room_name, "sectionTitle"))
            header.addStretch()
            header.addWidget(label(str(len(cameras)), "mono", C["primary"]))
            layout.addLayout(header)
            layout.addWidget(label("Configured cameras", "eyebrow"))
            for camera in cameras:
                row = QHBoxLayout()
                row.addWidget(label(camera.camera_id, "mono"))
                row.addWidget(label(camera.name))
                row.addStretch()
                row.addWidget(label("enabled", "mono", C["known"]))
                layout.addLayout(row)
            layout.addStretch()
            self.grid.addWidget(card, index // 3, index % 3)

        if not rooms:
            self.grid.addWidget(label("Camera room config topilmadi.", "muted"), 0, 0)
        for column in range(3):
            self.grid.setColumnStretch(column, 1)
