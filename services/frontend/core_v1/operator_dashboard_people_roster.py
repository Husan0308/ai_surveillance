from __future__ import annotations

import base64
from datetime import datetime
import uuid

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import operator_dashboard_face as face
from . import operator_dashboard_face_cuda as cuda


TH = face.TH
CAMERA_SPECS = face.CAMERA_SPECS
MAX_WORKERS = 10
ENROLLMENT_SAMPLES = 10


def _fmt_datetime(value) -> str:
    if not value:
        return "Never"
    text = str(value)
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%d %b %Y · %H:%M")
    except Exception:
        return text[:19]


def _active_people(state: dict) -> set[str]:
    active: set[str] = set()
    tracks = state.get("tracks") or {}
    for camera in (tracks.get("cameras") or {}).values():
        for row in camera.get("tracks") or []:
            person_id = str(row.get("person_id") or "").strip()
            if person_id:
                active.add(person_id)
    return active


def _clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child = item.layout()
        if widget is not None:
            widget.deleteLater()
        elif child is not None:
            _clear_layout(child)


class WorkerCard(QFrame):
    clicked = Signal()

    def __init__(
        self,
        slot: int,
        person: dict | None,
        avatar: QPixmap | None,
        present: bool,
        selected: bool,
    ):
        super().__init__()
        self.person = person
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumWidth(138)
        self.setMinimumHeight(190)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        border = TH.ACCENT if selected else TH.BORDER
        background = "#18212b" if selected else TH.CARD2
        self.setStyleSheet(
            f"QFrame{{background:{background};border:1px solid {border};border-radius:12px;}}"
            f"QFrame:hover{{border:1px solid {TH.ACC2};background:#1b2530;}}"
            "QLabel{border:none;background:transparent;}"
        )
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 11, 12, 11)
        v.setSpacing(5)

        top = QHBoxLayout()
        slot_label = QLabel(f"#{slot:02d}")
        slot_label.setStyleSheet(f"color:{TH.FAINT};font-size:8px;font-weight:800;")
        top.addWidget(slot_label)
        top.addStretch(1)
        if person:
            status = QLabel("● LIVE" if present else "● SAVED")
            status.setStyleSheet(
                f"color:{TH.OK if present else TH.DIM};font-size:8px;font-weight:800;"
            )
            top.addWidget(status)
        v.addLayout(top)

        photo = QLabel()
        photo.setFixedSize(82, 82)
        photo.setAlignment(Qt.AlignCenter)
        photo.setStyleSheet(
            f"background:#0e1318;border:2px solid {TH.BORDER};border-radius:41px;"
            f"color:{TH.FAINT};font-size:27px;font-weight:800;"
        )
        if person:
            if avatar is not None and not avatar.isNull():
                photo.setPixmap(
                    avatar.scaled(82, 82, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                )
            else:
                initials = "".join(
                    part[:1].upper() for part in str(person.get("name") or "?").split()[:2]
                )
                photo.setText(initials or "?")
        else:
            photo.setText("+")
            photo.setStyleSheet(
                f"background:#111820;border:2px dashed {TH.BORDER};border-radius:41px;"
                f"color:{TH.ACC2};font-size:32px;font-weight:500;"
            )
        photo_wrap = QHBoxLayout()
        photo_wrap.addStretch(1)
        photo_wrap.addWidget(photo)
        photo_wrap.addStretch(1)
        v.addLayout(photo_wrap)

        if person:
            name = QLabel(str(person.get("name") or "Unnamed"))
            name.setAlignment(Qt.AlignCenter)
            name.setWordWrap(True)
            name.setStyleSheet("color:#f3f7fb;font-size:11px;font-weight:800;")
            role = QLabel(str(person.get("job_title") or person.get("department") or "Worker"))
            role.setAlignment(Qt.AlignCenter)
            role.setWordWrap(True)
            role.setStyleSheet(f"color:{TH.DIM};font-size:9px;")
            samples = int(person.get("samples") or 0)
            face_state = QLabel(f"Face {min(samples, ENROLLMENT_SAMPLES)}/{ENROLLMENT_SAMPLES}")
            face_state.setAlignment(Qt.AlignCenter)
            face_state.setStyleSheet(f"color:{TH.OK if samples >= ENROLLMENT_SAMPLES else TH.WARN};font-size:8px;font-weight:700;")
            v.addWidget(name)
            v.addWidget(role)
            v.addWidget(face_state)
        else:
            title = QLabel("Add worker")
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet(f"color:{TH.ACC2};font-size:10px;font-weight:800;")
            sub = QLabel("Empty profile slot")
            sub.setAlignment(Qt.AlignCenter)
            sub.setStyleSheet(f"color:{TH.FAINT};font-size:8px;")
            v.addWidget(title)
            v.addWidget(sub)
        v.addStretch(1)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class WorkerEnrollmentDialog(QDialog):
    PROMPTS = [
        "Look straight at the camera",
        "Turn slightly LEFT",
        "Turn slightly RIGHT",
        "Look slightly UP",
        "Look slightly DOWN",
        "Straight again",
        "Left, a little closer",
        "Right, a little closer",
        "Neutral front",
        "Final straight sample",
    ]

    def __init__(self, hub):
        super().__init__(hub)
        self.hub = hub
        self.setWindowTitle("Add Worker · Face Enrollment")
        self.resize(1120, 720)
        self.setMinimumSize(940, 620)
        self.setModal(True)
        self.setStyleSheet(face.base.STYLE)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("ADD WORKER")
        title.setStyleSheet("font-size:17px;font-weight:900;color:white;")
        subtitle = QLabel("Profile + 10 quality-gated face samples")
        subtitle.setStyleSheet(f"color:{TH.DIM};font-size:10px;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch(1)
        close = QPushButton("✕ Close")
        close.setObjectName("btnGhost")
        close.clicked.connect(self.reject)
        header.addWidget(close)
        root.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(12)

        camera_panel = QFrame()
        camera_panel.setObjectName("camCard")
        camera_layout = QVBoxLayout(camera_panel)
        camera_layout.setContentsMargins(10, 10, 10, 10)
        camera_layout.setSpacing(8)

        selectors = QHBoxLayout()
        self.camera = QComboBox()
        for camera_id, _name, location in CAMERA_SPECS:
            self.camera.addItem(f"{camera_id} · {location}", camera_id)
        self.camera.currentIndexChanged.connect(self._change_camera)
        self.track = QComboBox()
        self.track.setMinimumWidth(245)
        selectors.addWidget(self.camera)
        selectors.addWidget(self.track, 1)
        camera_layout.addLayout(selectors)

        self.surface = face.base.CameraSurface(hub.feeds[CAMERA_SPECS[0][0]])
        camera_layout.addWidget(self.surface, 1)

        self.status = QLabel("Checking Face engine…")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color:{TH.WARN};font-size:10px;padding:5px;")
        camera_layout.addWidget(self.status)
        content.addWidget(camera_panel, 1)

        form_panel = QFrame()
        form_panel.setObjectName("chartCard")
        form_panel.setFixedWidth(395)
        form = QVBoxLayout(form_panel)
        form.setContentsMargins(17, 16, 17, 16)
        form.setSpacing(8)

        profile_title = QLabel("WORKER PROFILE")
        profile_title.setStyleSheet(
            f"color:{TH.ACC2};font-size:9px;font-weight:900;letter-spacing:1.5px;"
        )
        form.addWidget(profile_title)

        name_label = QLabel("Full Name  *")
        name_label.setStyleSheet(f"color:{TH.TXT};font-size:9px;font-weight:700;")
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. Husan Karimov")
        self.name.setClearButtonEnabled(True)
        role_label = QLabel("Job / Role  *")
        role_label.setStyleSheet(f"color:{TH.TXT};font-size:9px;font-weight:700;")
        self.role = QLineEdit()
        self.role.setPlaceholderText("e.g. AI Engineer, Security Officer")
        self.role.setClearButtonEnabled(True)
        form.addWidget(name_label)
        form.addWidget(self.name)
        form.addWidget(role_label)
        form.addWidget(self.role)

        tip = QLabel(
            "Enrollment tips\n"
            "• Keep one worker in the selected track\n"
            "• Face the camera with even light\n"
            "• Follow the angle prompt after each accepted shot"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet(
            f"color:{TH.DIM};background:#121820;border:1px solid {TH.BORDER};"
            "border-radius:7px;padding:8px;font-size:8.5px;"
        )
        form.addWidget(tip)

        self.progress_label = QLabel(f"Face samples  0 / {ENROLLMENT_SAMPLES}")
        self.progress_label.setStyleSheet(f"color:{TH.DIM};font-size:9px;font-weight:800;")
        self.progress = QProgressBar()
        self.progress.setRange(0, ENROLLMENT_SAMPLES)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        form.addWidget(self.progress_label)
        form.addWidget(self.progress)

        self.thumbs: list[QLabel] = []
        thumb_grid = QGridLayout()
        thumb_grid.setSpacing(6)
        for index in range(ENROLLMENT_SAMPLES):
            box = QLabel(str(index + 1))
            box.setFixedSize(60, 60)
            box.setAlignment(Qt.AlignCenter)
            box.setStyleSheet(
                f"color:{TH.FAINT};background:#11161c;border:1px dashed {TH.BORDER};border-radius:7px;"
            )
            thumb_grid.addWidget(box, index // 5, index % 5)
            self.thumbs.append(box)
        form.addLayout(thumb_grid)

        self.prompt = QLabel(self.PROMPTS[0])
        self.prompt.setWordWrap(True)
        self.prompt.setStyleSheet(
            f"color:{TH.ACC2};font-size:10px;font-weight:800;padding:5px 0;"
        )
        form.addWidget(self.prompt)
        form.addStretch(1)

        self.capture = QPushButton("📸 Start 10-sample capture")
        self.capture.setObjectName("btnGhost")
        self.capture.setEnabled(False)
        self.capture.clicked.connect(self._start_capture)
        self.save = QPushButton("✓ Create Worker Profile")
        self.save.setObjectName("btnPrimary")
        self.save.setEnabled(False)
        self.save.clicked.connect(self._commit)
        form.addWidget(self.capture)
        form.addWidget(self.save)
        content.addWidget(form_panel)
        root.addLayout(content, 1)

        self.tokens: list[str] = []
        self.session_id = uuid.uuid4().hex
        self._attempts = 0
        self._last_state: dict = {}

        self.capture_timer = QTimer(self)
        self.capture_timer.setInterval(1150)
        self.capture_timer.timeout.connect(self._capture_one)
        self.state_timer = QTimer(self)
        self.state_timer.setInterval(300)
        self.state_timer.timeout.connect(self._refresh_state)
        self.state_timer.start()

        self.name.textChanged.connect(self._update_actions)
        self.role.textChanged.connect(self._update_actions)
        self.track.currentIndexChanged.connect(self._update_actions)
        self._refresh_state()

    def closeEvent(self, event):
        self.capture_timer.stop()
        self.state_timer.stop()
        super().closeEvent(event)

    def reject(self):
        self.capture_timer.stop()
        self.state_timer.stop()
        super().reject()

    def accept(self):
        self.capture_timer.stop()
        self.state_timer.stop()
        super().accept()

    def _change_camera(self):
        camera_id = self.camera.currentData()
        if camera_id in self.hub.feeds:
            self.surface.set_feed(self.hub.feeds[camera_id])
        self._update_tracks()
        self._update_actions()

    def _refresh_state(self):
        state, _events = self.hub.state_reader.snapshot()
        self._last_state = state
        self._update_tracks()
        faces = state.get("faces") or {}
        metrics = faces.get("metrics") or {}
        ready = bool(faces.get("ready") or metrics.get("ready"))
        error = str(metrics.get("last_error") or "")
        has_track = self.track.currentData() is not None
        if not ready:
            self.status.setText(f"Face engine not ready · {error or 'loading model'}")
            self.status.setStyleSheet(f"color:{TH.WARN};font-size:10px;padding:5px;")
        elif not has_track:
            self.status.setText("Face ready · choose a camera where the worker is visible")
            self.status.setStyleSheet(f"color:{TH.WARN};font-size:10px;padding:5px;")
        elif not self.capture_timer.isActive() and len(self.tokens) < ENROLLMENT_SAMPLES:
            self.status.setText("Face ready · selected RAW track will be used for enrollment")
            self.status.setStyleSheet(f"color:{TH.OK};font-size:10px;padding:5px;")
        self._update_actions()

    def _update_tracks(self):
        camera_id = self.camera.currentData()
        current = self.track.currentData()
        camera = ((self._last_state.get("tracks") or {}).get("cameras") or {}).get(camera_id, {})
        rows = list(camera.get("tracks") or [])
        choices = []
        for row in rows:
            track_id = int(row.get("track_id") or 0)
            if track_id <= 0:
                continue
            local = str(row.get("display_id") or track_id)
            gid = str(row.get("global_id") or "")
            known_name = str(row.get("name") or "Unknown")
            label = local + (f" / {gid}" if gid else "") + f" · {known_name}"
            choices.append((label, track_id))
        ids = [value for _label, value in choices]
        existing = [self.track.itemData(index) for index in range(self.track.count())]
        if existing == ids:
            return
        self.track.blockSignals(True)
        self.track.clear()
        for label, value in choices:
            self.track.addItem(label, value)
        if current in ids:
            self.track.setCurrentIndex(ids.index(current))
        self.track.blockSignals(False)

    def _face_ready(self) -> bool:
        faces = self._last_state.get("faces") or {}
        metrics = faces.get("metrics") or {}
        return bool(faces.get("ready") or metrics.get("ready"))

    def _profile_valid(self) -> bool:
        return bool(self.name.text().strip() and self.role.text().strip())

    def _update_actions(self):
        ready = self._face_ready()
        has_track = self.track.currentData() is not None
        busy = self.capture_timer.isActive()
        self.capture.setEnabled(
            ready and has_track and self._profile_valid() and not busy and len(self.tokens) < ENROLLMENT_SAMPLES
        )
        self.save.setEnabled(
            len(self.tokens) >= ENROLLMENT_SAMPLES and self._profile_valid() and not busy
        )

    def _reset_capture(self):
        self.capture_timer.stop()
        self.tokens = []
        self.session_id = uuid.uuid4().hex
        self._attempts = 0
        self.progress.setValue(0)
        self.progress_label.setText(f"Face samples  0 / {ENROLLMENT_SAMPLES}")
        self.prompt.setText(self.PROMPTS[0])
        for index, box in enumerate(self.thumbs):
            box.clear()
            box.setText(str(index + 1))
            box.setStyleSheet(
                f"color:{TH.FAINT};background:#11161c;border:1px dashed {TH.BORDER};border-radius:7px;"
            )

    def _start_capture(self):
        if not self._profile_valid():
            self.hub.toast("⚠ Enter Full Name and Job / Role first")
            return
        if self.track.currentData() is None:
            self.hub.toast("⚠ Select a visible worker track")
            return
        self._reset_capture()
        self.capture_timer.start()
        self.status.setText("Capturing · " + self.PROMPTS[0])
        self._update_actions()
        self._capture_one()

    def _capture_one(self):
        if len(self.tokens) >= ENROLLMENT_SAMPLES:
            self.capture_timer.stop()
            self.status.setText("✅ 10 quality samples accepted · ready to create worker")
            self.status.setStyleSheet(f"color:{TH.OK};font-size:10px;font-weight:800;padding:5px;")
            self._update_actions()
            return
        camera_id = str(self.camera.currentData() or "")
        track_id = self.track.currentData()
        if not camera_id or track_id is None:
            self.capture_timer.stop()
            self.status.setText("Selected worker track disappeared")
            self._update_actions()
            return

        self._attempts += 1
        try:
            sample = face._json_request(
                "POST",
                f"/faces/enrollment/sample/{camera_id}/{int(track_id)}?session_id={self.session_id}",
                timeout=8.0,
            )
        except Exception as exc:
            self.status.setText(f"Waiting for a better face · {exc}")
            self.status.setStyleSheet(f"color:{TH.WARN};font-size:9px;padding:5px;")
            if self._attempts >= 40:
                self.capture_timer.stop()
                self.status.setText("Capture stopped · move closer / improve lighting and retry")
                self._update_actions()
            return

        token = str(sample.get("token") or "")
        if not token:
            return
        self.tokens.append(token)
        index = len(self.tokens) - 1
        raw = base64.b64decode(sample.get("thumbnail_jpeg_b64") or "")
        pixmap = QPixmap()
        if raw and pixmap.loadFromData(raw, "JPG"):
            self.thumbs[index].setPixmap(
                pixmap.scaled(60, 60, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            )
        self.thumbs[index].setStyleSheet(
            f"background:#11161c;border:2px solid {TH.OK};border-radius:7px;"
        )
        count = len(self.tokens)
        quality = float(sample.get("quality") or 0.0)
        self.progress.setValue(count)
        self.progress_label.setText(
            f"Face samples  {count} / {ENROLLMENT_SAMPLES} · quality {quality:.2f}"
        )
        if count < ENROLLMENT_SAMPLES:
            self.prompt.setText(self.PROMPTS[count])
            self.status.setText("Accepted ✅ · " + self.PROMPTS[count])
        else:
            self._capture_one()

    def _commit(self):
        name = self.name.text().strip()
        role = self.role.text().strip()
        if not name or not role:
            self.hub.toast("⚠ Full Name and Job / Role are required")
            return
        if len(self.tokens) < ENROLLMENT_SAMPLES:
            self.hub.toast("⚠ Capture all 10 accepted face samples first")
            return
        try:
            person = face._json_request(
                "POST",
                "/faces/enrollment/commit",
                {
                    "name": name,
                    "department": role,
                    "employee_id": "",
                    "sample_tokens": self.tokens,
                },
                timeout=12.0,
            )
        except Exception as exc:
            self.hub.toast(f"⚠ Enrollment failed · {exc}")
            return
        self.hub.toast(f"✅ Worker created · {person.get('name', name)}")
        self.accept()


class WorkerRosterPage(face.base.Page):
    def __init__(self, hub):
        super().__init__()
        self.hub = hub
        row = self.title_row("Workers", "10-person face roster")
        self.capacity = QLabel("0 / 10 enrolled")
        self.capacity.setStyleSheet(f"color:{TH.DIM};font-size:10px;font-weight:800;")
        row.addWidget(self.capacity)
        self.add_button = QPushButton("＋ Add Worker")
        self.add_button.setObjectName("btnPrimary")
        self.add_button.clicked.connect(self.open_enrollment)
        row.addWidget(self.add_button)

        hint = QLabel(
            "Each slot is one worker identity. Click a worker to open the profile; empty slots start enrollment."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{TH.DIM};font-size:9px;padding:0 2px 6px 2px;")
        self.v.addWidget(hint)

        body = QHBoxLayout()
        body.setSpacing(12)

        roster_frame = QFrame()
        roster_frame.setObjectName("chartCard")
        roster_layout = QVBoxLayout(roster_frame)
        roster_layout.setContentsMargins(12, 12, 12, 12)
        roster_layout.setSpacing(8)
        roster_head = QHBoxLayout()
        roster_title = QLabel("WORKER ROSTER")
        roster_title.setStyleSheet(
            f"color:{TH.ACC2};font-size:9px;font-weight:900;letter-spacing:1.4px;"
        )
        self.engine = QLabel("● FACE OFFLINE")
        self.engine.setStyleSheet(f"color:{TH.WARN};font-size:8px;font-weight:800;")
        roster_head.addWidget(roster_title)
        roster_head.addStretch(1)
        roster_head.addWidget(self.engine)
        roster_layout.addLayout(roster_head)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(8)
        self.grid.setVerticalSpacing(8)
        for col in range(5):
            self.grid.setColumnStretch(col, 1)
        scroll.setWidget(self.grid_host)
        roster_layout.addWidget(scroll, 1)
        body.addWidget(roster_frame, 1)

        self.profile = QFrame()
        self.profile.setObjectName("chartCard")
        self.profile.setFixedWidth(300)
        pv = QVBoxLayout(self.profile)
        pv.setContentsMargins(16, 16, 16, 16)
        pv.setSpacing(8)
        ptitle = QLabel("WORKER PROFILE")
        ptitle.setStyleSheet(
            f"color:{TH.ACC2};font-size:9px;font-weight:900;letter-spacing:1.4px;"
        )
        pv.addWidget(ptitle)

        self.profile_avatar = QLabel("—")
        self.profile_avatar.setFixedSize(116, 116)
        self.profile_avatar.setAlignment(Qt.AlignCenter)
        self.profile_avatar.setStyleSheet(
            f"background:#10161c;border:2px solid {TH.BORDER};border-radius:58px;"
            f"color:{TH.FAINT};font-size:30px;font-weight:900;"
        )
        avatar_row = QHBoxLayout()
        avatar_row.addStretch(1)
        avatar_row.addWidget(self.profile_avatar)
        avatar_row.addStretch(1)
        pv.addLayout(avatar_row)

        self.profile_name = QLabel("Select a worker")
        self.profile_name.setAlignment(Qt.AlignCenter)
        self.profile_name.setWordWrap(True)
        self.profile_name.setStyleSheet("color:white;font-size:15px;font-weight:900;")
        self.profile_role = QLabel("Worker profile details")
        self.profile_role.setAlignment(Qt.AlignCenter)
        self.profile_role.setWordWrap(True)
        self.profile_role.setStyleSheet(f"color:{TH.DIM};font-size:10px;")
        self.profile_status = QLabel("● —")
        self.profile_status.setAlignment(Qt.AlignCenter)
        self.profile_status.setStyleSheet(f"color:{TH.DIM};font-size:9px;font-weight:800;")
        pv.addWidget(self.profile_name)
        pv.addWidget(self.profile_role)
        pv.addWidget(self.profile_status)

        self.profile_meta = QLabel("Select one of the ten roster slots.")
        self.profile_meta.setWordWrap(True)
        self.profile_meta.setStyleSheet(
            f"color:{TH.TXT};background:#111820;border:1px solid {TH.BORDER};"
            "border-radius:8px;padding:10px;font-size:9px;line-height:1.5;"
        )
        pv.addWidget(self.profile_meta)
        pv.addStretch(1)

        self.delete_button = QPushButton("🗑 Delete Worker")
        self.delete_button.setObjectName("btnGhost")
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self.delete_selected)
        pv.addWidget(self.delete_button)
        body.addWidget(self.profile)
        self.v.addLayout(body, 1)

        self._state: dict = {}
        self._people: list[dict] = []
        self._avatars: dict[str, QPixmap | None] = {}
        self._selected_id = ""
        self._last_active: set[str] = set()

    def refresh(self, state: dict):
        self._state = state
        faces = state.get("faces") or {}
        metrics = faces.get("metrics") or {}
        people = list(faces.get("people") or [])[:MAX_WORKERS]
        active = _active_people(state)
        ready = bool(faces.get("ready") or metrics.get("ready"))
        self.engine.setText("● FACE READY" if ready else "● FACE OFFLINE")
        self.engine.setStyleSheet(
            f"color:{TH.OK if ready else TH.WARN};font-size:8px;font-weight:800;"
        )
        self.capacity.setText(f"{len(people)} / {MAX_WORKERS} enrolled")
        self.add_button.setEnabled(len(people) < MAX_WORKERS)
        self.add_button.setText("＋ Add Worker" if len(people) < MAX_WORKERS else "Roster Full 10/10")

        if people != self._people or active != self._last_active:
            self._people = people
            self._last_active = active
            if self._selected_id and not any(str(p.get("person_id")) == self._selected_id for p in people):
                self._selected_id = ""
            if not self._selected_id and people:
                self._selected_id = str(people[0].get("person_id") or "")
            self.rebuild()
        else:
            self._refresh_profile()

    def rebuild(self):
        _clear_layout(self.grid)
        active = self._last_active
        for slot in range(MAX_WORKERS):
            person = self._people[slot] if slot < len(self._people) else None
            avatar = None
            selected = False
            if person:
                person_id = str(person.get("person_id") or "")
                if person_id not in self._avatars and person.get("has_avatar"):
                    self._avatars[person_id] = face._fetch_avatar(person_id)
                avatar = self._avatars.get(person_id)
                selected = person_id == self._selected_id
            card = WorkerCard(
                slot + 1,
                person,
                avatar,
                bool(person and str(person.get("person_id") or "") in active),
                selected,
            )
            if person:
                pid = str(person.get("person_id") or "")
                card.clicked.connect(lambda pid=pid: self.select_worker(pid))
            else:
                card.clicked.connect(self.open_enrollment)
            self.grid.addWidget(card, slot // 5, slot % 5)
        self._refresh_profile()

    def select_worker(self, person_id: str):
        if person_id == self._selected_id:
            return
        self._selected_id = person_id
        self.rebuild()

    def _selected(self) -> dict | None:
        return next(
            (person for person in self._people if str(person.get("person_id") or "") == self._selected_id),
            None,
        )

    def _refresh_profile(self):
        person = self._selected()
        if not person:
            self.profile_avatar.clear()
            self.profile_avatar.setText("—")
            self.profile_name.setText("Select a worker")
            self.profile_role.setText("Worker profile details")
            self.profile_status.setText("● —")
            self.profile_status.setStyleSheet(f"color:{TH.DIM};font-size:9px;font-weight:800;")
            self.profile_meta.setText("Select one of the ten roster slots.")
            self.delete_button.setEnabled(False)
            return

        person_id = str(person.get("person_id") or "")
        avatar = self._avatars.get(person_id)
        self.profile_avatar.clear()
        if avatar is not None and not avatar.isNull():
            self.profile_avatar.setPixmap(
                avatar.scaled(116, 116, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            )
        else:
            initials = "".join(
                part[:1].upper() for part in str(person.get("name") or "?").split()[:2]
            )
            self.profile_avatar.setText(initials or "?")
        self.profile_name.setText(str(person.get("name") or "Unnamed"))
        self.profile_role.setText(str(person.get("job_title") or person.get("department") or "Worker"))
        present = person_id in self._last_active
        self.profile_status.setText("● CURRENTLY VISIBLE" if present else "● NOT IN VIEW")
        self.profile_status.setStyleSheet(
            f"color:{TH.OK if present else TH.DIM};font-size:9px;font-weight:800;"
        )
        samples = int(person.get("samples") or 0)
        self.profile_meta.setText(
            f"Worker ID       {person.get('employee_id') or person_id}\n"
            f"Face enrollment {min(samples, ENROLLMENT_SAMPLES)}/{ENROLLMENT_SAMPLES}\n"
            f"Recognitions    {int(person.get('recognitions') or 0)}\n"
            f"Last seen       {_fmt_datetime(person.get('last_seen'))}\n"
            f"Enrolled        {_fmt_datetime(person.get('created_at'))}"
        )
        self.delete_button.setEnabled(True)

    def open_enrollment(self):
        if len(self._people) >= MAX_WORKERS:
            self.hub.toast("⚠ Worker roster is full · maximum 10 profiles")
            return
        dialog = WorkerEnrollmentDialog(self.hub)
        dialog.exec()

    def delete_selected(self):
        person = self._selected()
        if not person:
            return
        person_id = str(person.get("person_id") or "")
        name = str(person.get("name") or person_id)
        answer = QMessageBox.question(
            self,
            "Delete worker",
            f"Delete {name}'s worker profile and enrolled face samples?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            face._json_request("DELETE", f"/faces/people/{person_id}", timeout=6.0)
        except Exception as exc:
            self.hub.toast(f"⚠ Delete failed · {exc}")
            return
        self._avatars.pop(person_id, None)
        self._selected_id = ""
        self.hub.toast(f"✓ Worker deleted · {name}")


class EnrollmentLauncherPage(face.base.Page):
    def __init__(self, hub):
        super().__init__()
        self.hub = hub
        self.title_row("Enrollment", "worker enrollment is integrated with Persons")
        card = QFrame()
        card.setObjectName("chartCard")
        card.setMaximumWidth(620)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(12)
        icon = QLabel("🪪")
        icon.setStyleSheet("font-size:42px;")
        title = QLabel("Add a worker profile")
        title.setStyleSheet("font-size:18px;font-weight:900;color:white;")
        text = QLabel(
            "Enrollment now uses the same 10-slot Worker roster. Enter Full Name and Job / Role, "
            "choose the visible camera track, then capture 10 quality-gated face samples."
        )
        text.setWordWrap(True)
        text.setStyleSheet(f"color:{TH.DIM};font-size:10px;")
        self.status = QLabel("Checking roster…")
        self.status.setStyleSheet(f"color:{TH.DIM};font-size:9px;font-weight:700;")
        button = QPushButton("＋ Open Worker Enrollment")
        button.setObjectName("btnPrimary")
        button.clicked.connect(self.open_dialog)
        self.open_button = button
        layout.addWidget(icon)
        layout.addWidget(title)
        layout.addWidget(text)
        layout.addWidget(self.status)
        layout.addSpacing(8)
        layout.addWidget(button)
        layout.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(card)
        row.addStretch(1)
        self.v.addStretch(1)
        self.v.addLayout(row)
        self.v.addStretch(1)
        self._state: dict = {}

    def refresh(self, state: dict):
        self._state = state
        people = list(((state.get("faces") or {}).get("people") or []))
        slots = min(len(people), MAX_WORKERS)
        self.status.setText(f"Roster capacity · {slots}/{MAX_WORKERS} workers")
        self.open_button.setEnabled(slots < MAX_WORKERS)
        if slots >= MAX_WORKERS:
            self.open_button.setText("Roster Full 10/10")
        else:
            self.open_button.setText("＋ Open Worker Enrollment")

    def open_dialog(self):
        people = list(((self._state.get("faces") or {}).get("people") or []))
        if len(people) >= MAX_WORKERS:
            self.hub.toast("⚠ Worker roster is full · maximum 10 profiles")
            return
        WorkerEnrollmentDialog(self.hub).exec()


def run():
    # Keep the proven MUKAMMAL + CUDA Face stack. Only replace the people/enrollment UX.
    face.PersonManagementPage = WorkerRosterPage
    face.EnrollmentPage = EnrollmentLauncherPage
    return cuda.run()
