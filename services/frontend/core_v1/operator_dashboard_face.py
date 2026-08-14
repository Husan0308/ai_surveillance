from __future__ import annotations

import base64
from datetime import datetime
import http.client
import json
import threading
import uuid

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import operator_dashboard_mukammal as base
from .dashboard import ML_HOST, ML_PORT


TH = base.TH
CAMERA_SPECS = base.CAMERA_SPECS


def _json_request(method: str, path: str, payload: dict | None = None, timeout: float = 5.0):
    connection = http.client.HTTPConnection(ML_HOST, ML_PORT, timeout=timeout)
    try:
        body = None
        headers = {"Connection": "close", "Cache-Control": "no-cache"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        data = {}
        if raw:
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception:
                data = {"raw": raw.decode("utf-8", "replace")}
        if response.status >= 400:
            detail = data.get("detail") if isinstance(data, dict) else None
            raise RuntimeError(str(detail or f"HTTP {response.status}"))
        return data
    finally:
        connection.close()


def _fetch_avatar(person_id: str) -> QPixmap | None:
    connection = http.client.HTTPConnection(ML_HOST, ML_PORT, timeout=3.0)
    try:
        connection.request(
            "GET",
            f"/faces/avatar/{person_id}",
            headers={"Connection": "close", "Cache-Control": "no-cache"},
        )
        response = connection.getresponse()
        payload = response.read()
        if response.status != 200 or not payload:
            return None
        pixmap = QPixmap()
        if not pixmap.loadFromData(payload, "JPG"):
            return None
        return pixmap
    except Exception:
        return None
    finally:
        connection.close()


class FaceBackendState(base.BackendState):
    def __init__(self):
        super().__init__()
        self.state["faces"] = {"enabled": False, "ready": False, "people": []}

    def _run(self):
        connection = None
        while not self._stop.is_set():
            try:
                if connection is None:
                    connection = http.client.HTTPConnection(ML_HOST, ML_PORT, timeout=2.5)
                health = self._get_json(connection, "/health")
                tracks = self._get_json(connection, "/tracks")
                faces = self._get_json(connection, "/faces")
                with self._lock:
                    self._camera_edges(health)
                    self.state = {
                        "connected": True,
                        "health": health,
                        "tracks": tracks,
                        "faces": faces,
                    }
            except Exception:
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass
                connection = None
                with self._lock:
                    self.state = {**self.state, "connected": False}
                self._stop.wait(0.20)
                continue
            self._stop.wait(0.35)


class PersonManagementPage(base.Page):
    def __init__(self, hub):
        super().__init__()
        self.hub = hub
        row = self.title_row("Person Management", "real Face DB")
        self.search = QLineEdit()
        self.search.setPlaceholderText("🔍 Search people…")
        self.search.setMaximumWidth(220)
        self.search.textChanged.connect(self.rebuild)
        row.addWidget(self.search)
        enroll = QPushButton("＋ Enroll New")
        enroll.setObjectName("btnPrimary")
        enroll.clicked.connect(lambda: hub.navigate("enroll"))
        row.addWidget(enroll)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Photo", "Name", "Department", "Status", "Last Seen", "Recognitions"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.v.addWidget(self.table, 1)

        self._people = []
        self._avatars: dict[str, QPixmap | None] = {}

    @staticmethod
    def _format_seen(value) -> str:
        if not value:
            return "Never"
        text = str(value)
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return dt.astimezone().strftime("%d %b  %H:%M")
        except Exception:
            return text[:16]

    def refresh(self, state: dict):
        faces = state.get("faces") or {}
        people = list(faces.get("people") or [])
        if people != self._people:
            self._people = people
            self.rebuild()

    def rebuild(self):
        query = self.search.text().strip().lower()
        rows = [
            person
            for person in self._people
            if not query
            or query
            in f"{person.get('name','')} {person.get('department','')} {person.get('employee_id','')}".lower()
        ]
        self.table.setRowCount(len(rows))
        for row, person in enumerate(rows):
            person_id = str(person.get("person_id") or "")
            if person_id not in self._avatars and person.get("has_avatar"):
                self._avatars[person_id] = _fetch_avatar(person_id)
            photo = QTableWidgetItem()
            avatar = self._avatars.get(person_id)
            if avatar is not None and not avatar.isNull():
                photo.setData(
                    Qt.DecorationRole,
                    avatar.scaled(42, 42, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation),
                )
            self.table.setItem(row, 0, photo)
            self.table.setItem(
                row,
                1,
                QTableWidgetItem(
                    f"{person.get('name','')}\n{person.get('employee_id') or person_id}"
                ),
            )
            self.table.setItem(row, 2, QTableWidgetItem(str(person.get("department") or "—")))
            status = QTableWidgetItem("● Known")
            status.setForeground(QColor(TH.OK))
            self.table.setItem(row, 3, status)
            self.table.setItem(row, 4, QTableWidgetItem(self._format_seen(person.get("last_seen"))))
            count = QTableWidgetItem(str(int(person.get("recognitions") or 0)))
            count.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 5, count)
            self.table.setRowHeight(row, 52)


class EnrollmentPage(base.Page):
    PROMPTS = [
        "Look straight at the camera",
        "Turn slightly LEFT",
        "Turn slightly RIGHT",
        "Look slightly UP",
        "Look slightly DOWN",
        "Straight again",
        "Left, a little closer",
        "Right, a little closer",
        "Small smile / neutral front",
        "Final straight sample",
    ]

    def __init__(self, hub):
        super().__init__()
        self.hub = hub
        self.title_row("Person Enrollment", "real face capture · 10 accepted samples")
        body = QHBoxLayout()
        body.setSpacing(12)

        left = QFrame()
        left.setObjectName("camCard")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)

        selectors = QHBoxLayout()
        self.camera = QComboBox()
        for camera_id, _name, location in CAMERA_SPECS:
            self.camera.addItem(f"{camera_id} · {location}", camera_id)
        self.camera.currentIndexChanged.connect(self.change_camera)
        self.track = QComboBox()
        self.track.setMinimumWidth(220)
        selectors.addWidget(self.camera)
        selectors.addWidget(self.track, 1)
        lv.addLayout(selectors)

        self.surface = base.CameraSurface(hub.feeds[CAMERA_SPECS[0][0]])
        lv.addWidget(self.surface, 1)
        self.status = QLabel("Waiting for Face engine…")
        self.status.setStyleSheet(f"color:{TH.WARN};padding:9px;font-size:10px;")
        lv.addWidget(self.status)
        body.addWidget(left, 1)

        right = QFrame()
        right.setObjectName("chartCard")
        right.setFixedWidth(370)
        form = QVBoxLayout(right)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(10)
        title = QLabel("REGISTER NEW PERSON")
        title.setStyleSheet(
            f"color:{TH.ACC2};font-size:9px;font-weight:800;letter-spacing:1.5px;"
        )
        form.addWidget(title)
        self.name = QLineEdit()
        self.name.setPlaceholderText("Full Name *")
        self.dept = QComboBox()
        self.dept.addItems(["Security", "IT", "Finance", "HR", "Operations", "Management"])
        self.emp = QLineEdit()
        self.emp.setPlaceholderText("Employee ID (optional)")
        form.addWidget(self.name)
        form.addWidget(self.dept)
        form.addWidget(self.emp)

        self.progress_label = QLabel("Captured 0 / 10")
        self.progress_label.setStyleSheet(f"color:{TH.DIM};font-size:10px;font-weight:700;")
        self.progress = QProgressBar()
        self.progress.setRange(0, 10)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        form.addWidget(self.progress_label)
        form.addWidget(self.progress)

        self.thumbs = []
        thumb_grid = QGridLayout()
        for index in range(10):
            box = QLabel(str(index + 1))
            box.setFixedSize(56, 56)
            box.setAlignment(Qt.AlignCenter)
            box.setStyleSheet(
                f"color:{TH.FAINT};background:#11161c;border:1px dashed {TH.BORDER};border-radius:6px;"
            )
            thumb_grid.addWidget(box, index // 5, index % 5)
            self.thumbs.append(box)
        form.addLayout(thumb_grid)

        self.prompt = QLabel(self.PROMPTS[0])
        self.prompt.setWordWrap(True)
        self.prompt.setStyleSheet(f"color:{TH.ACC2};font-size:10px;font-weight:700;padding:4px;")
        form.addWidget(self.prompt)
        form.addStretch(1)

        self.capture = QPushButton("📸 Capture 10 samples")
        self.capture.setObjectName("btnGhost")
        self.capture.setEnabled(False)
        self.capture.clicked.connect(self.start_capture)
        self.register = QPushButton("💾 Register")
        self.register.setObjectName("btnPrimary")
        self.register.setEnabled(False)
        self.register.clicked.connect(self.commit)
        self.name.textChanged.connect(
            lambda _text: self.register.setEnabled(len(self.tokens) >= 10 and bool(self.name.text().strip()))
        )
        form.addWidget(self.capture)
        form.addWidget(self.register)
        body.addWidget(right)
        self.v.addLayout(body, 1)

        self.tokens: list[str] = []
        self.session_id = uuid.uuid4().hex
        self._last_state = {}
        self._attempts = 0
        self.capture_timer = QTimer(self)
        self.capture_timer.setInterval(1150)
        self.capture_timer.timeout.connect(self.capture_one)

    def hideEvent(self, event):
        self.capture_timer.stop()
        super().hideEvent(event)

    def change_camera(self):
        camera_id = self.camera.currentData()
        if camera_id in self.hub.feeds:
            self.surface.set_feed(self.hub.feeds[camera_id])
        self._update_tracks(self._last_state)

    def refresh(self, state: dict):
        self._last_state = state
        faces = state.get("faces") or {}
        metrics = faces.get("metrics") or {}
        ready = bool(faces.get("ready") or metrics.get("ready"))
        error = str(metrics.get("last_error") or "")
        self._update_tracks(state)
        has_track = self.track.currentData() is not None
        self.capture.setEnabled(ready and has_track and not self.capture_timer.isActive())
        if not ready:
            self.status.setText(f"Face engine not ready · {error or 'loading model'}")
            self.status.setStyleSheet(f"color:{TH.WARN};padding:9px;font-size:10px;")
        elif not has_track:
            self.status.setText("Face ready · select a visible person track")
            self.status.setStyleSheet(f"color:{TH.WARN};padding:9px;font-size:10px;")
        elif not self.capture_timer.isActive() and len(self.tokens) < 10:
            self.status.setText("Face ready · selected track will be captured from RAW frame")
            self.status.setStyleSheet(f"color:{TH.OK};padding:9px;font-size:10px;")

    def _update_tracks(self, state: dict):
        camera_id = self.camera.currentData()
        current = self.track.currentData()
        camera = ((state.get("tracks") or {}).get("cameras") or {}).get(camera_id, {})
        rows = list(camera.get("tracks") or [])
        choices = []
        for row in rows:
            track_id = int(row.get("track_id") or 0)
            if track_id <= 0:
                continue
            label = str(row.get("display_id") or track_id)
            gid = str(row.get("global_id") or "")
            name = str(row.get("name") or "Unknown")
            text = f"{label}"
            if gid:
                text += f" / {gid}"
            text += f" · {name}"
            choices.append((text, track_id))
        existing = [self.track.itemData(index) for index in range(self.track.count())]
        new_ids = [value for _text, value in choices]
        if existing == new_ids:
            return
        self.track.blockSignals(True)
        self.track.clear()
        for text, value in choices:
            self.track.addItem(text, value)
        if current in new_ids:
            self.track.setCurrentIndex(new_ids.index(current))
        self.track.blockSignals(False)

    def _reset(self):
        self.capture_timer.stop()
        self.tokens = []
        self.session_id = uuid.uuid4().hex
        self._attempts = 0
        self.progress.setValue(0)
        self.progress_label.setText("Captured 0 / 10")
        self.prompt.setText(self.PROMPTS[0])
        self.register.setEnabled(False)
        for index, box in enumerate(self.thumbs):
            box.clear()
            box.setText(str(index + 1))
            box.setStyleSheet(
                f"color:{TH.FAINT};background:#11161c;border:1px dashed {TH.BORDER};border-radius:6px;"
            )

    def start_capture(self):
        if self.track.currentData() is None:
            self.hub.toast("⚠ Select a visible person track")
            return
        self._reset()
        self.capture.setEnabled(False)
        self.status.setText("Capturing · " + self.PROMPTS[0])
        self.capture_timer.start()
        self.capture_one()

    def capture_one(self):
        if len(self.tokens) >= 10:
            self.capture_timer.stop()
            self.register.setEnabled(bool(self.name.text().strip()))
            self.capture.setEnabled(True)
            self.status.setText("✅ 10 accepted samples · ready to register")
            self.status.setStyleSheet(f"color:{TH.OK};padding:9px;font-size:10px;font-weight:700;")
            return
        camera_id = str(self.camera.currentData() or "")
        track_id = self.track.currentData()
        if not camera_id or track_id is None:
            self.capture_timer.stop()
            self.status.setText("Selected track disappeared")
            return
        self._attempts += 1
        try:
            sample = _json_request(
                "POST",
                f"/faces/enrollment/sample/{camera_id}/{int(track_id)}?session_id={self.session_id}",
                timeout=8.0,
            )
        except Exception as exc:
            self.status.setText(f"Waiting for a good face · {exc}")
            self.status.setStyleSheet(f"color:{TH.WARN};padding:9px;font-size:10px;")
            if self._attempts >= 35:
                self.capture_timer.stop()
                self.capture.setEnabled(True)
                self.status.setText("Capture stopped · move closer/face camera and retry")
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
                pixmap.scaled(56, 56, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            )
        self.thumbs[index].setStyleSheet(
            f"background:#11161c;border:2px solid {TH.OK};border-radius:6px;"
        )
        count = len(self.tokens)
        self.progress.setValue(count)
        self.progress_label.setText(
            f"Captured {count} / 10 · quality {float(sample.get('quality') or 0):.2f}"
        )
        if count < 10:
            self.prompt.setText(self.PROMPTS[count])
            self.status.setText("Captured ✅ · " + self.PROMPTS[count])
        else:
            self.capture_one()

    def commit(self):
        name = self.name.text().strip()
        if not name:
            self.hub.toast("⚠ Enter the person's name")
            self.name.setFocus()
            return
        if len(self.tokens) < 10:
            self.hub.toast("⚠ Capture 10 accepted samples first")
            return
        try:
            person = _json_request(
                "POST",
                "/faces/enrollment/commit",
                {
                    "name": name,
                    "department": self.dept.currentText(),
                    "employee_id": self.emp.text().strip(),
                    "sample_tokens": self.tokens,
                },
                timeout=10.0,
            )
        except Exception as exc:
            self.hub.toast(f"⚠ Enrollment failed · {exc}")
            return
        self.hub.toast(f"✅ {person.get('name', name)} registered")
        self.name.clear()
        self.emp.clear()
        self._reset()
        self.hub.navigate("people")


class RightPanel(base.RightPanel):
    def refresh(self, state: dict, events: list[dict]):
        super().refresh(state, events)
        tracks = state.get("tracks") or {}
        known = set()
        unknown = set()
        for camera_id, camera in (tracks.get("cameras") or {}).items():
            for track in camera.get("tracks") or []:
                person_id = str(track.get("person_id") or "").strip()
                if person_id:
                    known.add(person_id)
                    continue
                gid = str(track.get("global_id") or "").strip()
                key = gid or f"{camera_id}:{track.get('track_id')}"
                unknown.add(key)
        self.known[1].setText(str(len(known)))
        self.unknown[1].setText(str(len(unknown)))


class SettingsPage(base.Page):
    def __init__(self):
        super().__init__()
        self.title_row("Settings", "runtime summary")
        tabs = QTabWidget()
        self.v.addWidget(tabs, 1)

        cameras = QWidget()
        cv = QVBoxLayout(cameras)
        for camera_id, name, location in CAMERA_SPECS:
            row = QFrame()
            row.setObjectName("chartCard")
            h = QHBoxLayout(row)
            h.addWidget(QLabel(f"🎥 {camera_id}  —  {name}"))
            h.addStretch(1)
            loc = QLabel(location)
            loc.setStyleSheet(f"color:{TH.DIM};")
            h.addWidget(loc)
            cv.addWidget(row)
        cv.addStretch(1)
        tabs.addTab(cameras, "🎥 Cameras")

        ai = QWidget()
        form = QFormLayout(ai)
        form.addRow("Detector", QLabel("YOLO26m · CUDA · person-only"))
        form.addRow("Tracking", QLabel("Camera-local Hungarian ownership tracker"))
        form.addRow("Cross-camera ReID", QLabel("OSNet-AIN · CPU"))
        form.addRow("Pose / Heatmap", QLabel("Not enabled"))
        tabs.addTab(ai, "🤖 AI")

        recognition = QWidget()
        rform = QFormLayout(recognition)
        rform.addRow("Face model", QLabel("InsightFace buffalo_l"))
        rform.addRow("Provider", QLabel("CPUExecutionProvider"))
        rform.addRow("Search", QLabel("upper person crop · 320×320"))
        rform.addRow("Enrollment", QLabel("10 quality-gated samples"))
        tabs.addTab(recognition, "🆔 Recognition")


class MainWindow(base.MainWindow):
    def refresh_state(self):
        super().refresh_state()
        state, _events = self.state_reader.snapshot()
        self.people.refresh(state)
        self.enroll.refresh(state)


def run():
    base.BackendState = FaceBackendState
    base.PersonManagementPage = PersonManagementPage
    base.EnrollmentPage = EnrollmentPage
    base.RightPanel = RightPanel
    base.SettingsPage = SettingsPage
    base.MainWindow = MainWindow
    return base.run()
