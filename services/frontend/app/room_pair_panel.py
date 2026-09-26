from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from services.mv3dt_room.acceptance_harness import AcceptanceHarness
from services.mv3dt_room.presence import (
    CurrentPresencePositionFilter,
    assert_presence_contract,
)


class RoomPairPanel(QWidget):
    """Small live room-pair presentation: two camera identity columns + BEV."""

    acceptance_mode_changed = Signal(bool)
    acceptance_candidates_requested = Signal(bool)
    acceptance_start_requested = Signal()
    acceptance_candidates_changed = Signal(list)
    acceptance_labels_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.last_presence_assertions: dict = {}
        self._snapshot: dict = {}
        self._transform = self._load_transform()
        self._map = self._load_map()
        self._show_native = os.getenv("FRONTEND_SHOW_NATIVE_IDS", "0") == "1"
        self._position_filter = CurrentPresencePositionFilter()
        self._session_id = ""
        self._acceptance_runtime_session = ""
        artifact_root = Path(os.getenv("AI_SURVEILLANCE_ROOT", Path.cwd())) / ".runtime/mv3dt/dev-room-cam01-cam04/acceptance"
        self.acceptance = AcceptanceHarness(artifact_root)
        self._acceptance_candidates: list[dict] = []
        self.title = QLabel("Dev Room · CAM-01 + CAM-04 · Shared BEV")
        self.title.setStyleSheet("font-weight: bold;")
        self.mode = QLabel("SOURCE: checking…")
        self.mode.setStyleSheet("color: #9fb3c8;")
        self.people = QLabel("Waiting for room-pair runtime…")
        self.people.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.bev = QLabel()
        self.bev.setMinimumSize(420, 420)
        self.bev.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bev.setStyleSheet("background: #111; color: white;")
        self.acceptance_toggle = QCheckBox("Acceptance mode · ground truth only")
        self.acceptance_toggle.setToolTip(
            "Operator annotations are saved for evaluation only and are never sent to tracker, MVA, OSNet, or identity services."
        )
        self.acceptance_controls = QWidget()
        form = QFormLayout(self.acceptance_controls)
        self.subtest_choice = QComboBox()
        self.subtest_choice.addItem("Subtest A · cross-camera + crossing", "A")
        self.subtest_choice.addItem("Subtest B · full exit + re-entry", "B")
        self.tester_choice = QComboBox()
        self.tester_choice.addItems(["Tester_A", "Tester_B"])
        self.tester_choice.currentTextChanged.connect(lambda _value: self._refresh_acceptance_panel())
        self.distinct_people_confirm = QCheckBox("I confirm Tester_A and Tester_B are different physical people")
        self.distinct_people_confirm.setVisible(False)
        self.distinct_people_confirm.setToolTip(
            "Acceptance-only ground truth. This confirmation is never sent to production identity, tracker, or BEV logic."
        )
        self.start_subtest_button = QPushButton("Start selected subtest")
        self.begin_exit_button = QPushButton("BEGIN EXIT TEST")
        self.begin_exit_button.setEnabled(False)
        self.begin_exit_button.setVisible(False)
        self.acceptance_status = QLabel("Enable acceptance mode to select a visible bbox.")
        self.acceptance_status.setWordWrap(True)
        self.acceptance_checklist = QLabel("Subtest checklist will appear here.")
        self.acceptance_checklist.setWordWrap(True)
        self.acceptance_dimensions = QLabel("Awaiting verified live camera dimensions")
        self.acceptance_dimensions.setWordWrap(True)
        form.addRow("Test:", self.subtest_choice)
        form.addRow("Source dimensions:", self.acceptance_dimensions)
        form.addRow("Assign clicked bbox to:", self.tester_choice)
        form.addRow(self.distinct_people_confirm)
        form.addRow(self.start_subtest_button)
        form.addRow(self.begin_exit_button)
        form.addRow("Next action:", self.acceptance_status)
        form.addRow("Checklist:", self.acceptance_checklist)
        self.acceptance_controls.setVisible(False)
        self.acceptance_toggle.toggled.connect(self._on_acceptance_toggled)
        self.distinct_people_confirm.toggled.connect(lambda _checked: self._refresh_acceptance_panel())
        self.start_subtest_button.clicked.connect(self.acceptance_start_requested.emit)
        self.begin_exit_button.clicked.connect(self._begin_exit_test)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title)
        layout.addWidget(self.mode)
        layout.addWidget(self.people)
        layout.addWidget(self.acceptance_toggle)
        layout.addWidget(self.acceptance_controls)
        layout.addWidget(self.bev, 1)

    def _on_acceptance_toggled(self, enabled: bool) -> None:
        self.acceptance.enable(enabled)
        self.acceptance_controls.setVisible(enabled)
        if not enabled:
            self._acceptance_candidates = []
            self.acceptance_candidates_changed.emit([])
            self.acceptance_labels_changed.emit({})
        self.acceptance_candidates_requested.emit(enabled)
        self._refresh_acceptance_panel()

    def start_subtest(self) -> None:
        if not self.acceptance.enabled:
            self.acceptance_status.setText("Enable acceptance mode first.")
            return
        session = str(self._snapshot.get("session_id") or "live")
        if self._snapshot.get("source_mode") != "live" or self._snapshot.get("status") not in {"running", "live"}:
            self.acceptance_status.setText("Acceptance timer not started: wait for an active LIVE CAM-01/CAM-04 runtime.")
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        subtest = str(self.subtest_choice.currentData())
        if subtest == "A":
            if self.acceptance.subtest != "A":
                self.distinct_people_confirm.setChecked(False)
                self.acceptance.start_subtest(subtest, f"{session}_{stamp}")
                self._acceptance_runtime_session = session
                self.acceptance.expected_runtime_session = session
            elif self.acceptance.started_monotonic is None:
                try:
                    self.acceptance.confirm_subtest_a_ready(self.distinct_people_confirm.isChecked())
                except RuntimeError as exc:
                    self.acceptance_status.setText(str(exc))
                    self._refresh_acceptance_panel()
                    return
        else:
            self.acceptance.start_subtest(subtest, f"{session}_{stamp}")
            self._acceptance_runtime_session = session
            self.acceptance.expected_runtime_session = session
            self.distinct_people_confirm.setChecked(False)
        self.begin_exit_button.setEnabled(False)
        self._refresh_acceptance_panel()

    def _begin_exit_test(self) -> None:
        tester = str(self.tester_choice.currentText())
        try:
            self.acceptance.begin_exit_test(tester)
        except RuntimeError as exc:
            self.acceptance_status.setText(str(exc))
            return
        self.acceptance.exit_tester = tester
        self._refresh_acceptance_panel()

    def handle_acceptance_bbox_selected(self, camera_id: str, row: dict, crop: QImage) -> None:
        if not self.acceptance.enabled:
            return
        if str(row.get("camera_id", "")) != camera_id:
            self.acceptance_status.setText("Selection rejected: bbox camera did not match the clicked camera tile.")
            return
        tester = str(self.tester_choice.currentText())
        record = self.acceptance.testers.get(tester)
        if record:
            expected = str(record.get("canonical_person_id", "Unknown"))
            candidate = str(row.get("global_person_id") or row.get("application_id") or "Unknown")
            if candidate not in {"", "Unknown"} and candidate != expected:
                candidate_crop_path = self._save_acceptance_candidate_crop(tester, camera_id, row, crop)
                decision = self._confirm_same_physical_tester(
                    tester, expected, candidate, camera_id, row, record.get("reference_crop"), crop
                )
                if decision is None:
                    self.acceptance.record_identity_candidate_review(tester, row, None, candidate_crop_path)
                    self.acceptance_status.setText("Comparison cancelled; Tester_A evidence is unchanged.")
                    return
                self.acceptance.record_identity_candidate_review(tester, row, decision, candidate_crop_path)
                if not decision:
                    self.acceptance_status.setText(
                        f"Recorded as a different person; {tester} remains anchored to {expected}."
                    )
                    return
                # Only a human-confirmed same-person observation reaches the
                # acceptance harness's identity-change check. It never enters
                # production identity, tracking, or BEV logic.
                reference_crop = candidate_crop_path
                try:
                    self.acceptance.assign(tester, row, reference_crop)
                except (RuntimeError, ValueError) as exc:
                    self.acceptance_status.setText(str(exc))
                self._refresh_acceptance_panel()
                return
        reference_crop = None
        if tester not in self.acceptance.testers:
            record = self.acceptance.session_id or "acceptance"
            crop_dir = self.acceptance.artifact_root / "reference-crops"
            crop_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{record}_{tester}_{camera_id}_native-{row.get('native_track_id')}.png"
            path = crop_dir / filename
            if not crop.isNull() and crop.save(str(path), "PNG"):
                reference_crop = str(path)
        try:
            self.acceptance.assign(tester, row, reference_crop)
        except (RuntimeError, ValueError) as exc:
            self.acceptance_status.setText(str(exc))
            return
        self._refresh_acceptance_panel()

    def _save_acceptance_candidate_crop(self, tester: str, camera_id: str, row: dict, crop: QImage) -> str | None:
        if crop.isNull():
            return None
        crop_dir = self.acceptance.artifact_root / "candidate-crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        filename = (
            f"{self.acceptance.session_id}_{tester}_{camera_id}_"
            f"frame-{int(row.get('frame', -1))}_native-{int(row.get('native_track_id', -1))}_{stamp}.png"
        )
        path = crop_dir / filename
        return str(path) if crop.save(str(path), "PNG") else None

    def _confirm_same_physical_tester(
        self,
        tester: str,
        expected_person_id: str,
        candidate_person_id: str,
        camera_id: str,
        row: dict,
        reference_crop_path: str | None,
        candidate_crop: QImage,
    ) -> bool | None:
        """Show both acceptance-only crops before any changed ID is evaluated."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Acceptance-only physical-person confirmation")
        outer = QVBoxLayout(dialog)
        outer.addWidget(QLabel(
            f"Is this the same physical {tester}?\n"
            f"Reference: {expected_person_id} · Candidate: {candidate_person_id} · "
            f"{camera_id} frame {row.get('frame')} native {row.get('native_track_id')}\n"
            "YES/NO is recorded only for acceptance and never changes production identity."
        ))
        crops = QHBoxLayout()
        for title, pixmap in (
            ("Original Tester_A reference", QPixmap(reference_crop_path or "")),
            ("Current selected candidate", QPixmap.fromImage(candidate_crop)),
        ):
            column = QVBoxLayout()
            column.addWidget(QLabel(title))
            image = QLabel()
            image.setMinimumSize(280, 280)
            image.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image.setPixmap(
                pixmap.scaled(360, 360, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                if not pixmap.isNull()
                else QPixmap()
            )
            if pixmap.isNull():
                image.setText("Crop unavailable")
            column.addWidget(image)
            crops.addLayout(column)
        outer.addLayout(crops)
        buttons = QDialogButtonBox(dialog)
        yes = buttons.addButton("YES — same person", QDialogButtonBox.ButtonRole.YesRole)
        no = buttons.addButton("NO — different person", QDialogButtonBox.ButtonRole.NoRole)
        cancel = buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        yes.clicked.connect(dialog.accept)
        no.clicked.connect(dialog.reject)
        cancel.clicked.connect(dialog.reject)
        outer.addWidget(buttons)
        result = {"same": None}
        yes.clicked.connect(lambda: result.update(same=True))
        no.clicked.connect(lambda: result.update(same=False))
        dialog.exec()
        return result["same"]

    def update_acceptance_candidates(self, payload: dict) -> None:
        if not self.acceptance.enabled:
            return
        if payload.get("acceptance_only") is not True:
            self.acceptance_status.setText("Acceptance candidate endpoint did not identify as acceptance-only.")
            return
        if payload.get("source_mode") != "live":
            self.acceptance_status.setText("Acceptance harness requires source mode LIVE.")
            self._acceptance_candidates = []
            self.acceptance_candidates_changed.emit([])
            return
        if payload.get("coordinate_dimensions_ready") is not True:
            self._acceptance_candidates = []
            self.acceptance_candidates_changed.emit([])
            self.acceptance_dimensions.setText("Waiting for verified CAM-01/CAM-04 source dimensions")
            self.acceptance_status.setText(
                "Waiting for verified CAM-01/CAM-04 source dimensions; bbox selection is disabled to avoid a wrong annotation."
            )
            return
        dimensions = payload.get("source_dimensions_by_camera", {})
        provenance = payload.get("source_dimensions_source", {})
        labels = []
        for camera in ("CAM-01", "CAM-04"):
            item = dimensions.get(camera, {})
            suffix = "" if provenance.get(camera) == "runtime_crop_provenance" else " · validated profile"
            labels.append(f"{camera} {item.get('width', '?')}×{item.get('height', '?')}{suffix}")
        self.acceptance_dimensions.setText(" | ".join(labels))
        self._acceptance_candidates = list(payload.get("candidates", []))
        self.acceptance_candidates_changed.emit(self._acceptance_candidates)
        session = str(payload.get("session_id") or "")
        if (
            self.acceptance.subtest is not None
            and self._acceptance_runtime_session
            and self._acceptance_runtime_session != session
            and not self.acceptance._runtime_session_warning_logged
        ):
            self.acceptance._runtime_session_warning_logged = True
            warning = {
                "kind": "runtime_session_changed",
                "expected_session": self._acceptance_runtime_session,
                "actual_session": session,
            }
            self.acceptance.failure = self.acceptance.failure or warning
            self.acceptance._emit("LIVE_SESSION_CHANGED", {
                "expected_session": self._acceptance_runtime_session,
                "actual_session": session,
                "acceptance_invalidated": True,
                "harness_warning": "acceptance annotations remain local and were not passed to production logic",
            })
        self.acceptance.observe({
            "source_mode": payload.get("source_mode"),
            "session_id": session,
            "people": self._acceptance_candidates,
        })
        self._refresh_acceptance_panel()

    def _refresh_acceptance_panel(self) -> None:
        if self.acceptance.subtest is None:
            if self.acceptance.enabled and str(self.subtest_choice.currentData()) == "A":
                self.acceptance_status.setText("WAITING FOR: start the Subtest A waiting stage; its timer will not start yet.")
                self.acceptance_checklist.setText("[ ] Tester_A selected\n[ ] Tester_B selected")
                self.start_subtest_button.setText("Start Subtest A waiting stage")
                self.start_subtest_button.setEnabled(True)
            else:
                self.acceptance_status.setText("Start a subtest, then select a visible bbox.")
                self.acceptance_checklist.setText("No subtest active.")
                self.start_subtest_button.setText("Start selected subtest")
                self.start_subtest_button.setEnabled(True)
            self.distinct_people_confirm.setVisible(False)
            self.begin_exit_button.setVisible(False)
            self.begin_exit_button.setEnabled(False)
            return
        checklist = self.acceptance.checklist()
        lines = [f"[{ 'x' if item['done'] else ' ' }] {item['label']}" for item in checklist["items"]]
        self.acceptance_checklist.setText("\n".join(lines))
        waiting_for_a = self.acceptance.subtest == "A" and self.acceptance.started_monotonic is None
        both_a_assigned = all(name in self.acceptance.testers for name in ("Tester_A", "Tester_B"))
        self.distinct_people_confirm.setVisible(waiting_for_a and both_a_assigned)
        self.start_subtest_button.setText(
            "Confirm different people and start timed Subtest A"
            if waiting_for_a and both_a_assigned
            else "Waiting for Tester_A / Tester_B selections"
            if waiting_for_a
            else "Start selected subtest"
        )
        self.start_subtest_button.setEnabled(
            not waiting_for_a
            or (
                not self.acceptance.failure
                and both_a_assigned
                and self.distinct_people_confirm.isChecked()
            )
        )
        is_subtest_b = self.acceptance.subtest == "B"
        selected_tester = str(self.tester_choice.currentText())
        self.begin_exit_button.setVisible(is_subtest_b)
        self.begin_exit_button.setEnabled(self.acceptance.can_begin_exit_test(selected_tester))
        if is_subtest_b:
            self.begin_exit_button.setText(
                f"BEGIN EXIT TEST · {selected_tester}"
                if self.acceptance.can_begin_exit_test(selected_tester)
                else f"Waiting for stable {selected_tester} observations"
            )
        if self.acceptance.failure:
            self.acceptance_status.setText(f"ACCEPTANCE FAILURE: {self.acceptance.failure}")
        else:
            self.acceptance_status.setText(f"NEXT ACTION: {self.acceptance.next_action()}")
        labels = {}
        for tester, record in self.acceptance.testers.items():
            current = record.get("current_observation")
            if current:
                labels[(current["camera_id"], int(current["native_track_id"]))] = tester
        self.acceptance_labels_changed.emit(labels)

    @staticmethod
    def _load_transform() -> np.ndarray | None:
        path = Path(os.getenv("MV3DT_ROOM_CALIBRATION", ".runtime/mv3dt/calibration/dev-room-cam01-cam04-vggt-v1/transforms.yml"))
        try:
            return np.asarray(yaml.safe_load(path.read_text())["T_ov2px"], dtype=np.float64).reshape(3, 3)
        except (OSError, KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _load_map() -> QPixmap:
        path = Path(os.getenv("MV3DT_ROOM_MAP", ".runtime/mv3dt/calibration/dev-room-cam01-cam04-vggt-v1/map.png"))
        return QPixmap(str(path)) if path.exists() else QPixmap()

    def update_snapshot(self, snapshot: dict) -> None:
        self._snapshot = snapshot
        source_mode = str(snapshot.get("source_mode", "live")).upper()
        session_id = str(snapshot.get("session_id", ""))
        if session_id != self._session_id:
            self._position_filter.reset()
            self._session_id = session_id
        self.mode.setText(f"SOURCE: {source_mode}" + (f" · {session_id}" if session_id else ""))
        people = snapshot.get("people", [])
        if people:
            lines = []
            for person in people:
                label = f"{person.get('application_id', 'Unknown')} | {person.get('camera_id')}"
                if self._show_native:
                    label += f"  native:{person.get('debug', {}).get('native_mv3dt_id', '?')}"
                world = person.get("world")
                if world:
                    label += f"  BEV=({float(world[0]):.2f}, {float(world[1]):.2f})m"
                lines.append(label)
            self.people.setText("\n".join(lines))
        else:
            self.people.setText(f"Room-pair runtime: {snapshot.get('status', 'idle')}\nNo visible people")
        self._draw_bev(people)

    def _draw_bev(self, people: list[dict]) -> None:
        positions = self._position_filter.update(people)
        if self._map.isNull():
            self.last_presence_assertions = assert_presence_contract(
                people,
                set(),
                native_rendered_ids=(),
                stale_rendered_ids=(),
            )
            self.bev.setText("Shared BEV/map unavailable")
            return
        canvas = self._map.scaled(self.bev.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        sx = canvas.width() / max(1, self._map.width())
        sy = canvas.height() / max(1, self._map.height())
        colors = [QColor(80, 220, 80), QColor(40, 165, 255), QColor(255, 180, 60)]
        rendered_marker_ids = []
        for index, (identity, world) in enumerate(sorted(positions.items())):
            if self._transform is None:
                continue
            homogeneous = self._transform @ np.asarray([float(world[0]), float(world[1]), 1.0])
            if abs(homogeneous[2]) < 1e-9:
                continue
            x = int(homogeneous[0] / homogeneous[2] * sx)
            y = int(homogeneous[1] / homogeneous[2] * sy)
            color = colors[index % len(colors)]
            painter.setPen(QPen(color, 3))
            painter.setBrush(color)
            painter.drawEllipse(x - 6, y - 6, 12, 12)
            painter.drawText(x + 10, y - 8, identity)
            cameras = sorted({str(row.get('camera_id')) for row in people if row.get('application_id') == identity})
            if cameras:
                painter.drawText(x + 10, y + 10, " + ".join(cameras))
            rendered_marker_ids.append(identity)
        painter.end()
        duplicate_ids = {
            identity
            for identity in rendered_marker_ids
            if rendered_marker_ids.count(identity) > 1
        }
        self.last_presence_assertions = assert_presence_contract(
            people,
            rendered_marker_ids,
            duplicate_ids=duplicate_ids,
            native_rendered_ids=(),
            stale_rendered_ids=(),
        )
        self.bev.setPixmap(canvas)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._draw_bev(list(self._snapshot.get("people", [])))
