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
            self.analytics.push()
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

    # ---------------- Analytics ----------------
    def patched_analytics_push(self):
        try:
            data = sm.analytics_service.get_current_data()

            self.occ.set_series([
                (data.get("occupancy_series", []), ui.TH.ACCENT, "occ"),
            ])

            self.gpu_fps.set_series([
                (data.get("gpu_series", []), ui.TH.WARN, "gpu"),
                (data.get("fps_series", []), ui.TH.OK, "fps"),
            ])

            known = 0
            unknown = 0

            for state in sm.identity_manager.states.values():
                known += state.known
                unknown += state.unknown

            self.donut.set_values(known, unknown)

            visitors = self.hub.sys.visitors

            self.visitors.set_data(
                [visitors.get(s.id, 0) for s in self.hub.sys.sims[:6]],
                [s.id[-2:] for s in self.hub.sys.sims[:6]],
            )

            self.peak.set_data(
                self.hub.sys.peak,
                [f"{i:02d}" for i in range(24)],
                ui.TH.ACC2,
            )

            usage = self.hub.sys.usage

            self.usage.set_data(
                [usage.get(s.id, 0) for s in self.hub.sys.sims[:6]],
                [s.id[-2:] for s in self.hub.sys.sims[:6]],
                ui.TH.OK,
            )

            self.heatsum.update()

        except Exception as e:
            log.error("analytics_push error: %s", e)

    ui.AnalyticsPage.push = patched_analytics_push

    # ---------------- Person Management ----------------
    orig_pm_init = ui.PersonManagementPage.__init__

    def patched_pm_init(self, hub):
        orig_pm_init(self, hub)

        try:
            self.tbl.setContextMenuPolicy(Qt.CustomContextMenu)
            self.tbl.customContextMenuRequested.connect(
                lambda pos: pm_context_menu(self, pos)
            )

            hub.sys.people_changed.connect(self.rebuild)

        except Exception as e:
            log.error("pm_init patch error: %s", e)

    ui.PersonManagementPage.__init__ = patched_pm_init

    def patched_pm_add_record(self, rec):
        try:
            self.hub.sys._reload_people()
            self.rebuild()
        except Exception as e:
            log.error("pm_add_record error: %s", e)

    ui.PersonManagementPage.add_record = patched_pm_add_record

    def pm_context_menu(page, pos):
        try:
            menu = QMenu(page)

            row = page.tbl.rowAt(pos.y())

            if row >= 0:
                item = page.tbl.item(row, 1)

                if item is None:
                    return

                rec = item.data(Qt.UserRole)

                act_edit = menu.addAction("✏️ Edit")
                act_faces = menu.addAction("🖼 Update Faces")
                act_history = menu.addAction("📜 History")
                act_delete = menu.addAction("🗑 Delete")

                chosen = menu.exec(page.tbl.viewport().mapToGlobal(pos))

                if chosen == act_edit:
                    pm_edit_person(page, rec)

                elif chosen == act_faces:
                    pm_update_faces(page, rec)

                elif chosen == act_history:
                    ui.ProfileDialog(rec).exec()

                elif chosen == act_delete:
                    pm_delete_person(page, rec)

            else:
                act_add = menu.addAction("＋ Add Employee")
                act_unknown = menu.addAction("❓ Unknown Faces")

                chosen = menu.exec(page.tbl.viewport().mapToGlobal(pos))

                if chosen == act_add:
                    pm_add_person(page)

                elif chosen == act_unknown:
                    dlg = UnknownFacesDialog(sm.unknown_service, page)
                    dlg.exec()

        except Exception as e:
            log.error("pm_context_menu error: %s", e)

    def pm_add_person(page):
        dlg = PersonFormDialog(None, page)

        if dlg.exec() != ui.QDialog.Accepted:
            return

        data = dlg.get_data()

        sm.person_service.add_person(
            name=data.get("name"),
            department=data.get("department"),
            employee_id=data.get("employee_id"),
        )

    def pm_edit_person(page, rec):
        dlg = PersonFormDialog(
            {
                "name": rec.name,
                "department": rec.dept,
                "employee_id": rec.emp_id,
                "status": rec.status,
            },
            page,
        )

        if dlg.exec() != ui.QDialog.Accepted:
            return

        data = dlg.get_data()

        sm.person_service.update_person(
            person_id=rec.db_id,
            name=data.get("name"),
            department=data.get("department"),
            employee_id=data.get("employee_id"),
            status=data.get("status"),
        )

    def pm_delete_person(page, rec):
        reply = QMessageBox.question(
            page,
            "Delete employee",
            f"Delete {rec.name}?\nEmbeddings will also be deleted.",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            sm.person_service.delete_person(rec.db_id)

    def pm_update_faces(page, rec):
        files, _ = QFileDialog.getOpenFileNames(
            page,
            "Select face images",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )

        if not files:
            return

        images = []

        for f in files:
            img = cv2.imread(f)

            if img is not None:
                images.append(img)

        ok = sm.person_service.update_faces(rec.db_id, images)

        if ok:
            page.hub.toast(f"✅ Faces updated for {rec.name}")
        else:
            page.hub.toast("⚠ No valid face found in selected images")

    # ---------------- Enrollment ----------------
    orig_en_init = ui.EnrollmentPage.__init__

    def patched_en_init(self, hub):
        orig_en_init(self, hub)

        try:
            svc = sm.enrollment_service

            self.flash_overlay = FlashOverlay(self.surface)

            svc.status_changed.connect(self.face_status.setText)

            svc.capture_progress.connect(
                lambda c, t: (
                    self.prog.setValue(c),
                    self.prog_lbl.setText(f"Captured {c} / {t}"),
                )
            )

            svc.thumbnail_captured.connect(lambda pm: en_add_thumbnail(self, pm))

            svc.embedding_ready.connect(lambda ok, h: en_embedding(self, ok, h))

            svc.finished.connect(lambda ok, msg: en_finished(self, ok, msg))

            svc.shutter_requested.connect(sound_player.play_shutter)
            svc.flash_requested.connect(self.flash_overlay.flash)

            # original fake capture timer is not used
            # self.cap_timer.stop()

        except Exception as e:
            log.error("enrollment_init patch error: %s", e)

    ui.EnrollmentPage.__init__ = patched_en_init

    # def patched_en_start_capture(self):
    #     try:
    #         # clear thumbnails
    #         for i in reversed(range(self.thumbs.count())):
    #             w = self.thumbs.itemAt(i).widget()
    #             if w:
    #                 w.deleteLater()

    #         self.prog.setValue(0)
    #         self.prog_lbl.setText("Captured 0 / 10")
    #         self.emb.setText("Embedding: —")
    #         self.btn_reg.setEnabled(False)
    #         self.btn_cap.setEnabled(False)

    #         sm.enrollment_service.begin_capture_sequence()

    #     except Exception as e:
    #         log.error("en_start_capture error: %s", e)

    # ui.EnrollmentPage.start_capture = patched_en_start_capture

    # def patched_en_capture_one(self):
    #     # EnrollmentService state machine handles capture.
    #     return

    # ui.EnrollmentPage.capture_one = patched_en_capture_one

    def patched_en_register(self):
        try:
            name = self.name.text().strip()

            if not name:
                self.hub.toast("⚠ Please enter the person's name")
                self.name.setFocus()
                return

            svc = sm.enrollment_service

            if svc.state != "ready" or svc.current_embedding is None:
                self.hub.toast("⚠ Capture or upload images first")
                return

            print(f"[Enroll] Calling register_person for {name}", flush=True)

            ok = svc.register_person(
                name=name,
                department=self.dept.currentText(),
                employee_id=self.emp.text().strip(),
            )

            if ok:
                print(f"[Enroll] register_person returned True", flush=True)

                # ✅ MAJBURIY: Keyingi event loop da PM yangilash
                # Bu deadlock va race condition oldini oladi
                from PySide6.QtCore import QTimer

                def _deferred_pm_update():
                    try:
                        print(f"[Enroll] Deferred PM update START", flush=True)

                        # To'g'ridan-to'g'ri DB dan o'qish
                        raw_rows = sm.db.get_persons()
                        print(f"[Enroll] DB returned {len(raw_rows)} persons", flush=True)

                        for r in raw_rows:
                            print(f"  → id={r.get('id')} name={r.get('name')}", flush=True)

                        # PersonRecordUI ga o'tkazish
                        new_people = []
                        for row in raw_rows:
                            try:
                                rec = self.PersonRecordUI(row, sm.person_service)
                                new_people.append(rec)
                            except Exception as ex:
                                print(f"[Enroll] PersonRecordUI error for {row.get('name')}: {ex}", flush=True)

                        # hub.sys.people ni ALMASHTIRISH
                        self.hub.sys.people = new_people
                        print(f"[Enroll] hub.sys.people = {len(new_people)} records", flush=True)

                        # PM sahifasini REBUILD
                        if hasattr(self.hub, 'pm'):
                            self.hub.pm.rebuild()
                            print(f"[Enroll] PM table rows: {self.hub.pm.tbl.rowCount()}", flush=True)
                        else:
                            print(f"[Enroll] WARNING: hub.pm not found!", flush=True)

                        print(f"[Enroll] Deferred PM update DONE", flush=True)

                    except Exception as e:
                        print(f"[Enroll] Deferred PM update ERROR: {e}", flush=True)
                        import traceback
                        traceback.print_exc()

                # 50ms kechikish — enrollment thread to'liq tugashi uchun
                QTimer.singleShot(50, _deferred_pm_update)

                # UI tozalash
                self.name.clear()
                self.emp.clear()

                for i in reversed(range(self.thumbs.count())):
                    w = self.thumbs.itemAt(i).widget()
                    if w:
                        w.deleteLater()

                self.prog.setValue(0)
                self.prog_lbl.setText("Captured 0 / 10")
                self.emb.setText("Embedding: —")
                self.btn_reg.setEnabled(False)
                self.face_status.setText("🟢 Ready for next enrollment")

                self.hub.toast(f"✅ {name} registered successfully")

            else:
                print(f"[Enroll] register_person returned False", flush=True)
                self.hub.toast("⚠ Registration failed — check terminal logs")

        except Exception as e:
            print(f"[Enroll] Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self.hub.toast(f"⚠ Register error: {str(e)[:50]}")


    def en_add_thumbnail(page, pm):
        try:
            lb = QLabel()
            lb.setFixedSize(58, 58)
            lb.setPixmap(
                pm.scaled(
                    58,
                    58,
                    Qt.KeepAspectRatioByExpanding,
                    Qt.SmoothTransformation,
                )
            )
            lb.setStyleSheet("border:2px solid #2ecc71; border-radius:6px;")

            n = page.thumbs.count()
            page.thumbs.addWidget(lb, n // 5, n % 5)

        except Exception as e:
            log.error("en_add_thumbnail error: %s", e)

    def en_embedding(page, ok, hex_hash):
        try:
            if ok:
                page.emb.setText(f"Embedding: 512-d · {hex_hash}…  ✓")
                page.emb.setStyleSheet(
                    "color:#2ecc71; font-size:9.5px; font-family:Consolas,monospace;"
                )
            else:
                page.emb.setText("Embedding: failed")
                page.emb.setStyleSheet(
                    "color:#ef5350; font-size:9.5px; font-family:Consolas,monospace;"
                )
        except Exception as e:
            log.error("en_embedding error: %s", e)

    def en_finished(page, ok, msg):
        try:
            page.btn_cap.setEnabled(True)

            if ok:
                page.btn_reg.setEnabled(True)
            else:
                page.btn_reg.setEnabled(False)

        except Exception as e:
            log.error("en_finished error: %s", e)

    # ---------------- Settings ----------------
    orig_settings_init = ui.SettingsPage.__init__

    def patched_settings_init(self, hub):
        orig_settings_init(self, hub)

        try:
            # Cameras tab: add camera management button
            cam_tab = self.tabs.widget(0)
            lay = cam_tab.layout()

            cam_btn = QPushButton("＋ Add / Edit / Delete Cameras")
            cam_btn.setObjectName("btnGhost")
            cam_btn.setCursor(Qt.PointingHandCursor)

            cam_btn.clicked.connect(
                lambda: CamerasDialog(sm.settings_service, self).exec()
            )

            # insert before stretch
            lay.insertWidget(max(0, lay.count() - 1), cam_btn)

            # Database buttons real actions
            for btn in self.findChildren(QPushButton):
                txt = btn.text()

                if txt == "Backup Now":
                    try:
                        btn.clicked.disconnect()
                    except Exception:
                        pass
                    btn.clicked.connect(sm.settings_service.backup_database)

                elif txt == "Vacuum":
                    try:
                        btn.clicked.disconnect()
                    except Exception:
                        pass
                    btn.clicked.connect(sm.settings_service.vacuum_database)

        except Exception as e:
            log.error("settings_init patch error: %s", e)

    ui.SettingsPage.__init__ = patched_settings_init

    def patched_settings_save(self):
        try:
            data = {
                "det_conf": self.det.value() / 100.0,
                "face_threshold": self.face_th.value() / 100.0,
                "sound_enabled": self.snd.isChecked(),
            }

            sm.settings_service.save_settings(data)

            sound_player.set_enabled(self.snd.isChecked())

        except Exception as e:
            log.error("settings_save error: %s", e)

    ui.SettingsPage._save = patched_settings_save

    def patched_settings_change_pwd(self):
        try:
            if self.pw1.text() and self.pw1.text() == self.pw2.text():
                ok = sm.settings_service.force_set_password(self.pw1.text())

                if ok:
                    self.pw1.clear()
                    self.pw2.clear()
            else:
                self.hub.toast("⚠ Passwords do not match")

        except Exception as e:
            log.error("settings_change_pwd error: %s", e)

    ui.SettingsPage._change_pwd = patched_settings_change_pwd

    # ---------------- Events ----------------
    def patched_event_ack(self, sys_):
        try:
            self.e["ack"] = True

            src = self.e.get("_src", self.e)

            sm.events_service.ack_event(src)

            sys_.push_event(
                dict(
                    type="system",
                    level="info",
                    cam=self.e.get("cam", "SYS"),
                    person=f"Event acknowledged: {self.e.get('person', '')}",
                    conf=1.0,
                )
            )

            self.accept()

        except Exception as e:
            log.error("event_ack error: %s", e)

    ui.EventDetailDialog._ack = patched_event_ack

    def patched_events_export(self):
        try:
            events = []

            for e in self.hub.sys.events:
                if self._match(e):
                    events.append(e.get("_src", e))

            path = sm.export_service.export_events_csv(events)

            if path:
                self.hub.toast(f"📄 Exported → {path}")

        except Exception as e:
            log.error("events_export error: %s", e)

    ui.EventsPage.export_csv = patched_events_export

    # ---------------- Alerts sound ----------------
    try:
        sm.alerts_service.sound_requested.connect(sound_player.play_alert)
    except Exception as e:
        log.error("alerts sound connect error: %s", e)

    log.info("UI patches applied")