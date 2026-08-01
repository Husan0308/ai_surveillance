with open('ui.py', 'r') as f:
    c = f.read()

changes = 0

# 1. Birinchi showEvent ni O'CHIRISH (dublikat)
old1 = """    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, 'db_sync'):
            self.db_sync()

    def _on_persons_online"""
new1 = """    def _on_persons_online"""
if old1 in c:
    c = c.replace(old1, new1)
    print("✅ 1. Dublikat showEvent o'chirildi")
    changes += 1

# 2. Ikkinchi showEvent ni TOZALASH (db_sync ga o'tkazish)
old2 = '''    def showEvent(self, e):

        # PM_AUTO_SYNC_FIX
        try:
            if hasattr(self, "sync_db"):
                self.sync_db()
        except Exception as e:
            print(f"[PM] sync_db error: {e}", flush=True)

        try:
            if hasattr(self, "load_persons"):
                self.load_persons()
        except Exception as e:
            print(f"[PM] load_persons error: {e}", flush=True)

        try:
            if hasattr(self, "rebuild"):
                self.rebuild()
        except Exception as e:
            print(f"[PM] rebuild error: {e}", flush=True)

        try:
            if hasattr(self, "refresh"):
                self.refresh()
        except Exception:
            pass

        print("[PM] ✅ Auto DB sync on enter", flush=True)
        """Person Management ochilganda DB dan avto-sync."""
        super().showEvent(e)
        import time as _st
        _n = _st.time()
        if not hasattr(self, "_last_db_sync") or _n - self._last_db_sync > 3:
            self._last_db_sync = _n
            try:
                self.force_refresh()
            except Exception as _se:
                print(f"[PM] showEvent sync xato: {_se}", flush=True)'''

new2 = '''    def showEvent(self, e):
        """Person Management ochilganda DB dan avto-sync."""
        super().showEvent(e)
        import time as _st
        _n = _st.time()
        if not hasattr(self, "_last_db_sync") or _n - self._last_db_sync > 3:
            self._last_db_sync = _n
            try:
                self.db_sync()
                print("[PM] ✅ Auto DB sync on enter", flush=True)
            except Exception as _se:
                print(f"[PM] showEvent sync xato: {_se}", flush=True)'''

if old2 in c:
    c = c.replace(old2, new2)
    print("✅ 2. showEvent tozalandi → db_sync")
    changes += 1

# 3. _auto_refresh_presence NameError tuzatish
old3 = """                if is_online:
                    status_text, status_color = "🟢 Online", "#2ecc71"
                else:
                    pass  # block emptied
                st = QTableWidgetItem(status_text)
                st.setForeground(QColor(status_color))
                self.tbl.setItem(r, 3, st)"""
new3 = """                if is_online:
                    status_text, status_color = "🟢 Online", "#2ecc71"
                else:
                    status_text, status_color = "", "#95a5a6"
                st = QTableWidgetItem(status_text)
                st.setForeground(QColor(status_color))
                self.tbl.setItem(r, 3, st)"""
if old3 in c:
    c = c.replace(old3, new3)
    print("✅ 3. _auto_refresh_presence NameError tuzatildi")
    changes += 1

# 4. 3s auto timer ni O'CHIRISH (juda og'ir)
old4 = """        # PM_AUTO_TIMER_FIX
        try:
            from PySide6.QtCore import QTimer

            if not hasattr(self, "_pm_auto_timer"):
                self._pm_auto_timer = QTimer(self)
                self._pm_auto_timer.timeout.connect(self._pm_auto_refresh)
                self._pm_auto_timer.start(3000)
        except Exception:
            pass"""
new4 = """        # PM_AUTO_TIMER_FIX — OLIB TASHLANDI (showEvent + persons_online signal yetarli)"""
if old4 in c:
    c = c.replace(old4, new4)
    print("✅ 4. 3s auto timer o'chirildi")
    changes += 1

# 5. Dublikat print ni o'chirish
old5 = '''            print(f"[PM] rebuild: {len(people)} people to display", flush=True)
            print(f"[PM] rebuild: {len(people)} people to display", flush=True)'''
new5 = '''            print(f"[PM] rebuild: {len(people)} people to display", flush=True)'''
if old5 in c:
    c = c.replace(old5, new5)
    print("✅ 5. Dublikat print o'chirildi")
    changes += 1

if changes > 0:
    with open('ui.py', 'w') as f:
        f.write(c)
    print(f"\n💾 ui.py saqlandi ({changes} ta o'zgarish)")
else:
    print("❌ Hech narsa o'zgarmadi")
