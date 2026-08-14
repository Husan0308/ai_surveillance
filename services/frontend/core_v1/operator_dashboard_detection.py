from __future__ import annotations

from collections import deque
import http.client
import json
import threading
import time
from datetime import datetime

from . import operator_dashboard_v2 as v2

base = v2.base


class DetectionRealtimeState:
    """Frontend state reader for the detection-only API surface."""

    def __init__(self):
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self.state = {
            "connected": False,
            "health": {},
            "detections": {},
            "reid": {},
            "room_mapping": {},
        }
        self.recent = deque(maxlen=30)
        self.events = deque(maxlen=100)
        self._online = {}

    def start(self):
        self._thread = threading.Thread(
            target=self._run,
            name="ui-detection-state",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop.set()

    def snapshot(self):
        with self._lock:
            return dict(self.state), list(self.recent), list(self.events)

    @staticmethod
    def _get_json(connection, path):
        connection.request(
            "GET",
            path,
            headers={"Connection": "keep-alive", "Cache-Control": "no-cache"},
        )
        response = connection.getresponse()
        payload = response.read()
        if response.status != 200:
            raise RuntimeError(f"GET {path}: {response.status}")
        return json.loads(payload.decode("utf-8") or "{}")

    def _record_camera_edges(self, health):
        now = datetime.now().strftime("%H:%M:%S")
        cameras = health.get("cameras") or {}
        for camera_id, metrics in cameras.items():
            online = bool(metrics.get("online"))
            previous = self._online.get(camera_id)
            self._online[camera_id] = online
            if previous is None or previous == online:
                continue
            entry = {
                "time": now,
                "camera": camera_id,
                "global_id": "Camera back online" if online else "Camera offline",
                "reason": "camera_online" if online else "camera_offline",
            }
            self.recent.appendleft(entry)
            self.events.appendleft(entry)

    def _run(self):
        connection = None
        while not self._stop.is_set():
            try:
                if connection is None:
                    connection = http.client.HTTPConnection(
                        base.ML_HOST,
                        base.ML_PORT,
                        timeout=2.0,
                    )
                health = self._get_json(connection, "/health")
                detections = self._get_json(connection, "/detections")
                with self._lock:
                    self._record_camera_edges(health)
                    self.state = {
                        "connected": True,
                        "health": health,
                        "detections": detections,
                        "reid": {},
                        "room_mapping": {},
                    }
            except Exception:
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass
                connection = None
                with self._lock:
                    self.state = {
                        **self.state,
                        "connected": False,
                    }
                self._stop.wait(0.15)
                continue
            self._stop.wait(0.25)


class DashboardPage(v2.DashboardPage):
    """Keep the old 3x2 visual layout, but expose detection only."""

    def __init__(self, toggle_callback):
        super().__init__(toggle_callback)
        self.heat.hide()
        self.pose.hide()

    def set_overlay_state(self, heatmap: bool, pose: bool):
        # Overlay features do not exist in this baseline.
        return


class RightRail(v2.RightRail):
    def update_state(self, state, events):
        detections = (state.get("detections") or {}).get("cameras") or {}
        detected = sum(len(value.get("boxes") or []) for value in detections.values())
        self.known[1].setText("0")
        self.unknown[1].setText(str(detected))

        health = state.get("health") or {}
        resources = health.get("service_resources") or {}
        gpu = float(resources.get("gpu_utilization_percent") or 0.0)
        cpu = float(resources.get("cpu_percent") or 0.0)
        publishers = health.get("publishers") or {}
        rates = [float(value.get("publish_rate") or 0.0) for value in publishers.values()]
        fps = sum(rates) / len(rates) if rates else 0.0

        self.gpu[0].setText(f"{gpu:.0f}%")
        self.gpu[1].setValue(max(0, min(100, int(round(gpu)))))
        self.cpu[0].setText(f"{cpu:.0f}%")
        self.cpu[1].setValue(max(0, min(100, int(round(cpu)))))
        self.fps[0].setText(f"{fps:.1f}")
        self.fps[1].setValue(max(0, min(100, int(round(fps / 30.0 * 100.0)))))

        self._clear(self.alerts)
        offline = [
            camera_id
            for camera_id, value in (health.get("cameras") or {}).items()
            if not value.get("online")
        ]
        for camera_id in offline[:3]:
            label = base.QLabel(f"│ {camera_id} — Camera offline")
            label.setStyleSheet(
                f"color:{base.RED};background:#1c222b;padding:8px;border-radius:5px;"
            )
            self.alerts.addWidget(label)
        if not offline:
            label = base.QLabel("● All cameras healthy")
            label.setStyleSheet(f"color:{base.GREEN};padding:6px;")
            self.alerts.addWidget(label)

        self._clear(self.recent)
        for entry in events[:8]:
            label = base.QLabel(
                f"{entry.get('time','')}  ●  {entry.get('camera','')} · "
                f"{entry.get('global_id','')}"
            )
            label.setFont(base.font(9))
            label.setStyleSheet(f"color:{base.MUTED};padding:2px;")
            self.recent.addWidget(label)


BaseOperatorWindow = base.OperatorWindow


class OperatorWindow(BaseOperatorWindow):
    def __init__(self):
        super().__init__()
        # Detection-only baseline: future feature pages remain visible in the old
        # console design but cannot be opened accidentally.
        for index in (1, 2, 3, 4):
            button = self.sidebar.buttons[index]
            button.setEnabled(False)
            button.setToolTip("Disabled in detection-only baseline")
        self.setWindowTitle("AI Surveillance — Detection Baseline")

    def set_overlay(self, kind, enabled):
        return


# OperatorWindow resolves these names from operator_dashboard globals at runtime.
base.DashboardPage = DashboardPage
base.RightRail = RightRail
base.RealtimeState = DetectionRealtimeState
base.OperatorWindow = OperatorWindow


def run():
    return base.run()
