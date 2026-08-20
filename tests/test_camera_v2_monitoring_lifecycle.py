from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QTimer
from PySide6.QtWidgets import QApplication

import services.camera_v2.sentinel_ui_monitoring_native as monitoring


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def wait(qt_app, milliseconds: int) -> None:
    loop = __import__("PySide6.QtCore", fromlist=["QEventLoop"]).QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


class FakeProcess:
    def __init__(self, alive: bool = True) -> None:
        self.alive = alive
        self.exitcode = None

    def is_alive(self) -> bool:
        return self.alive


class FakeController:
    def __init__(self) -> None:
        self.process = None
        self.started: list[int] = []
        self.focused: list[int | None] = []
        self.stop_count = 0

    def start_or_bind(self, xid: int) -> None:
        self.started.append(int(xid))
        self.process = FakeProcess(True)

    def poll(self):
        return type("Status", (), {"state": "STARTING"})(), {"cameras": []}

    def focus(self, source_id: int | None) -> None:
        self.focused.append(source_id)

    def stop(self) -> None:
        self.stop_count += 1
        self.process = None


def test_native_host_can_republish_the_same_xid(qt_app) -> None:
    host = monitoring.NativeVideoSurface()
    emitted: list[int] = []
    host.nativeReady.connect(emitted.append)
    host.resize(900, 700)
    host.show()
    wait(qt_app, 250)

    assert emitted and emitted[-1] > 0
    xid = emitted[-1]
    initial_count = len(emitted)
    host.publish_current_xid()
    assert len(emitted) == initial_count
    host.publish_current_xid(force=True)
    assert len(emitted) == initial_count + 1
    assert emitted[-1] == xid

    host.close()


def test_native_surface_maps_the_aspect_fitted_camera_grid(qt_app) -> None:
    surface = monitoring.NativeVideoSurface(camera_count=6)
    surface.resize(1200, 900)

    assert surface.source_at(QPoint(20, 450)) is None
    assert surface.source_at(QPoint(200, 100)) == 0
    assert surface.source_at(QPoint(1000, 800)) == 5
    surface.set_focused_source(5)
    assert surface.source_at(QPoint(20, 450)) == 5

    surface.close()


def test_monitoring_restarts_after_early_pipeline_exit(qt_app, monkeypatch) -> None:
    monkeypatch.setattr(monitoring, "ProPipelineController", FakeController)
    page = monitoring.MonitoringPage()
    page.resize(1200, 800)
    page.show()
    wait(qt_app, 450)

    assert page.controller.started

    # Hiding/showing the embedded native child may reuse the same XID. It still
    # has to be rebound so the live sink exposes its last frame again.
    first_xid = page.controller.started[0]
    starts_before_show = len(page.controller.started)
    page.hide()
    wait(qt_app, 180)
    page.show()
    wait(qt_app, 250)
    assert page.controller.started[-1] == first_xid
    assert len(page.controller.started) > starts_before_show

    page.open_fullscreen_grid()
    assert page.identity_panel.isHidden()
    assert page._root_layout.getContentsMargins() == (0, 0, 0, 0)

    page.surface.cameraActivated.emit(4)
    assert page.controller.focused[-1] == 4
    assert page._focused_source == 4

    page.exit_fullscreen()
    assert page.controller.focused[-1] is None
    assert page._focused_source is None
    assert page.identity_panel.isHidden()

    page.controller.process.alive = False
    page._restart_not_before = 0.0
    page._ensure_started()

    assert page.controller.stop_count == 1
    assert page.controller.started[-1] == first_xid

    page.shutdown()
    page.close()
