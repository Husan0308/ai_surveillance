from __future__ import annotations

import os
import queue
import time

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QToolButton, QWidget

from .sentinel_video import (
    CAMERA_COUNT,
    GRID_COLUMNS,
    GRID_ROWS,
    WALL_HEIGHT,
    WALL_WIDTH,
    LiveVideoWall,
    PipelineController,
    UiStatus,
    _put_status,
)


def _pipeline_process_pro(window_id: int, command_q, status_q) -> None:
    runtime = None
    try:
        if int(window_id) <= 0:
            raise RuntimeError("invalid Qt native window id")

        os.environ["CAMERA_V2_WALL_WIDTH"] = str(WALL_WIDTH)
        os.environ["CAMERA_V2_WALL_HEIGHT"] = str(WALL_HEIGHT)
        # Sentinel starts visually clean. Heat history still accumulates and each
        # camera can be enabled independently from its hover action.
        os.environ.setdefault("CAMERA_V2_HEATMAP", "1")
        os.environ.setdefault("CAMERA_V2_HEATMAP_VISIBLE", "0")

        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstVideo

        from .person_tracking_reid_heatmap import CameraPersonTrackingReIDHeatmap

        runtime = CameraPersonTrackingReIDHeatmap()
        if len(runtime.cameras) != CAMERA_COUNT:
            raise RuntimeError(
                f"Sentinel Monitoring expects {CAMERA_COUNT} enabled cameras, "
                f"found {len(runtime.cameras)}"
            )

        runtime.wall_width = WALL_WIDTH
        runtime.wall_height = WALL_HEIGHT
        runtime.tiler_rows = GRID_ROWS
        runtime.tiler_columns = GRID_COLUMNS
        runtime.tiler.set_property("rows", GRID_ROWS)
        runtime.tiler.set_property("columns", GRID_COLUMNS)
        runtime.tiler.set_property("width", WALL_WIDTH)
        runtime.tiler.set_property("height", WALL_HEIGHT)
        if runtime.tiler.find_property("show-source") is not None:
            runtime.tiler.set_property("show-source", -1)

        current_xid = int(window_id)
        current_focus = -1

        def bind_overlay(overlay, xid: int | None = None) -> None:
            nonlocal current_xid
            target = int(current_xid if xid is None else xid)
            if target <= 0:
                return
            current_xid = target
            GstVideo.VideoOverlay.set_window_handle(overlay, target)
            try:
                GstVideo.VideoOverlay.handle_events(overlay, False)
            except Exception:
                pass

        def set_focus(source_id: int | None) -> None:
            nonlocal current_focus
            if runtime.tiler.find_property("show-source") is None:
                return
            sid = (
                int(source_id)
                if source_id is not None and 0 <= int(source_id) < CAMERA_COUNT
                else -1
            )
            current_focus = sid
            runtime.tiler.set_property("show-source", sid)
            _put_status(status_q, "FOCUS", str(sid))

        bind_overlay(runtime.sink, current_xid)

        def on_sync_message(_bus, message, _data=None):
            try:
                prepare = GstVideo.is_video_overlay_prepare_window_handle_message(message)
            except Exception:
                structure = message.get_structure()
                prepare = bool(
                    structure and structure.get_name() == "prepare-window-handle"
                )
            if not prepare:
                return Gst.BusSyncReply.PASS
            try:
                bind_overlay(message.src)
                # Window-handle preparation can happen again after a rebind.
                # Preserve the selected source instead of silently returning to
                # the tiled wall.
                set_focus(current_focus)
                _put_status(status_q, "VIDEO_BOUND", f"xid={current_xid}")
                return Gst.BusSyncReply.DROP
            except Exception as exc:
                _put_status(status_q, "ERROR", f"video overlay: {exc}")
                return Gst.BusSyncReply.PASS

        runtime.bus.set_sync_handler(on_sync_message, None)

        def observe_bus(_bus, message):
            if message.type == Gst.MessageType.STATE_CHANGED and message.src == runtime.pipeline:
                try:
                    _old, new, _pending = message.parse_state_changed()
                    if new == Gst.State.PLAYING:
                        _put_status(
                            status_q,
                            "LIVE",
                            "6-camera DeepStream/NvDCF + ReID + native heatmap PLAYING",
                        )
                except Exception:
                    pass
            elif message.type == Gst.MessageType.ERROR:
                try:
                    err, _debug = message.parse_error()
                    src = message.src.get_name() if message.src else "unknown"
                    _put_status(status_q, "PIPELINE_WARNING", f"{src}: {err.message}")
                except Exception:
                    pass

        runtime.bus.connect("message", observe_bus)

        def poll_commands() -> bool:
            stop_requested = False
            latest_focus = None
            got_focus = False
            latest_bind = None
            latest_bind_focus = None
            heatmap_changes: dict[int, bool] = {}

            while True:
                try:
                    command, value = command_q.get_nowait()
                except queue.Empty:
                    break

                if command == "stop":
                    stop_requested = True
                elif command == "focus":
                    latest_focus = int(value)
                    got_focus = True
                elif command == "bind":
                    latest_bind = int(value)
                elif command == "bind_focus":
                    try:
                        xid, source_id = value
                        latest_bind_focus = (int(xid), int(source_id))
                    except Exception:
                        pass
                elif command == "heatmap":
                    try:
                        source_id, enabled = value
                        heatmap_changes[int(source_id)] = bool(enabled)
                    except Exception:
                        pass

            # A per-camera fullscreen transition must be atomic: first bind the
            # EGL overlay to the new native Qt surface, then set show-source on
            # the same GLib iteration. This avoids a tiled frame being rebound
            # without the requested camera focus.
            if latest_bind_focus is not None:
                xid, source_id = latest_bind_focus
                if xid > 0:
                    bind_overlay(runtime.sink, xid)
                    set_focus(source_id)
                    _put_status(
                        status_q,
                        "VIDEO_BOUND",
                        f"xid={xid} focus={source_id}",
                    )
                latest_bind = None
                got_focus = False

            if latest_bind is not None and latest_bind > 0:
                bind_overlay(runtime.sink, latest_bind)
                _put_status(status_q, "VIDEO_BOUND", f"xid={latest_bind}")

            if got_focus:
                set_focus(latest_focus)

            setter = getattr(runtime, "set_heatmap_source_enabled", None)
            if callable(setter):
                for source_id, enabled in heatmap_changes.items():
                    if 0 <= source_id < CAMERA_COUNT:
                        setter(source_id, enabled)
                        _put_status(
                            status_q,
                            "HEATMAP",
                            f"source={source_id} enabled={int(enabled)}",
                        )

            if stop_requested:
                runtime.stop()
                return False
            return True

        last_frames = {camera.camera_id: 0 for camera in runtime.cameras}
        last_seen = {camera.camera_id: 0.0 for camera in runtime.cameras}
        last_metric_t = time.monotonic()

        def publish_metrics() -> bool:
            nonlocal last_metric_t
            now = time.monotonic()
            elapsed = max(0.20, now - last_metric_t)
            last_metric_t = now
            rows = []

            for index, camera in enumerate(runtime.cameras):
                stat = runtime.stats[camera.camera_id]
                previous = int(last_frames.get(camera.camera_id, 0))
                current = int(stat.frames)
                delta = max(0, current - previous)
                last_frames[camera.camera_id] = current
                if delta > 0:
                    last_seen[camera.camera_id] = now
                online = now - last_seen.get(camera.camera_id, 0.0) <= 2.5
                rows.append(
                    {
                        "id": camera.camera_id,
                        "source_id": index,
                        "fps": delta / elapsed,
                        "online": online,
                    }
                )

            identity_metrics = {}
            identity = getattr(runtime, "identity", None)
            if identity is not None:
                try:
                    identity_metrics = identity.metrics()
                except Exception:
                    identity_metrics = {}

            heatmap_sources = {}
            source_states = getattr(runtime, "heatmap_source_states", None)
            if callable(source_states):
                try:
                    heatmap_sources = {
                        int(k): bool(v) for k, v in source_states().items()
                    }
                except Exception:
                    heatmap_sources = {}

            _put_status(
                status_q,
                "METRICS",
                {
                    "cameras": rows,
                    "total_people": int(getattr(runtime, "tracked_now", 0)),
                    "global_identity": identity_metrics,
                    "heatmap_sources": heatmap_sources,
                },
            )
            return True

        runtime.GLib.timeout_add(50, poll_commands)
        runtime.GLib.timeout_add(500, publish_metrics)
        _put_status(
            status_q,
            "STARTING",
            "professional 2x3 wall; per-camera heatmap; DeepStream hot path unchanged",
        )
        rc = runtime.run()
        _put_status(status_q, "STOPPED", f"exit={rc}")
    except BaseException as exc:
        _put_status(status_q, "ERROR", f"{type(exc).__name__}: {exc}")
        try:
            if runtime is not None:
                runtime.stop()
                runtime.pipeline.set_state(runtime.Gst.State.NULL)
        except Exception:
            pass


class ProPipelineController(PipelineController):
    def start_or_bind(self, window_id: int) -> None:
        xid = int(window_id)
        if xid <= 0:
            return
        if self.process is not None and self.process.is_alive():
            self.bind(xid)
            return
        self.process = self.ctx.Process(
            target=_pipeline_process_pro,
            args=(xid, self.command_q, self.status_q),
            name="sentinel-pro-live-camera-wall",
            daemon=False,
        )
        self.process.start()
        self.last_status = UiStatus("STARTING")

    def bind_focus(self, window_id: int, source_id: int | None) -> None:
        xid = int(window_id)
        if xid <= 0:
            return
        sid = -1 if source_id is None else int(source_id)
        try:
            self.command_q.put_nowait(("bind_focus", (xid, sid)))
        except queue.Full:
            pass

    def set_heatmap(self, source_id: int, enabled: bool) -> None:
        try:
            self.command_q.put_nowait(("heatmap", (int(source_id), bool(enabled))))
        except queue.Full:
            pass


class ProLiveVideoWall(LiveVideoWall):
    heatmapToggled = Signal(int, bool)

    def __init__(self, cameras, people, parent: QWidget | None = None) -> None:
        super().__init__(cameras, people, parent)
        self._hover_source: int | None = None
        self.action_frames: list[QFrame] = []
        self.heatmap_buttons: list[QToolButton] = []
        self.fullscreen_buttons: list[QToolButton] = []
        self.tile_borders: list[tuple[QFrame, QFrame, QFrame, QFrame]] = []

        for source_id in range(min(CAMERA_COUNT, len(self.cameras))):
            borders = tuple(QFrame(self) for _ in range(4))
            for border in borders:
                border.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            self.tile_borders.append(borders)

            actions = QFrame(self)
            actions.setObjectName("cameraHoverActions")
            actions.setStyleSheet(
                "QFrame#cameraHoverActions{"
                "background:rgba(7,12,18,225);"
                "border:1px solid rgba(74,96,116,175);"
                "border-radius:6px;}"
                "QToolButton{background:transparent;color:#dce6ee;"
                "border:0;border-radius:4px;padding:5px 8px;font-weight:600;}"
                "QToolButton:hover{background:#182632;color:#ffffff;}"
                "QToolButton:checked{background:#11352f;color:#39d9c5;}"
            )
            row = QHBoxLayout(actions)
            row.setContentsMargins(3, 3, 3, 3)
            row.setSpacing(2)

            heat = QToolButton(actions)
            heat.setText("Heatmap")
            heat.setToolTip("Shu camera heatmapini yoqish/o'chirish")
            heat.setCheckable(True)
            heat.setCursor(Qt.PointingHandCursor)
            heat.toggled.connect(
                lambda checked, sid=source_id: self.heatmapToggled.emit(sid, checked)
            )
            row.addWidget(heat)

            full = QToolButton(actions)
            full.setText("⛶")
            full.setToolTip("Shu camerani fullscreen ochish")
            full.setCursor(Qt.PointingHandCursor)
            full.clicked.connect(
                lambda _checked=False, sid=source_id: self.cameraDoubleClicked.emit(sid)
            )
            row.addWidget(full)

            actions.adjustSize()
            actions.hide()
            self.action_frames.append(actions)
            self.heatmap_buttons.append(heat)
            self.fullscreen_buttons.append(full)

        self._layout_overlays()
        self._refresh_tile_frames()

    def _refresh_tile_frames(self) -> None:
        for sid, borders in enumerate(self.tile_borders):
            hovered = sid == self._hover_source
            color = "#39d9c5" if hovered else "rgba(45,62,78,205)"
            thickness = 2 if hovered else 1
            for border in borders:
                border.setStyleSheet(f"background:{color};border:0;")
                border.setProperty("tileThickness", thickness)
                border.raise_()
        for sid, action in enumerate(self.action_frames):
            action.setVisible(sid == self._hover_source)
            if action.isVisible():
                action.raise_()

    def _layout_overlays(self) -> None:
        super()._layout_overlays()
        if not hasattr(self, "tile_borders"):
            return
        for sid, borders in enumerate(self.tile_borders):
            left, top, width, height = self._tile_rect(sid)
            hovered = sid == self._hover_source
            t = 2 if hovered else 1
            top_b, right_b, bottom_b, left_b = borders
            top_b.setGeometry(left, top, width, t)
            right_b.setGeometry(left + width - t, top, t, height)
            bottom_b.setGeometry(left, top + height - t, width, t)
            left_b.setGeometry(left, top, t, height)

            if sid < len(self.action_frames):
                actions = self.action_frames[sid]
                actions.adjustSize()
                actions.move(
                    left + width - actions.width() - 10,
                    top + 38,
                )

    def _set_hover_source(self, source_id: int | None) -> None:
        if source_id == self._hover_source:
            return
        self._hover_source = source_id
        self._refresh_tile_frames()
        self._layout_overlays()

    def mouseMoveEvent(self, event) -> None:
        self._set_hover_source(self.source_at(event.position().toPoint()))
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self._set_hover_source(None)
        super().leaveEvent(event)

    def update_metrics(self, metrics: dict) -> None:
        super().update_metrics(metrics)
        states = dict((metrics or {}).get("heatmap_sources") or {})
        for sid, button in enumerate(self.heatmap_buttons):
            state = states.get(sid, states.get(str(sid), button.isChecked()))
            state = bool(state)
            if button.isChecked() != state:
                button.blockSignals(True)
                button.setChecked(state)
                button.blockSignals(False)
