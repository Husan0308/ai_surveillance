from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication, QGridLayout, QMainWindow, QWidget

from shared.config import PROJECT_ENV_LOADED, camera_config
from services.ml_service.core_v1.manager import CameraManager
from services.ml_service.core_v1.stable_detector import StableYoloDetectorWorker
from services.ml_service.cameras.deepstream import deepstream_available


ROOT = Path(__file__).resolve().parents[3]
CAMERA_IDS = [f"CAM-{index:02d}" for index in range(1, 7)]


def _expand(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _load_core() -> dict:
    with open(ROOT / "config/core_v1.yaml", "r", encoding="utf-8") as handle:
        base = _expand(yaml.safe_load(handle) or {}).get("core_v1", {})
    core = dict(base)
    # Direct-view mode has one owner for each camera pipeline and no frame IPC.
    # Force the validated DeepStream/NVDEC backend so a hidden fallback cannot
    # make camera-start failures ambiguous.
    core.update(
        {
            "profile": "direct-camera-detection-v1",
            "capture_backend": "deepstream",
            "capture_output_width": 960,
            "capture_output_height": 540,
            "drop_on_latency": True,
            "postdecode_queue_buffers": 1,
            "decoder_extra_surfaces": 4,
            "startup_stagger_sec": 0.30,
            "startup_grace_sec": 10.0,
            "capture_timeout_ms": 700,
            "max_read_timeouts": 6,
            "reconnect_delay_sec": 0.75,
            "capture_metrics_interval_sec": 5.0,
            "max_pipeline_lag_ms": 500,
            "max_pipeline_lag_samples": 12,
        }
    )
    # Never override every camera with the legacy global 150 ms jitterbuffer.
    core.pop("rtsp_latency_ms", None)
    detector = dict(core.get("detector") or {})
    detector.update(
        {
            "enabled": True,
            "batch_size": 2,
            # Give all six NVDEC sessions time to negotiate before CUDA model
            # warmup starts. Detection remains latest-only after startup.
            "start_delay_sec": 2.0,
            "min_submit_interval_ms": 85,
            "max_submit_age_ms": 260,
            "max_result_age_ms": 700,
            "overlay_max_age_ms": 700,
        }
    )
    core["detector"] = detector
    return core


def _load_cameras() -> list[dict]:
    cameras = []
    for item in camera_config().get("cameras", []):
        if not item.get("online", True):
            continue
        camera = dict(item)
        codec = str(camera.get("display_codec") or camera.get("codec") or "").lower()
        # 20 ms is too aggressive for a six-stream RTSP wall: small LAN jitter
        # becomes visible as dropped/late frames. Use a still-low bounded floor.
        floor_ms = 80 if codec in {"h265", "hevc"} else 60
        camera["latency_ms"] = max(floor_ms, int(camera.get("latency_ms", floor_ms)))
        camera["drop_on_latency"] = True
        cameras.append(camera)
    return cameras


def _camera_preflight(cameras: list[dict]) -> None:
    ids = [str(camera.get("id", "")) for camera in cameras]
    if ids != CAMERA_IDS:
        raise RuntimeError(f"DIRECT_PREFLIGHT expected={CAMERA_IDS} got={ids}")

    if not deepstream_available():
        raise RuntimeError("DIRECT_PREFLIGHT DeepStream nvurisrcbin plugin is unavailable")

    missing_auth = []
    bad_sources = []
    for camera in cameras:
        cid = str(camera.get("id", "?"))
        source = str(camera.get("display_source") or camera.get("source") or "")
        parsed = urlsplit(source) if source else None
        if not source.lower().startswith(("rtsp://", "rtsps://")):
            bad_sources.append(cid)
            continue
        embedded_auth = bool(parsed and "@" in parsed.netloc)
        if not embedded_auth and (not camera.get("username") or not camera.get("password")):
            missing_auth.append(cid)

    if bad_sources:
        raise RuntimeError("DIRECT_PREFLIGHT invalid RTSP source: " + ",".join(bad_sources))
    if missing_auth:
        raise RuntimeError(
            "DIRECT_PREFLIGHT RTSP credentials are empty for "
            + ",".join(missing_auth)
            + f"; project_env_loaded={int(PROJECT_ENV_LOADED)}; "
            "check .env SURVEILLANCE_RTSP_USERNAME/SURVEILLANCE_RTSP_PASSWORD"
        )

    print(
        "DIRECT_PREFLIGHT=PASS "
        f"cameras={len(cameras)} auth={len(cameras) - len(missing_auth)}/{len(cameras)} "
        f"env_file={int(PROJECT_ENV_LOADED)} backend=deepstream-nvurisrcbin "
        "latest_queue=1 display=960x540",
        flush=True,
    )


def _wait_camera_start(manager: CameraManager, timeout_sec: float = 15.0) -> int:
    """Let camera/NVDEC negotiation finish before CUDA detector warmup.

    The window should never silently sit on six black tiles. If no camera can
    produce a first frame, fail with the real per-camera backend errors instead.
    """

    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    next_report = 0.0
    last_metrics = {}
    while time.monotonic() < deadline:
        last_metrics = manager.metrics()
        online = [cid for cid, item in last_metrics.items() if bool(item.get("online"))]
        if len(online) == len(manager.workers):
            print(f"DIRECT_CAMERA_START=PASS online={len(online)}/{len(manager.workers)}", flush=True)
            return len(online)

        now = time.monotonic()
        if now >= next_report:
            states = []
            for cid in CAMERA_IDS:
                item = last_metrics.get(cid) or {}
                if item.get("online"):
                    states.append(f"{cid}:LIVE")
                elif item.get("last_error"):
                    error = str(item.get("last_error")).replace("\n", " ")
                    states.append(f"{cid}:ERR[{error[:120]}]")
                else:
                    states.append(f"{cid}:WAIT")
            print("DIRECT_CAMERA_START " + " | ".join(states), flush=True)
            next_report = now + 1.0
        time.sleep(0.10)

    last_metrics = manager.metrics()
    online = [cid for cid, item in last_metrics.items() if bool(item.get("online"))]
    if online:
        print(
            f"DIRECT_CAMERA_START=PARTIAL online={len(online)}/{len(manager.workers)} "
            "continuing_without_blocking_healthy_cameras=1",
            flush=True,
        )
        return len(online)

    errors = []
    for cid in CAMERA_IDS:
        item = last_metrics.get(cid) or {}
        detail = str(item.get("last_error") or "no first frame")
        errors.append(f"{cid}={detail}")
    raise RuntimeError("DIRECT_CAMERA_START=FAIL " + " | ".join(errors))


class DirectCameraTile(QWidget):
    def __init__(self, camera_id: str, store, detector, manager: CameraManager):
        super().__init__()
        self.camera_id = camera_id
        self.store = store
        self.detector = detector
        self.manager = manager
        self._version = 0
        self._frame_ref = None
        self._image: QImage | None = None
        self._last_frame_mono = 0.0
        self.setMinimumSize(320, 180)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

    def refresh_if_new(self) -> None:
        frame, version = self.store.get()
        if frame is None or version <= self._version:
            return
        array = frame.image
        if array is None or getattr(array, "ndim", 0) != 3 or array.shape[2] != 3:
            return
        if not array.flags.c_contiguous:
            import numpy as np
            array = np.ascontiguousarray(array)
            frame = type(frame)(
                frame.camera_id,
                frame.frame_id,
                frame.captured_at,
                frame.captured_monotonic,
                array,
                frame.width,
                frame.height,
            )
        image = QImage(
            array.data,
            int(frame.width),
            int(frame.height),
            int(array.strides[0]),
            QImage.Format.Format_BGR888,
        )
        if image.isNull():
            return
        self._frame_ref = frame
        self._image = image
        self._version = int(version)
        self._last_frame_mono = time.monotonic()
        self.update()

    def _draw_detection(self, painter: QPainter, x: float, y: float, scale: float) -> None:
        result = self.detector.results.get(self.camera_id)
        if result is None:
            return
        age_ms = max(0.0, (time.monotonic() - float(result.produced_monotonic)) * 1000.0)
        max_age = float(self.detector.config.get("overlay_max_age_ms", 700))
        if max_age > 0 and age_ms > max_age:
            return
        pen = QPen(QColor("#00ff66"))
        pen.setWidth(2)
        painter.setPen(pen)
        font = QFont()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        for box in result.boxes:
            left = x + float(box.x1) * scale
            top = y + float(box.y1) * scale
            width = max(1.0, (float(box.x2) - float(box.x1)) * scale)
            height = max(1.0, (float(box.y2) - float(box.y1)) * scale)
            painter.drawRect(QRectF(left, top, width, height))
            label = f"Person {float(box.confidence):.2f}"
            label_w = max(82.0, painter.fontMetrics().horizontalAdvance(label) + 10.0)
            label_h = 20.0
            painter.fillRect(QRectF(left, max(y, top - label_h), label_w, label_h), QColor(0, 0, 0, 180))
            painter.setPen(QColor("#00ff66"))
            painter.drawText(QRectF(left + 5, max(y, top - label_h), label_w - 8, label_h), Qt.AlignmentFlag.AlignVCenter, label)
            painter.setPen(pen)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000"))
        image = self._image
        if image is not None and not image.isNull():
            sw = max(1, image.width())
            sh = max(1, image.height())
            scale = min(self.width() / sw, self.height() / sh)
            dw = max(1, round(sw * scale))
            dh = max(1, round(sh * scale))
            x = (self.width() - dw) / 2.0
            y = (self.height() - dh) / 2.0
            painter.drawImage(QRectF(x, y, dw, dh), image)
            self._draw_detection(painter, x, y, scale)
        else:
            painter.setPen(QColor("#777777"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "WAITING FOR CAMERA")

        painter.fillRect(8, 8, 82, 24, QColor(0, 0, 0, 160))
        painter.setPen(QColor("#ffffff"))
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(15, 25, self.camera_id)

        if self._last_frame_mono and time.monotonic() - self._last_frame_mono > 1.5:
            painter.setPen(QColor("#ff5a5a"))
            painter.drawText(self.width() - 75, 25, "OFFLINE")
        painter.end()


class DirectDetectionWindow(QMainWindow):
    def __init__(self, manager: CameraManager, detector: StableYoloDetectorWorker):
        super().__init__()
        self.manager = manager
        self.detector = detector
        self.setWindowTitle("AI Surveillance — Direct Cameras + Detection")
        self.setStyleSheet("background:#000;")

        body = QWidget()
        body.setStyleSheet("background:#000;")
        grid = QGridLayout(body)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(2)
        grid.setVerticalSpacing(2)

        self.tiles: list[DirectCameraTile] = []
        for index, camera_id in enumerate(CAMERA_IDS):
            store = manager.stores.get(camera_id)
            if store is None:
                continue
            tile = DirectCameraTile(camera_id, store, detector, manager)
            grid.addWidget(tile, index // 3, index % 3)
            self.tiles.append(tile)
        for column in range(3):
            grid.setColumnStretch(column, 1)
        for row in range(2):
            grid.setRowStretch(row, 1)
        self.setCentralWidget(body)

        self.render_timer = QTimer(self)
        self.render_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.render_timer.timeout.connect(self._render)
        self.render_timer.start(33)

        self.metrics_timer = QTimer(self)
        self.metrics_timer.timeout.connect(self._print_metrics)
        self.metrics_timer.start(5000)

    def _render(self) -> None:
        for tile in self.tiles:
            tile.refresh_if_new()

    def _print_metrics(self) -> None:
        camera_metrics = self.manager.metrics()
        detector = self.detector.metrics()
        parts = []
        for cid in CAMERA_IDS:
            item = camera_metrics.get(cid) or {}
            parts.append(
                f"{cid}:{float(item.get('source_fps') or 0):.1f}fps"
                f"/{str(item.get('capture_backend') or '?')}"
            )
        print(
            "DIRECT_CAMERA " + " ".join(parts)
            + f" | detector_ready={bool(detector.get('ready'))}"
            + f" batch_ms={float(detector.get('last_batch_ms') or 0):.1f}"
            + f" gpu_inputs_s={float(detector.get('camera_input_rate') or 0):.1f}",
            flush=True,
        )

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.showMaximized()
            return
        if event.key() == Qt.Key.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.render_timer.stop()
        self.metrics_timer.stop()
        self.detector.stop()
        self.detector.join(10)
        self.manager.stop()
        event.accept()


def run() -> int:
    core = _load_core()
    cameras = _load_cameras()
    _camera_preflight(cameras)

    manager = CameraManager(cameras, core)
    detector = StableYoloDetectorWorker(manager.stores, dict(core.get("detector") or {}), ROOT)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("AI Surveillance Direct Detection")

    manager.start()
    try:
        _wait_camera_start(manager)
        detector.start()
        window = DirectDetectionWindow(manager, detector)
        window.showMaximized()
        return app.exec()
    finally:
        detector.stop()
        detector.join(10)
        manager.stop()


if __name__ == "__main__":
    raise SystemExit(run())
