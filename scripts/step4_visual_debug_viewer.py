#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from services.shared.camera_config import CameraConfig, load_settings

FRAME_PREFIX = "ML_TRACK_FRAME "
OBJECT_PREFIX = "ML_TRACK_OBJECT "


def _kv(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        out[key] = value
    return out


@dataclass
class VisualTrack:
    track_id: str
    state: str
    confirmed: bool
    predicted: bool
    score: float
    box: tuple[float, float, float, float]
    velocity: tuple[float, float, float, float]
    seen_at: float


class TrackLogTailer:
    def __init__(self, path: Path, camera_id: str) -> None:
        self.path = path
        self.camera_id = camera_id
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._tracks: dict[str, VisualTrack] = {}
        self._current_ids: set[str] = set()
        self._thread = threading.Thread(target=self._run, name="step4-track-log", daemon=True)
        self.last_error = ""
        self.frames = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 1.5) -> None:
        self._thread.join(timeout)

    def snapshot(self) -> list[VisualTrack]:
        with self._lock:
            rows = [self._tracks[tid] for tid in self._current_ids if tid in self._tracks]
        return sorted(rows, key=lambda row: row.track_id)

    def _parse_frame(self, line: str) -> None:
        fields = _kv(line)
        if fields.get("camera") != self.camera_id:
            return
        raw = fields.get("ids", "-")
        ids = set() if raw == "-" else {x for x in raw.split(",") if x}
        with self._lock:
            self._current_ids = ids
            self._tracks = {tid: row for tid, row in self._tracks.items() if tid in ids}
            self.frames += 1

    def _parse_object(self, line: str) -> None:
        fields = _kv(line)
        if fields.get("camera") != self.camera_id:
            return
        tid = fields.get("id", "")
        if not tid:
            return
        try:
            box = tuple(float(v) for v in fields["box_norm"].split(","))
            vel = tuple(float(v) for v in fields["vel_norm_s"].split(","))
            if len(box) != 4 or len(vel) != 4:
                return
            row = VisualTrack(
                track_id=tid,
                state=fields.get("state", "tracked"),
                confirmed=fields.get("confirmed", "0") == "1",
                predicted=fields.get("predicted", "0") == "1",
                score=float(fields.get("score", "0")),
                box=(box[0], box[1], box[2], box[3]),
                velocity=(vel[0], vel[1], vel[2], vel[3]),
                seen_at=time.monotonic(),
            )
        except (KeyError, ValueError):
            return
        with self._lock:
            self._tracks[tid] = row

    def _run(self) -> None:
        while not self._stop.is_set() and not self.path.exists():
            self.last_error = f"waiting for {self.path}"
            self._stop.wait(0.20)
        if self._stop.is_set():
            return

        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as fh:
                # Visual acceptance is live-only; do not replay stale historical boxes.
                fh.seek(0, 2)
                self.last_error = ""
                while not self._stop.is_set():
                    line = fh.readline()
                    if not line:
                        self._stop.wait(0.01)
                        continue
                    if line.startswith(FRAME_PREFIX):
                        self._parse_frame(line[len(FRAME_PREFIX) :])
                    elif line.startswith(OBJECT_PREFIX):
                        self._parse_object(line[len(OBJECT_PREFIX) :])
        except Exception as exc:
            self.last_error = str(exc)


class MainStreamSource:
    """One temporary main-stream debug decode; production Camera Service is untouched."""

    def __init__(self, camera: CameraConfig, width: int, height: int, latency_ms: int) -> None:
        Gst.init(None)
        self.camera = camera
        self.width = width
        self.height = height
        self.latency_ms = latency_ms
        self._lock = threading.Lock()
        self._image: QImage | None = None
        self._version = 0
        self.frames = 0
        self.last_error = ""
        self._fps_at = time.monotonic()
        self._fps_frames = 0
        self.fps = 0.0

        for plugin in ("nvurisrcbin", "queue", "nvvideoconvert", "capsfilter", "appsink"):
            if Gst.ElementFactory.find(plugin) is None:
                raise RuntimeError(f"missing GStreamer/DeepStream plugin: {plugin}")

        self.pipeline = Gst.Pipeline.new("step4-visual-debug")
        self.source = Gst.ElementFactory.make("nvurisrcbin", "source")
        self.queue = Gst.ElementFactory.make("queue", "latest")
        self.convert = Gst.ElementFactory.make("nvvideoconvert", "convert")
        self.caps = Gst.ElementFactory.make("capsfilter", "rgba")
        self.sink = Gst.ElementFactory.make("appsink", "sink")
        if not all((self.pipeline, self.source, self.queue, self.convert, self.caps, self.sink)):
            raise RuntimeError("could not build debug video pipeline")

        self._set_if(self.queue, "max-size-buffers", 1)
        self._set_if(self.queue, "max-size-bytes", 0)
        self._set_if(self.queue, "max-size-time", 0)
        self._set_if(self.queue, "leaky", 2)
        self._set_if(self.queue, "silent", True)

        self.source.connect("deep-element-added", self._configure_rtsp_child)
        self.source.connect("pad-added", self._source_pad_added)
        self.source.set_property("uri", camera.uri)
        self._set_if(self.source, "disable-audio", True)
        self._set_if(self.source, "select-rtp-protocol", 4)
        self._set_if(self.source, "latency", latency_ms)
        self._set_if(self.source, "drop-on-latency", True)
        self._set_if(self.source, "num-extra-surfaces", 4)
        self._set_if(self.source, "cudadec-memtype", 0)
        self._set_if(self.source, "rtsp-reconnect-interval", 2)
        self._set_if(self.source, "rtsp-reconnect-attempts", 3)
        self._set_if(self.source, "message-forward", True)
        self._set_if(self.source, "async-handling", True)

        self._set_if(self.convert, "compute-hw", 1)
        self._set_if(self.convert, "interpolation-method", 1)
        self.caps.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw,format=RGBA,width={self.width},height={self.height}"
            ),
        )

        self._set_if(self.sink, "emit-signals", True)
        self._set_if(self.sink, "max-buffers", 1)
        self._set_if(self.sink, "drop", True)
        self._set_if(self.sink, "sync", False)
        self._set_if(self.sink, "async", False)
        self._set_if(self.sink, "qos", False)
        self._set_if(self.sink, "max-lateness", -1)
        self._set_if(self.sink, "processing-deadline", 0)
        self.sink.connect("new-sample", self._new_sample)

        for element in (self.source, self.queue, self.convert, self.caps, self.sink):
            self.pipeline.add(element)
        if not self.queue.link(self.convert):
            raise RuntimeError("queue->convert failed")
        if not self.convert.link(self.caps):
            raise RuntimeError("convert->caps failed")
        if not self.caps.link(self.sink):
            raise RuntimeError("caps->appsink failed")
        self.bus = self.pipeline.get_bus()

    @staticmethod
    def _set_if(element, prop: str, value) -> bool:
        if element.find_property(prop) is None:
            return False
        element.set_property(prop, value)
        return True

    def _configure_rtsp_child(self, _bin, _sub_bin, element) -> None:
        factory = element.get_factory()
        if factory is None or factory.get_name() != "rtspsrc":
            return
        if self.camera.username:
            self._set_if(element, "user-id", self.camera.username)
            self._set_if(element, "user-pw", self.camera.password)
        self._set_if(element, "protocols", 4)
        self._set_if(element, "tcp-timestamp", True)
        self._set_if(element, "latency", self.latency_ms)
        self._set_if(element, "drop-on-latency", True)
        self._set_if(element, "buffer-mode", 3)
        self._set_if(element, "do-rtsp-keep-alive", True)

    def _source_pad_added(self, _source, pad) -> None:
        caps = pad.get_current_caps()
        if caps is None:
            try:
                caps = pad.query_caps(None)
            except Exception:
                caps = None
        if caps is not None and caps.get_size() > 0 and not caps.is_any():
            try:
                media = caps.get_structure(0).get_name()
            except Exception:
                media = ""
            if media and not media.startswith("video/"):
                return
        sink_pad = self.queue.get_static_pad("sink")
        if sink_pad is None or sink_pad.is_linked():
            return
        result = pad.link(sink_pad)
        if result != Gst.PadLinkReturn.OK:
            self.last_error = f"source pad link failed: {result}"

    def _new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        caps = sample.get_caps()
        buffer = sample.get_buffer()
        if caps is None or buffer is None or caps.get_size() == 0:
            return Gst.FlowReturn.ERROR
        struct = caps.get_structure(0)
        width = int(struct.get_value("width"))
        height = int(struct.get_value("height"))
        ok, mapped = buffer.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        try:
            size = len(mapped.data)
            stride = size // height if height > 0 and size % height == 0 else width * 4
            if stride < width * 4:
                return Gst.FlowReturn.ERROR
            image = QImage(
                mapped.data,
                width,
                height,
                stride,
                QImage.Format.Format_RGBA8888,
            ).copy()
        finally:
            buffer.unmap(mapped)

        now = time.monotonic()
        with self._lock:
            self._image = image
            self._version += 1
            self.frames += 1
            self._fps_frames += 1
            elapsed = now - self._fps_at
            if elapsed >= 1.0:
                self.fps = self._fps_frames / elapsed
                self._fps_frames = 0
                self._fps_at = now
            self.last_error = ""
        return Gst.FlowReturn.OK

    def start(self) -> None:
        result = self.pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("debug main-stream pipeline failed to start")

    def stop(self) -> None:
        self.pipeline.set_state(Gst.State.NULL)

    def latest(self) -> tuple[QImage | None, int, float]:
        with self._lock:
            return self._image, self._version, self.fps

    def poll_error(self) -> str:
        while True:
            msg = self.bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.WARNING)
            if msg is None:
                break
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                self.last_error = f"{err}: {debug or ''}".strip()
            else:
                err, debug = msg.parse_warning()
                self.last_error = f"warning: {err}: {debug or ''}".strip()
        return self.last_error


class Viewer(QWidget):
    def __init__(self, source: MainStreamSource, tracks: TrackLogTailer, camera_id: str) -> None:
        super().__init__()
        self.source = source
        self.tracks = tracks
        self.camera_id = camera_id
        self.image: QImage | None = None
        self.version = -1
        self.video_fps = 0.0
        self.setWindowTitle(f"Step 4 Visual Acceptance - {camera_id}")
        self.resize(1280, 760)
        self.setMinimumSize(800, 480)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)

    def _tick(self) -> None:
        image, version, fps = self.source.latest()
        if image is not None and version != self.version:
            self.image = image
            self.version = version
        self.video_fps = fps
        self.source.poll_error()
        self.update()

    @staticmethod
    def _predict(row: VisualTrack) -> tuple[float, float, float, float]:
        # UI-only interpolation. Never feeds prediction back into tracking.
        dt = min(0.55, max(0.0, time.monotonic() - row.seen_at))
        x1, y1, x2, y2 = row.box
        vx, vy, vw, vh = row.velocity
        cx = 0.5 * (x1 + x2) + vx * dt
        cy = 0.5 * (y1 + y2) + vy * dt
        w = max(0.01, (x2 - x1) + vw * dt)
        h = max(0.01, (y2 - y1) + vh * dt)
        return (
            max(0.0, min(1.0, cx - 0.5 * w)),
            max(0.0, min(1.0, cy - 0.5 * h)),
            max(0.0, min(1.0, cx + 0.5 * w)),
            max(0.0, min(1.0, cy + 0.5 * h)),
        )

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        if self.image is None or self.image.isNull():
            painter.setPen(QColor(230, 230, 230))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Connecting main stream..."
            )
            return

        iw, ih = self.image.width(), self.image.height()
        scale = min(self.width() / iw, self.height() / ih)
        dw, dh = iw * scale, ih * scale
        ox = 0.5 * (self.width() - dw)
        oy = 0.5 * (self.height() - dh)
        target = self.rect()
        target.setRect(int(ox), int(oy), int(dw), int(dh))
        painter.drawImage(target, self.image)

        tracks = [row for row in self.tracks.snapshot() if row.confirmed]
        font = QFont("Sans Serif", 10)
        font.setBold(True)
        painter.setFont(font)

        for row in tracks:
            x1, y1, x2, y2 = self._predict(row)
            left = ox + x1 * dw
            top = oy + y1 * dh
            right = ox + x2 * dw
            bottom = oy + y2 * dh
            predicted = row.predicted or row.state == "lost"
            color = QColor(255, 196, 0) if predicted else QColor(0, 235, 120)
            pen = QPen(color, 2.0)
            if predicted:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(
                int(left),
                int(top),
                int(max(1.0, right - left)),
                int(max(1.0, bottom - top)),
            )

            short_id = row.track_id.split("-")[-1]
            label = f"{short_id} {row.score:.2f}" + (" PRED" if predicted else "")
            metrics = painter.fontMetrics()
            tw = metrics.horizontalAdvance(label) + 10
            th = metrics.height() + 6
            ly = max(oy, top - th)
            painter.fillRect(int(left), int(ly), int(tw), int(th), QColor(0, 0, 0, 180))
            painter.setPen(color)
            painter.drawText(int(left + 5), int(ly + th - 5), label)

        header = (
            f"{self.camera_id}  MAIN STREAM  {self.video_fps:.1f} FPS  "
            f"tracks={len(tracks)}  Step4 V3 overlay"
        )
        painter.fillRect(12, 12, 540, 30, QColor(0, 0, 0, 170))
        painter.setPen(QColor(245, 245, 245))
        painter.drawText(22, 33, header)

        error = self.source.last_error or self.tracks.last_error
        if error:
            painter.fillRect(12, 48, min(self.width() - 24, 900), 28, QColor(120, 0, 0, 190))
            painter.setPen(QColor(255, 230, 230))
            painter.drawText(22, 68, error[:150])

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.close()
            return
        if event.key() == Qt.Key.Key_F:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.tracks.stop()
        self.source.stop()
        self.tracks.join()
        event.accept()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show one smooth main camera stream with Step 4 V3 tracker metadata overlay."
    )
    parser.add_argument("--camera", default="CAM-01")
    parser.add_argument("--track-log", default="/tmp/ML_STEP4_V3_VISUAL.log")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--latency-ms", type=int, default=80)
    args = parser.parse_args()

    settings = load_settings()
    by_id = {camera.camera_id: camera for camera in settings.cameras}
    if args.camera not in by_id:
        raise SystemExit(f"unknown camera {args.camera}; available={','.join(by_id)}")

    log_path = Path(args.track_log)
    tracks = TrackLogTailer(log_path, args.camera)
    source = MainStreamSource(
        by_id[args.camera],
        max(320, args.width),
        max(180, args.height),
        max(40, args.latency_ms),
    )

    print(
        "STEP4_VISUAL_DEBUG "
        f"camera={args.camera} source=main-stream direct_debug_decode=1 "
        f"tracker_log={log_path} overlay=qpaint prediction=velocity-bounded "
        "production_camera_service_modified=0",
        flush=True,
    )
    print("STEP4_VISUAL_KEYS q/esc=quit f=fullscreen", flush=True)

    app = QApplication(sys.argv)
    tracks.start()
    source.start()
    viewer = Viewer(source, tracks, args.camera)
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
