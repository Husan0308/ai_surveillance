from __future__ import annotations

import os
import threading
import time

from services.camera_v11.deepstream_trt86_multi_v1 import V11DeepStreamTRT86MultiCameraV1
from services.camera_v11.ui_preview_ipc_v1 import DEFAULT_PATH, PreviewFrameWriter


class V11DeepStreamTRT86MultiCameraUICam01V1(V11DeepStreamTRT86MultiCameraV1):
    """Add one non-blocking CAM-01 UI preview tap without replacing the frozen six-camera runtime."""

    def __init__(self) -> None:
        self.ui_preview_camera = os.environ.get("V11_UI_PREVIEW_CAMERA", "CAM-01").strip() or "CAM-01"
        self.ui_preview_path = os.environ.get("V11_UI_PREVIEW_PATH", DEFAULT_PATH)
        self.ui_preview_hz = max(5.0, min(20.0, float(os.environ.get("V11_UI_PREVIEW_HZ", "15.0"))))
        self.ui_preview_period = 1.0 / self.ui_preview_hz
        self.ui_preview_stop = threading.Event()
        self.ui_preview_thread: threading.Thread | None = None
        self.ui_preview_writer: PreviewFrameWriter | None = None
        self.ui_preview_exported = 0
        self.ui_preview_errors = 0
        self.ui_preview_next_mono = 0.0
        super().__init__()
        if self.ui_preview_camera not in self.states:
            raise RuntimeError(f"UI preview camera not configured: {self.ui_preview_camera}")
        self.ui_preview_writer = PreviewFrameWriter(self.ui_preview_path, self.width, self.height, self.width * 4)
        print(
            "CAMERA_V11_UI_PREVIEW_ARCH "
            f"camera={self.ui_preview_camera} source=post-osd-same-pipeline rtsp_extra=0 "
            f"queue=latest1 fps={self.ui_preview_hz:.1f} transport=raw-bgrx-shm path={self.ui_preview_path}",
            flush=True,
        )

    def _build_camera(self, state) -> None:
        super()._build_camera(state)
        cid = state.camera.camera_id
        if cid != self.ui_preview_camera:
            return
        safe = cid.lower().replace("-", "_")
        pipeline = state.pipeline
        osd = pipeline.get_by_name(f"osd_{safe}")
        sink = state.sink
        if osd is None or sink is None:
            raise RuntimeError(f"{cid}: UI preview could not resolve osd/sink")
        # Gst.Element.unlink() is a void operation; the next link calls are the
        # authoritative failure check. Never interpret unlink() as a bool.
        osd.unlink(sink)

        ui_tee = self._make("tee", f"ui_tee_{safe}")
        ui_q = self._make("queue", f"ui_q_{safe}")
        ui_convert = self._make("nvvideoconvert", f"ui_convert_{safe}")
        ui_caps = self._make("capsfilter", f"ui_caps_{safe}")
        ui_sink = self._make("appsink", f"ui_sink_{safe}")
        self._latest_queue(ui_q)
        self._set_if(ui_convert, "gpu-id", self.gpu_id)
        self._set_if(ui_convert, "compute-hw", 1)
        self._set_if(ui_convert, "interpolation-method", 2)
        ui_caps.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw,format=BGRx,width={self.width},height={self.height},pixel-aspect-ratio=1/1"
            ),
        )
        for prop, value in (
            ("emit-signals", False), ("sync", False), ("async", False), ("drop", True),
            ("max-buffers", 1), ("enable-last-sample", False), ("wait-on-eos", False),
        ):
            self._set_if(ui_sink, prop, value)
        for element in (ui_tee, ui_q, ui_convert, ui_caps, ui_sink):
            pipeline.add(element)
        if not osd.link(ui_tee):
            raise RuntimeError(f"{cid}: osd->ui_tee link failed")
        tee_display = ui_tee.request_pad_simple("src_%u") if hasattr(ui_tee, "request_pad_simple") else ui_tee.get_request_pad("src_%u")
        tee_preview = ui_tee.request_pad_simple("src_%u") if hasattr(ui_tee, "request_pad_simple") else ui_tee.get_request_pad("src_%u")
        sink_pad = sink.get_static_pad("sink")
        ui_q_sink = ui_q.get_static_pad("sink")
        if tee_display is None or tee_preview is None or sink_pad is None or ui_q_sink is None:
            raise RuntimeError(f"{cid}: UI preview tee pad missing")
        if tee_display.link(sink_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: ui_tee->display sink failed")
        if tee_preview.link(ui_q_sink) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: ui_tee->ui_q failed")
        for src, dst, label in (
            (ui_q, ui_convert, "ui_q->convert"),
            (ui_convert, ui_caps, "ui_convert->caps"),
            (ui_caps, ui_sink, "ui_caps->appsink"),
        ):
            if not src.link(dst):
                raise RuntimeError(f"{cid}: UI preview link failed: {label}")
        ui_q_src = ui_q.get_static_pad("src")
        if ui_q_src is None:
            raise RuntimeError(f"{cid}: UI preview queue src missing")
        ui_q_src.add_probe(self.Gst.PadProbeType.BUFFER, self._ui_gate_probe, cid)
        state.ui_sink = ui_sink
        state.ui_q = ui_q
        state.ui_convert = ui_convert
        state.ui_caps = ui_caps
        state.ui_tee = ui_tee

    def _ui_gate_probe(self, _pad, info, cid: str):
        if info.get_buffer() is None or cid != self.ui_preview_camera:
            return self.Gst.PadProbeReturn.DROP
        now = time.monotonic()
        if now < self.ui_preview_next_mono:
            return self.Gst.PadProbeReturn.DROP
        self.ui_preview_next_mono = now + self.ui_preview_period
        return self.Gst.PadProbeReturn.OK

    def _ui_preview_loop(self) -> None:
        state = self.states[self.ui_preview_camera]
        ui_sink = getattr(state, "ui_sink", None)
        writer = self.ui_preview_writer
        if ui_sink is None or writer is None:
            self.ui_preview_errors += 1
            return
        expected = self.width * self.height * 4
        print("CAMERA_V11_UI_PREVIEW_THREAD state=START", flush=True)
        try:
            while not self.ui_preview_stop.is_set():
                sample = ui_sink.emit("try-pull-sample", 100_000_000)
                if sample is None:
                    continue
                buffer = sample.get_buffer()
                if buffer is None:
                    continue
                ok, map_info = buffer.map(self.Gst.MapFlags.READ)
                if not ok:
                    self.ui_preview_errors += 1
                    continue
                try:
                    if len(map_info.data) < expected:
                        raise RuntimeError(f"UI preview buffer too small {len(map_info.data)}<{expected}")
                    now = time.monotonic()
                    with self.lock:
                        snapshot = state.latest_snapshot
                        age = now - snapshot.completed_mono if snapshot.completed_mono > 0 else 999.0
                        object_count = len(snapshot.boxes) if age <= self.box_stale_sec else 0
                    writer.publish(memoryview(map_info.data)[:expected], object_count=object_count)
                    self.ui_preview_exported += 1
                except Exception as exc:
                    self.ui_preview_errors += 1
                    if self.ui_preview_errors <= 5 or self.ui_preview_errors % 100 == 0:
                        print(f"CAMERA_V11_UI_PREVIEW warning={type(exc).__name__}:{exc} errors={self.ui_preview_errors}", flush=True)
                finally:
                    buffer.unmap(map_info)
        finally:
            print(
                "CAMERA_V11_UI_PREVIEW_THREAD "
                f"state=STOP exported={self.ui_preview_exported} errors={self.ui_preview_errors}", flush=True,
            )

    def run(self) -> int:
        self.ui_preview_thread = threading.Thread(target=self._ui_preview_loop, name="v11-ui-preview-cam01", daemon=False)
        self.ui_preview_thread.start()
        return super().run()

    def stop(self) -> bool:
        self.ui_preview_stop.set()
        return super().stop()

    def close(self) -> None:
        self.ui_preview_stop.set()
        if self.ui_preview_thread is not None and self.ui_preview_thread.is_alive():
            self.ui_preview_thread.join(timeout=3.0)
        if self.ui_preview_writer is not None:
            try:
                self.ui_preview_writer.close(unlink=True)
            except Exception:
                pass
            self.ui_preview_writer = None
        super().close()


def main() -> int:
    return V11DeepStreamTRT86MultiCameraUICam01V1().run()


if __name__ == "__main__":
    raise SystemExit(main())
