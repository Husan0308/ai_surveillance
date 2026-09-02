from __future__ import annotations

import gc
import os
import threading
import time

from services.camera_v11.deepstream_trt86_multi_v1 import V11DeepStreamTRT86MultiCameraV1
from services.camera_v11.monitoring_telemetry_ipc_v1 import RuntimeMonitoringTelemetryPublisher
from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameWriter


DEFAULT_UI_CAMERAS = ("CAM-01", "CAM-02")


def _camera_env_key(camera_id: str) -> str:
    return camera_id.upper().replace("-", "")


def _default_preview_path(camera_id: str) -> str:
    slug = camera_id.lower().replace("-", "")
    return f"/dev/shm/v11_ui_preview_{slug}_v1.bin"


class V11DeepStreamTRT86MultiCameraUIV1(V11DeepStreamTRT86MultiCameraV1):
    """Add selected non-blocking UI preview taps to the frozen six-camera runtime."""

    def __init__(self) -> None:
        raw_cameras = os.environ.get("V11_UI_PREVIEW_CAMERAS", ",".join(DEFAULT_UI_CAMERAS))
        requested = tuple(dict.fromkeys(part.strip() for part in raw_cameras.split(",") if part.strip()))
        self.ui_preview_cameras = requested or DEFAULT_UI_CAMERAS
        self.ui_preview_hz = max(5.0, min(20.0, float(os.environ.get("V11_UI_PREVIEW_HZ", "15.0"))))
        self.ui_preview_period = 1.0 / self.ui_preview_hz
        self.ui_preview_stop = threading.Event()
        self.ui_preview_threads: dict[str, threading.Thread] = {}
        self.ui_preview_writers: dict[str, PreviewFrameWriter] = {}
        self.ui_preview_paths: dict[str, str] = {}
        self.ui_preview_exported: dict[str, int] = {cid: 0 for cid in self.ui_preview_cameras}
        self.ui_preview_errors: dict[str, int] = {cid: 0 for cid in self.ui_preview_cameras}
        self.ui_preview_next_mono: dict[str, float] = {cid: 0.0 for cid in self.ui_preview_cameras}
        self.ui_preview_last_mono: dict[str, float] = {cid: 0.0 for cid in self.ui_preview_cameras}
        super().__init__()

        missing = [cid for cid in self.ui_preview_cameras if cid not in self.states]
        if missing:
            raise RuntimeError(f"UI preview cameras not configured: {','.join(missing)}")

        for cid in self.ui_preview_cameras:
            env_key = f"V11_UI_PREVIEW_PATH_{_camera_env_key(cid)}"
            self.ui_preview_paths[cid] = os.environ.get(env_key, _default_preview_path(cid))
        if len(set(self.ui_preview_paths.values())) != len(self.ui_preview_paths):
            raise RuntimeError("UI preview paths must be unique per camera")

        for cid in self.ui_preview_cameras:
            path = self.ui_preview_paths[cid]
            self.ui_preview_writers[cid] = PreviewFrameWriter(
                path, self.width, self.height, self.width * 4
            )
            print(
                "CAMERA_V11_UI_PREVIEW_ARCH "
                f"camera={cid} source=post-osd-same-pipeline rtsp_extra=0 "
                f"queue=latest1 fps={self.ui_preview_hz:.1f} transport=raw-bgrx-shm path={path}",
                flush=True,
            )
        self.monitoring_telemetry = RuntimeMonitoringTelemetryPublisher(self)
        print(
            "CAMERA_V11_MONITORING_TELEMETRY_ARCH "
            f"path={self.monitoring_telemetry.writer.path} "
            f"period_sec={self.monitoring_telemetry.period_sec:.3f} "
            "transport=atomic-json-shm queue=latest1 log_parsing=0",
            flush=True,
        )

    def _build_camera(self, state) -> None:
        super()._build_camera(state)
        cid = state.camera.camera_id
        if cid not in self.ui_preview_cameras:
            return

        safe = cid.lower().replace("-", "_")
        pipeline = state.pipeline
        osd = pipeline.get_by_name(f"osd_{safe}")
        sink = state.sink
        if osd is None or sink is None:
            raise RuntimeError(f"{cid}: UI preview could not resolve osd/sink")

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
            ("emit-signals", False),
            ("sync", False),
            ("async", False),
            ("drop", True),
            ("max-buffers", 1),
            ("enable-last-sample", False),
            ("wait-on-eos", False),
        ):
            self._set_if(ui_sink, prop, value)

        for element in (ui_tee, ui_q, ui_convert, ui_caps, ui_sink):
            pipeline.add(element)
        if not osd.link(ui_tee):
            raise RuntimeError(f"{cid}: osd->ui_tee link failed")

        tee_display = (
            ui_tee.request_pad_simple("src_%u")
            if hasattr(ui_tee, "request_pad_simple")
            else ui_tee.get_request_pad("src_%u")
        )
        tee_preview = (
            ui_tee.request_pad_simple("src_%u")
            if hasattr(ui_tee, "request_pad_simple")
            else ui_tee.get_request_pad("src_%u")
        )
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
        state.ui_tee_display = tee_display
        state.ui_tee_preview = tee_preview

    def _ui_gate_probe(self, _pad, info, cid: str):
        if info.get_buffer() is None or cid not in self.ui_preview_next_mono:
            return self.Gst.PadProbeReturn.DROP
        now = time.monotonic()
        if now < self.ui_preview_next_mono[cid]:
            return self.Gst.PadProbeReturn.DROP
        self.ui_preview_next_mono[cid] = now + self.ui_preview_period
        return self.Gst.PadProbeReturn.OK

    def _ui_preview_loop(self, cid: str) -> None:
        state = self.states[cid]
        ui_sink = getattr(state, "ui_sink", None)
        writer = self.ui_preview_writers.get(cid)
        if ui_sink is None or writer is None:
            self.ui_preview_errors[cid] += 1
            return

        expected = self.width * self.height * 4
        print(f"CAMERA_V11_UI_PREVIEW_THREAD camera={cid} state=START", flush=True)
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
                    self.ui_preview_errors[cid] += 1
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
                    self.ui_preview_exported[cid] += 1
                    self.ui_preview_last_mono[cid] = time.monotonic()
                except Exception as exc:
                    self.ui_preview_errors[cid] += 1
                    errors = self.ui_preview_errors[cid]
                    if errors <= 5 or errors % 100 == 0:
                        print(
                            f"CAMERA_V11_UI_PREVIEW camera={cid} warning={type(exc).__name__}:{exc} errors={errors}",
                            flush=True,
                        )
                finally:
                    buffer.unmap(map_info)
        finally:
            print(
                "CAMERA_V11_UI_PREVIEW_THREAD "
                f"camera={cid} state=STOP exported={self.ui_preview_exported[cid]} errors={self.ui_preview_errors[cid]}",
                flush=True,
            )

    @staticmethod
    def _queue_level(element) -> int:
        try:
            if element is not None and element.find_property("current-level-buffers") is not None:
                return int(element.get_property("current-level-buffers"))
        except Exception:
            pass
        return 0

    def _stats(self) -> bool:
        keep_running = super()._stats()
        now = time.monotonic()
        for cid in self.ui_preview_cameras:
            state = self.states[cid]
            writer = self.ui_preview_writers[cid]
            last_mono = self.ui_preview_last_mono[cid]
            age_ms = (now - last_mono) * 1000.0 if last_mono > 0.0 else -1.0
            thread = self.ui_preview_threads.get(cid)
            print(
                "CAMERA_V11_UI_PREVIEW_STATS "
                f"camera={cid} exported={self.ui_preview_exported[cid]} "
                f"sequence={writer.sequence} errors={self.ui_preview_errors[cid]} "
                f"queue={self._queue_level(getattr(state, 'ui_q', None))} "
                f"age_ms={age_ms:.1f} thread_alive={int(thread is not None and thread.is_alive())} "
                f"file_exists={int(os.path.isfile(self.ui_preview_paths[cid]))} "
                f"width={writer.width} height={writer.height} stride={writer.stride} path={self.ui_preview_paths[cid]}",
                flush=True,
            )
        return keep_running

    def run(self) -> int:
        self.monitoring_telemetry.start()
        for cid in self.ui_preview_cameras:
            thread = threading.Thread(
                target=self._ui_preview_loop,
                args=(cid,),
                name=f"v11-ui-preview-{cid.lower()}",
                daemon=False,
            )
            self.ui_preview_threads[cid] = thread
            thread.start()
        return super().run()

    def stop(self) -> bool:
        self.ui_preview_stop.set()
        return super().stop()

    def _dispose_ui_preview_branches(self) -> None:
        # Keep all temporary Gst references scoped to this helper. Once it
        # returns, gc.collect() can release every additive NVMM converter while
        # CUDA is still alive, before the shared TRT client is closed.
        for cid in self.ui_preview_cameras:
            state = self.states[cid]
            pipeline = state.pipeline
            if pipeline is None:
                continue
            try:
                pipeline.set_state(self.Gst.State.NULL)
                pipeline.get_state(3 * self.Gst.SECOND)
            except Exception:
                pass
            tee = getattr(state, "ui_tee", None)
            if tee is not None:
                for pad_name in ("ui_tee_display", "ui_tee_preview"):
                    pad = getattr(state, pad_name, None)
                    if pad is not None:
                        try:
                            peer = pad.get_peer()
                            if peer is not None:
                                pad.unlink(peer)
                            tee.release_request_pad(pad)
                        except Exception:
                            pass
                        setattr(state, pad_name, None)
            for attr in ("ui_sink", "ui_caps", "ui_convert", "ui_q", "ui_tee"):
                element = getattr(state, attr, None)
                if element is not None:
                    try:
                        element.set_state(self.Gst.State.NULL)
                        pipeline.remove(element)
                    except Exception:
                        pass
                    setattr(state, attr, None)
            print(f"CAMERA_V11_UI_PREVIEW_CLEANUP camera={cid} state=COMPLETE", flush=True)

    def close(self) -> None:
        self.monitoring_telemetry.stop()
        self.ui_preview_stop.set()
        for thread in self.ui_preview_threads.values():
            if thread.is_alive():
                thread.join(timeout=3.0)
        self.ui_preview_threads.clear()
        for cid, writer in list(self.ui_preview_writers.items()):
            try:
                writer.close(unlink=True)
            except Exception:
                pass
            finally:
                self.ui_preview_writers.pop(cid, None)

        self._dispose_ui_preview_branches()
        # PyGObject may defer final unrefs. Collect here, before super().close()
        # destroys the shared TRT/CUDA context, so NvBufSurface never sees
        # cudaErrorCudartUnloading during interpreter teardown.
        gc.collect()
        super().close()


def main() -> int:
    return V11DeepStreamTRT86MultiCameraUIV1().run()


if __name__ == "__main__":
    raise SystemExit(main())
