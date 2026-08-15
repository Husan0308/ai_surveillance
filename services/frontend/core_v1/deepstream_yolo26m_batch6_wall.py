from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

from services.ml_service.cameras.gstreamer import owned_bgr_from_mapped
from . import deepstream_native_camera_wall as native

ROOT = Path(__file__).resolve().parents[3]
MODEL_SPEC = os.environ.get("AI_YOLO_MODEL", "yolo26m.pt")
INFER_WIDTH = max(320, int(os.environ.get("AI_YOLO_INFER_WIDTH", "640")))
INFER_HEIGHT = max(180, int(os.environ.get("AI_YOLO_INFER_HEIGHT", "360")))
PREDICT_HEIGHT = max(32, int(os.environ.get("AI_YOLO_PREDICT_HEIGHT", "384")))
PREDICT_WIDTH = max(32, int(os.environ.get("AI_YOLO_PREDICT_WIDTH", "640")))


class LatestRequestedFrames:
    def __init__(self):
        self._condition = threading.Condition()
        self._frames: dict[str, tuple[int, float, object]] = {}
        self._versions: dict[str, int] = {}
        self._closed = False

    def put(self, camera_id: str, captured_mono: float, frame) -> None:
        with self._condition:
            version = self._versions.get(camera_id, 0) + 1
            self._versions[camera_id] = version
            self._frames[camera_id] = (version, float(captured_mono), frame)
            self._condition.notify_all()

    def wait_new_batch(
        self,
        camera_ids: list[str],
        last_versions: dict[str, int],
        timeout: float,
    ):
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while not self._closed:
                ready = all(
                    cid in self._frames
                    and self._frames[cid][0] > last_versions.get(cid, 0)
                    for cid in camera_ids
                )
                if ready:
                    return [self._frames[cid] for cid in camera_ids]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
        return None

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class NativeCameraYolo26mBatch6Wall(native.NativeCameraWall):
    """Keep the proven native camera wall hot path and add YOLO26m as a sidecar.

    Display path (unchanged from the stable baseline):
        RTSP -> nvurisrcbin/NVDEC -> tee -> latest-only display queue
             -> nvstreammux -> nvmultistreamtiler -> EGL

    Detector side path (non-blocking and ticketed):
        tee -> latest-only infer queue -> DROP unless one capture ticket exists
            -> nvvideoconvert -> 640x360 BGRx appsink
            -> exactly six fresh frames -> one YOLO26m PyTorch CUDA batch

    Only one inference frame per camera is converted/copied for each detector
    batch. All other inference-branch frames are dropped before nvvideoconvert,
    so the display path keeps the same cadence/latency as the camera-only wall.
    """

    def __init__(self):
        self._capture_lock = threading.Lock()
        self._capture_requested: dict[str, bool] = {}
        self._tee_request_pads = []
        self.infer_queues = {}
        self.infer_converters = {}
        self.infer_sinks = {}
        self.latest: LatestRequestedFrames | None = None
        self.stop_event = threading.Event()
        self.infer_thread: threading.Thread | None = None

        self.det_lock = threading.Lock()
        self.detector_ready = False
        self.detector_error = ""
        self.batch_calls = 0
        self.batch_inputs = 0
        self.batch_errors = 0
        self.capture_timeouts = 0
        self.last_batch_ms = 0.0
        self.last_batch_age_ms = 0.0
        self.last_capture_spread_ms = 0.0
        self.total_detections = 0
        self.last_counts: dict[str, int] = {}
        self.latest_detections: dict[str, dict] = {}
        self.detector_started_mono = 0.0
        self.last_min_camera_fps = 0.0

        self.batch_fps_min = max(0.25, float(os.environ.get("AI_YOLO_MIN_BATCH_FPS", "0.50")))
        self.batch_fps_max = max(self.batch_fps_min, float(os.environ.get("AI_YOLO_MAX_BATCH_FPS", "2.00")))
        self.batch_fps_cap = min(
            self.batch_fps_max,
            max(self.batch_fps_min, float(os.environ.get("AI_YOLO_START_BATCH_FPS", "1.25"))),
        )
        self.max_gpu_duty = min(0.80, max(0.10, float(os.environ.get("AI_YOLO_MAX_GPU_DUTY", "0.35"))))
        self.camera_fps_floor = max(1.0, float(os.environ.get("AI_CAMERA_FPS_FLOOR", "18.5")))
        self.camera_fps_good = max(self.camera_fps_floor, float(os.environ.get("AI_CAMERA_FPS_GOOD", "19.5")))
        self.capture_timeout = max(0.10, float(os.environ.get("AI_YOLO_CAPTURE_TIMEOUT", "0.35")))
        self.startup_delay = max(0.0, float(os.environ.get("AI_YOLO_STARTUP_DELAY", "2.0")))

        super().__init__()
        self.camera_ids = [str(camera["id"]) for camera in self.cameras]
        if len(self.camera_ids) != 6:
            raise RuntimeError(f"YOLO26m batch6 mode requires exactly 6 cameras, found {len(self.camera_ids)}")
        self.latest = LatestRequestedFrames()

    def _request_tee_pad(self, tee):
        request_simple = getattr(tee, "request_pad_simple", None)
        pad = request_simple("src_%u") if request_simple else None
        if pad is None:
            pad = tee.get_request_pad("src_%u")
        if pad is None:
            raise RuntimeError(f"{tee.get_name()}: failed to allocate tee src pad")
        self._tee_request_pads.append((tee, pad))
        return pad

    def _link_tee_branch(self, tee, queue, cid: str, branch: str) -> None:
        src_pad = self._request_tee_pad(tee)
        sink_pad = queue.get_static_pad("sink")
        if src_pad.link(sink_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: tee -> {branch} queue failed")

    def _add_source(self, index: int, camera: dict) -> None:
        cid = str(camera["id"])
        uri = native.authenticated_source(
            {**camera, "source": camera.get("display_source") or camera.get("source")}
        )
        if not isinstance(uri, str) or not uri.startswith(("rtsp://", "rtsps://")):
            raise RuntimeError(f"{cid}: invalid RTSP source")

        source = self._make("nvurisrcbin", f"src_{index}")
        tee = self._make("tee", f"tee_{index}")
        display_q = self._make("queue", f"src_queue_{index}")
        infer_q = self._make("queue", f"infer_queue_{index}")
        infer_convert = self._make("nvvideoconvert", f"infer_convert_{index}")
        infer_caps = self._make("capsfilter", f"infer_caps_{index}")
        appsink = self._make("appsink", f"infer_sink_{index}")

        source.set_property("uri", uri)
        self._set_if(source, "disable-audio", True)
        self._set_if(source, "select-rtp-protocol", 0)
        self._set_if(source, "latency", self.rtsp_latency_ms)
        self._set_if(source, "drop-on-latency", True)
        self._set_if(source, "low-latency-mode", True)
        self._set_if(source, "num-extra-surfaces", self.extra_surfaces)
        self._set_if(source, "cudadec-memtype", 0)
        self._set_if(source, "udp-buffer-size", self.udp_buffer)
        self._set_if(source, "rtsp-reconnect-interval", 5)
        self._set_if(source, "rtsp-reconnect-attempts", -1)

        for q in (display_q, infer_q):
            self._set_if(q, "max-size-buffers", 1)
            self._set_if(q, "max-size-bytes", 0)
            self._set_if(q, "max-size-time", 0)
            self._set_if(q, "leaky", 2)

        self._set_if(infer_convert, "gpu-id", 0)
        infer_caps.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw,width={INFER_WIDTH},height={INFER_HEIGHT},format=BGRx"
            ),
        )
        self._set_if(appsink, "drop", True)
        self._set_if(appsink, "max-buffers", 1)
        self._set_if(appsink, "sync", False)
        self._set_if(appsink, "emit-signals", True)
        self._set_if(appsink, "wait-on-eos", False)
        self._set_if(appsink, "enable-last-sample", False)

        for element in (source, tee, display_q, infer_q, infer_convert, infer_caps, appsink):
            self.pipeline.add(element)

        self._link_tee_branch(tee, display_q, cid, "display")
        self._link_tee_branch(tee, infer_q, cid, "infer")

        mux_pad = self._request_mux_pad(index)
        display_src = display_q.get_static_pad("src")
        if display_src.link(mux_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: display queue -> streammux failed")

        if not infer_q.link(infer_convert):
            raise RuntimeError(f"{cid}: infer queue -> nvvideoconvert failed")
        if not infer_convert.link(infer_caps):
            raise RuntimeError(f"{cid}: nvvideoconvert -> caps failed")
        if not infer_caps.link(appsink):
            raise RuntimeError(f"{cid}: caps -> appsink failed")

        display_src.add_probe(self.Gst.PadProbeType.BUFFER, self._source_probe, cid)
        infer_q.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._infer_gate_probe,
            cid,
        )
        appsink.connect("new-sample", self._on_infer_sample, cid)
        source.connect("pad-added", self._on_yolo_source_pad_added, tee, cid)

        self._capture_requested[cid] = False
        self.queues[cid] = display_q
        self.infer_queues[cid] = infer_q
        self.infer_converters[cid] = infer_convert
        self.infer_sinks[cid] = appsink

    def _on_yolo_source_pad_added(self, _source, pad, tee, cid: str) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            return
        structure = caps.get_structure(0)
        if not str(structure.get_name()).startswith("video/"):
            return
        self.stats[cid].caps_text = caps.to_string()
        print(f"NATIVE_CAMERA {cid} decoded_caps={caps.to_string()}", flush=True)
        sinkpad = tee.get_static_pad("sink")
        if sinkpad.is_linked():
            return
        result = pad.link(sinkpad)
        if result != self.Gst.PadLinkReturn.OK:
            print(f"YOLO26M_BATCH6 {cid} source -> tee failed: {result}", flush=True)

    def _infer_gate_probe(self, _pad, _info, cid: str):
        with self._capture_lock:
            if not self._capture_requested.get(cid, False):
                return self.Gst.PadProbeReturn.DROP
            self._capture_requested[cid] = False
        return self.Gst.PadProbeReturn.OK

    def _retry_capture(self, cid: str) -> None:
        with self._capture_lock:
            self._capture_requested[cid] = True

    def _request_capture_batch(self) -> None:
        with self._capture_lock:
            for cid in self.camera_ids:
                self._capture_requested[cid] = True

    def _clear_capture_requests(self) -> None:
        with self._capture_lock:
            for cid in self.camera_ids:
                self._capture_requested[cid] = False

    def _on_infer_sample(self, sink, cid: str):
        sample = sink.emit("pull-sample")
        if sample is None:
            self._retry_capture(cid)
            return self.Gst.FlowReturn.OK

        caps = sample.get_caps().get_structure(0)
        width = int(caps.get_value("width"))
        height = int(caps.get_value("height"))
        fmt = str(caps.get_value("format"))
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            self._retry_capture(cid)
            return self.Gst.FlowReturn.OK
        try:
            frame = owned_bgr_from_mapped(mapped.data, width, height, fmt)
        except Exception:
            self._retry_capture(cid)
            return self.Gst.FlowReturn.OK
        finally:
            buffer.unmap(mapped)

        latest = self.latest
        if latest is not None:
            latest.put(cid, time.monotonic(), frame)
        return self.Gst.FlowReturn.OK

    def _resolve_model_spec(self) -> str:
        candidate = Path(MODEL_SPEC)
        if candidate.is_file():
            return str(candidate)
        rooted = ROOT / MODEL_SPEC
        if rooted.is_file():
            return str(rooted)
        return MODEL_SPEC

    def _wait_for_camera_baseline(self) -> bool:
        if self.stop_event.wait(self.startup_delay):
            return False
        deadline = time.monotonic() + 6.0
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            if all(self.stats[cid].frames >= 10 for cid in self.camera_ids):
                return True
            self.stop_event.wait(0.10)
        return not self.stop_event.is_set()

    def _infer_loop(self) -> None:
        if not self._wait_for_camera_baseline():
            return

        try:
            import numpy as np
            import torch
            from ultralytics import YOLO

            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch CUDA is not available")
            torch.cuda.set_device(0)
            try:
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            torch.backends.cudnn.benchmark = True

            capability = torch.cuda.get_device_capability(0)
            device_name = torch.cuda.get_device_name(0)
            model_spec = self._resolve_model_spec()
            print(
                "YOLO26M_BATCH6 loading "
                f"model={model_spec} cuda={torch.version.cuda} device={device_name} "
                f"sm={capability[0]}.{capability[1]} infer_copy={INFER_WIDTH}x{INFER_HEIGHT} "
                f"predict={PREDICT_WIDTH}x{PREDICT_HEIGHT}",
                flush=True,
            )
            model = YOLO(model_spec)
            predict_kwargs = {
                "imgsz": (PREDICT_HEIGHT, PREDICT_WIDTH),
                "rect": True,
                "classes": [0],
                "conf": float(os.environ.get("AI_YOLO_CONF", "0.15")),
                "iou": float(os.environ.get("AI_YOLO_IOU", "0.50")),
                "max_det": int(os.environ.get("AI_YOLO_MAX_DET", "20")),
                "device": "cuda:0",
                "verbose": False,
                "stream": False,
            }
            warm = [
                np.zeros((INFER_HEIGHT, INFER_WIDTH, 3), dtype=np.uint8)
                for _ in self.camera_ids
            ]
            with torch.inference_mode():
                model.predict(source=warm, **predict_kwargs)

            with self.det_lock:
                self.detector_ready = True
                self.detector_started_mono = time.monotonic()
            print(
                "YOLO26M_BATCH6 ready: strict 6-camera batch; ticketed preprocess; "
                f"start_cap={self.batch_fps_cap:.2f}/s max_duty={self.max_gpu_duty:.0%}; "
                "display hot path unchanged",
                flush=True,
            )
        except BaseException as exc:
            with self.det_lock:
                self.detector_error = f"{type(exc).__name__}: {exc}"
            print(
                f"YOLO26M_BATCH6 disabled but camera wall stays alive: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return

        latest = self.latest
        if latest is None:
            return

        last_versions = {cid: 0 for cid in self.camera_ids}
        next_batch_allowed = time.monotonic()

        while not self.stop_event.is_set():
            now = time.monotonic()
            if now < next_batch_allowed:
                if self.stop_event.wait(min(0.05, next_batch_allowed - now)):
                    break
                continue

            self._request_capture_batch()
            rows = latest.wait_new_batch(
                self.camera_ids,
                last_versions,
                timeout=self.capture_timeout,
            )
            self._clear_capture_requests()
            if rows is None:
                with self.det_lock:
                    self.capture_timeouts += 1
                next_batch_allowed = time.monotonic() + 0.10
                continue

            frames = []
            captured = []
            versions = {}
            for cid, row in zip(self.camera_ids, rows):
                version, captured_mono, frame = row
                versions[cid] = int(version)
                captured.append(float(captured_mono))
                frames.append(frame)
            last_versions.update(versions)

            started = time.monotonic()
            try:
                with torch.inference_mode():
                    predictions = model.predict(source=frames, **predict_kwargs)

                counts = {}
                snapshots = {}
                batch_detections = 0
                ended = time.monotonic()
                for cid, frame, prediction, captured_mono in zip(
                    self.camera_ids, frames, predictions, captured
                ):
                    boxes = getattr(prediction, "boxes", None)
                    items = []
                    if boxes is not None and len(boxes):
                        xyxy = boxes.xyxy.detach().cpu().tolist()
                        confs = boxes.conf.detach().cpu().tolist()
                        for coords, confidence in zip(xyxy, confs):
                            items.append(
                                {
                                    "xyxy": [float(v) for v in coords],
                                    "confidence": float(confidence),
                                }
                            )
                    counts[cid] = len(items)
                    batch_detections += len(items)
                    snapshots[cid] = {
                        "captured_mono": captured_mono,
                        "age_ms": max(0.0, (ended - captured_mono) * 1000.0),
                        "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
                        "boxes": items,
                    }
            except BaseException as exc:
                ended = time.monotonic()
                with self.det_lock:
                    self.batch_errors += 1
                    if "out of memory" in str(exc).lower():
                        self.batch_fps_cap = self.batch_fps_min
                print(
                    f"YOLO26M_BATCH6 batch error: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if "out of memory" in str(exc).lower():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                next_batch_allowed = ended + 0.50
                continue

            batch_ms = max(0.0, (ended - started) * 1000.0)
            age_ms = max(0.0, (ended - min(captured)) * 1000.0)
            spread_ms = max(0.0, (max(captured) - min(captured)) * 1000.0)

            with self.det_lock:
                self.batch_calls += 1
                self.batch_inputs += len(frames)
                self.total_detections += batch_detections
                self.last_batch_ms = batch_ms
                self.last_batch_age_ms = age_ms
                self.last_capture_spread_ms = spread_ms
                self.last_counts = counts
                self.latest_detections = snapshots
                cap = self.batch_fps_cap

            cap_interval = 1.0 / max(self.batch_fps_min, cap)
            duty_interval = (batch_ms / 1000.0) / self.max_gpu_duty if batch_ms > 0 else 0.0
            next_batch_allowed = max(ended, started + max(cap_interval, duty_interval))

    def _adapt_detector_rate(self, min_camera_fps: float) -> None:
        with self.det_lock:
            if not self.detector_ready:
                return
            old = self.batch_fps_cap
            if min_camera_fps < self.camera_fps_floor:
                self.batch_fps_cap = max(self.batch_fps_min, old * 0.80)
            elif min_camera_fps >= self.camera_fps_good:
                self.batch_fps_cap = min(self.batch_fps_max, old + 0.10)

    def _print_stats(self) -> bool:
        now = time.monotonic()
        parts = []
        fps_values = []
        for cid, stat in self.stats.items():
            elapsed = max(0.001, now - stat.last_print)
            fps = (stat.frames - stat.last_frames) / elapsed
            stat.last_frames = stat.frames
            stat.last_print = now
            q = int(self.queues[cid].get_property("current-level-buffers"))
            pts = f"{stat.pts_ms_ema:.1f}ms" if stat.pts_ms_ema is not None else "?"
            parts.append(f"{cid}:{fps:.1f}fps pts={pts} q={q}")
            fps_values.append(fps)
        min_camera_fps = min(fps_values) if fps_values else 0.0
        self.last_min_camera_fps = min_camera_fps
        print("NATIVE_CAMERA " + " | ".join(parts), flush=True)

        self._adapt_detector_rate(min_camera_fps)
        with self.det_lock:
            ready = self.detector_ready
            error = self.detector_error
            calls = self.batch_calls
            inputs = self.batch_inputs
            errors = self.batch_errors
            timeouts = self.capture_timeouts
            batch_ms = self.last_batch_ms
            age_ms = self.last_batch_age_ms
            spread_ms = self.last_capture_spread_ms
            cap = self.batch_fps_cap
            counts = dict(self.last_counts)
            started = self.detector_started_mono

        elapsed = max(0.001, now - started) if started else 1.0
        batch_rate = calls / elapsed if ready else 0.0
        input_rate = inputs / elapsed if ready else 0.0
        duty_est = min(1.0, batch_rate * batch_ms / 1000.0) if batch_ms > 0 else 0.0
        count_text = " ".join(f"{cid}:{counts.get(cid, 0)}" for cid in self.camera_ids)
        print(
            "YOLO26M_BATCH6 "
            f"ready={int(ready)} batches={batch_rate:.2f}/s inputs={input_rate:.1f}/s "
            f"batch={batch_ms:.1f}ms age={age_ms:.1f}ms spread={spread_ms:.1f}ms "
            f"cap={cap:.2f}/s duty~{duty_est:.0%} min_cam={min_camera_fps:.1f}fps "
            f"timeouts={timeouts} errors={errors} persons=[{count_text}]"
            + (f" error={error}" if error else ""),
            flush=True,
        )
        return True

    def run(self) -> int:
        result = self.pipeline.set_state(self.Gst.State.PLAYING)
        if result == self.Gst.StateChangeReturn.FAILURE:
            self.pipeline.set_state(self.Gst.State.NULL)
            raise RuntimeError("native camera + YOLO26m pipeline failed to PLAY")

        self.infer_thread = threading.Thread(
            target=self._infer_loop,
            name="yolo26m-batch6-sidecar",
            daemon=True,
        )
        self.infer_thread.start()

        print(
            "NATIVE_CAMERA+YOLO26M started: display baseline preserved; "
            "YOLO sidecar uses ticketed frames and strict batch=6",
            flush=True,
        )
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_event.set()
            self._clear_capture_requests()
            if self.latest is not None:
                self.latest.close()
            self.pipeline.set_state(self.Gst.State.NULL)
            if self.infer_thread is not None:
                self.infer_thread.join(timeout=2.0)
        return 0


def run() -> int:
    return NativeCameraYolo26mBatch6Wall().run()


if __name__ == "__main__":
    raise SystemExit(run())
