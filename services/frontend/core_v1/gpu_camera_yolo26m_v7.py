from __future__ import annotations

import os
import sys
import time

# V7 goal: camera motion first, detection second.
# DeepStream handles RTSP negotiation and NVDEC; no manual rtspsrc/depay chain.
os.environ.setdefault("AI_RTSP_LATENCY_MS", "120")
os.environ.setdefault("AI_RTSP_TRANSPORT", "tcp")
os.environ.setdefault("AI_DECODER_EXTRA_SURFACES", "8")
os.environ.setdefault("AI_MUX_TIMEOUT_US", "50000")
os.environ.setdefault("AI_WALL_WIDTH", "1920")
os.environ.setdefault("AI_WALL_HEIGHT", "720")
os.environ.setdefault("AI_WALL_SINK_SYNC", "0")

# Keep YOLO26m + strict six-camera batch, but shorten each CUDA burst.
os.environ.setdefault("AI_YOLO_INFER_WIDTH", "416")
os.environ.setdefault("AI_YOLO_INFER_HEIGHT", "256")
os.environ.setdefault("AI_YOLO_PREDICT_WIDTH", "416")
os.environ.setdefault("AI_YOLO_PREDICT_HEIGHT", "256")
os.environ.setdefault("AI_YOLO_START_BATCH_FPS", "0.55")
os.environ.setdefault("AI_YOLO_MAX_BATCH_FPS", "0.75")
os.environ.setdefault("AI_YOLO_MIN_BATCH_FPS", "0.35")
os.environ.setdefault("AI_YOLO_MAX_GPU_DUTY", "0.12")
os.environ.setdefault("AI_YOLO_CONF", "0.18")
os.environ.setdefault("AI_YOLO_IOU", "0.50")
os.environ.setdefault("AI_YOLO_MAX_DET", "16")
os.environ.setdefault("AI_DETECTION_BOX_HOLD_MS", "3200")
os.environ.setdefault("AI_CAMERA_FPS_FLOOR", "19.5")
os.environ.setdefault("AI_CAMERA_FPS_GOOD", "19.9")

from . import deepstream_yolo26m_batch6_wall as base
from . import deepstream_yolo26m_detection_only as detection


class GpuCameraYolo26mV7(detection.NativeCameraYolo26mDetectionOnly):
    """All-GPU six-camera display with isolated/ticketed YOLO26m side branch.

    Camera hot path:
      RTSP -> nvurisrcbin -> NVDEC/NVMM -> tee -> 2-buffer leaky queue
           -> nvstreammux(sync-inputs=0) -> nvmultistreamtiler
           -> native NvDs object metadata -> nvdsosd -> EGL

    Detection side path:
      decoded tee -> 1-buffer queue -> DROP unless ticketed -> nvvideoconvert
           -> 416x256 BGRx system frame -> exactly six fresh frames
           -> one YOLO26m PyTorch call on the lowest-priority CUDA stream.

    There is no manual H264/H265 negotiation, no CPU decoder, no Qt/UI,
    no tracker/ReID/face/pose/heatmap.
    """

    def __init__(self):
        self.transport_name = os.environ.get("AI_RTSP_TRANSPORT", "tcp").strip().lower()
        if self.transport_name not in {"tcp", "auto"}:
            self.transport_name = "tcp"
        self.low_cuda_priority = None
        super().__init__()

        # Detection-only parent intentionally experimented with synchronized mux.
        # For a camera wall, never let one jittery source stall the other five.
        # nvmultistreamtiler caches an old frame for a source if needed.
        self._set_if(self.mux, "batched-push-timeout", 50000)
        self._set_if(self.mux, "sync-inputs", False)
        self._set_if(self.mux, "max-latency", 0)
        self._set_if(self.mux, "buffer-pool-size", 10)
        self.mux_timeout_us = 50000

        self._set_if(self.tiler, "width", 1920)
        self._set_if(self.tiler, "height", 720)
        self.wall_width = 1920
        self.wall_height = 720

        # The renderer is never allowed to declare a frame late or push QoS drops
        # back into decode. Latest-frame queues already bound display latency.
        self._set_if(self.sink, "sync", False)
        self._set_if(self.sink, "qos", False)
        self._set_if(self.sink, "max-lateness", -1)
        self._set_if(self.sink, "processing-deadline", 0)
        self._set_if(self.sink, "render-delay", 0)
        self._set_if(self.sink, "throttle-time", 0)
        self._set_if(self.sink, "enable-last-sample", False)
        self.sink_sync = False

        # Two display buffers absorb tiny scheduling jitter without building a
        # long latency tail. The inference branches stay at one latest frame.
        for q in self.queues.values():
            self._set_if(q, "max-size-buffers", 2)
            self._set_if(q, "max-size-bytes", 0)
            self._set_if(q, "max-size-time", 0)
            self._set_if(q, "leaky", 2)
        for q in self.infer_queues.values():
            self._set_if(q, "max-size-buffers", 1)
            self._set_if(q, "max-size-bytes", 0)
            self._set_if(q, "max-size-time", 0)
            self._set_if(q, "leaky", 2)

        print(
            "GPU_V7 configured: nvurisrcbin/NVDEC negotiation; "
            f"transport={self.transport_name} latency={self.rtsp_latency_ms}ms; "
            "mux_sync=0 timeout=50000us; wall=1920x720; "
            "YOLO26m strict_batch=6 input=416x256 low-priority CUDA",
            flush=True,
        )

    def _add_source(self, index: int, camera: dict) -> None:
        """Use nvurisrcbin so DeepStream owns RTSP codec negotiation/NVDEC."""
        cid = str(camera["id"])
        uri = base.native.authenticated_source(
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
        # nvurisrcbin values: 0=multi/auto, 4=TCP-only.
        self._set_if(source, "select-rtp-protocol", 4 if self.transport_name == "tcp" else 0)
        self._set_if(source, "latency", self.rtsp_latency_ms)
        self._set_if(source, "drop-on-latency", True)
        self._set_if(source, "low-latency-mode", True)
        self._set_if(source, "num-extra-surfaces", self.extra_surfaces)
        self._set_if(source, "cudadec-memtype", 0)
        self._set_if(source, "udp-buffer-size", self.udp_buffer)
        self._set_if(source, "rtsp-reconnect-interval", 5)
        self._set_if(source, "rtsp-reconnect-attempts", -1)
        self._set_if(source, "async-handling", True)

        self._set_if(display_q, "max-size-buffers", 2)
        self._set_if(display_q, "max-size-bytes", 0)
        self._set_if(display_q, "max-size-time", 0)
        self._set_if(display_q, "leaky", 2)
        self._set_if(infer_q, "max-size-buffers", 1)
        self._set_if(infer_q, "max-size-bytes", 0)
        self._set_if(infer_q, "max-size-time", 0)
        self._set_if(infer_q, "leaky", 2)

        self._set_if(infer_convert, "gpu-id", 0)
        infer_caps.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw,width={base.INFER_WIDTH},height={base.INFER_HEIGHT},format=BGRx"
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

            # A large positive value is clamped to the least-priority stream.
            # Priority is only a scheduler hint; reducing model pixels is the
            # primary mechanism that shortens the visible CUDA burst.
            low_stream = torch.cuda.Stream(device=0, priority=999999)
            self.low_cuda_priority = int(low_stream.priority)

            capability = torch.cuda.get_device_capability(0)
            device_name = torch.cuda.get_device_name(0)
            model_spec = self._resolve_model_spec()
            print(
                "YOLO26M_V7 loading "
                f"model={model_spec} cuda={torch.version.cuda} device={device_name} "
                f"sm={capability[0]}.{capability[1]} input={base.INFER_WIDTH}x{base.INFER_HEIGHT} "
                f"strict_batch=6 cuda_stream_priority={self.low_cuda_priority}",
                flush=True,
            )
            model = YOLO(model_spec)
            predict_kwargs = {
                "imgsz": (base.PREDICT_HEIGHT, base.PREDICT_WIDTH),
                "rect": True,
                "classes": [0],
                "conf": float(os.environ.get("AI_YOLO_CONF", "0.18")),
                "iou": float(os.environ.get("AI_YOLO_IOU", "0.50")),
                "max_det": int(os.environ.get("AI_YOLO_MAX_DET", "16")),
                "device": "cuda:0",
                "verbose": False,
                "stream": False,
            }
            warm = [
                np.zeros((base.INFER_HEIGHT, base.INFER_WIDTH, 3), dtype=np.uint8)
                for _ in self.camera_ids
            ]
            with torch.cuda.stream(low_stream), torch.inference_mode():
                model.predict(source=warm, **predict_kwargs)
            low_stream.synchronize()

            with self.det_lock:
                self.detector_ready = True
                self.detector_started_mono = time.monotonic()
            print(
                "YOLO26M_V7 ready: YOLO26m strict batch=6; ticketed preprocess; "
                f"start_cap={self.batch_fps_cap:.2f}/s max_duty={self.max_gpu_duty:.0%}; "
                "camera hot path remains NVMM",
                flush=True,
            )
        except BaseException as exc:
            with self.det_lock:
                self.detector_error = f"{type(exc).__name__}: {exc}"
            print(
                f"YOLO26M_V7 disabled; camera wall stays alive: {type(exc).__name__}: {exc}",
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
                with torch.cuda.stream(low_stream), torch.inference_mode():
                    predictions = model.predict(source=frames, **predict_kwargs)
                low_stream.synchronize()
                ended = time.monotonic()

                counts = {}
                snapshots = {}
                batch_detections = 0
                for cid, frame, prediction, captured_mono in zip(
                    self.camera_ids, frames, predictions, captured
                ):
                    boxes = getattr(prediction, "boxes", None)
                    items = []
                    if boxes is not None and len(boxes):
                        xyxy = boxes.xyxy.detach().cpu().tolist()
                        confs = boxes.conf.detach().cpu().tolist()
                        for coords, confidence in zip(xyxy, confs):
                            items.append({
                                "xyxy": [float(v) for v in coords],
                                "confidence": float(confidence),
                            })
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
                    f"YOLO26M_V7 batch error: {type(exc).__name__}: {exc}",
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
        # Smoothness controller: camera wins immediately; detector recovers slowly.
        with self.det_lock:
            if not self.detector_ready:
                return
            if min_camera_fps < 19.5:
                self.batch_fps_cap = max(self.batch_fps_min, self.batch_fps_cap * 0.72)
                self.max_gpu_duty = max(0.08, self.max_gpu_duty * 0.85)
            elif min_camera_fps >= 19.9:
                self.batch_fps_cap = min(self.batch_fps_max, self.batch_fps_cap + 0.03)
                self.max_gpu_duty = min(0.12, self.max_gpu_duty + 0.003)

    def _print_stats(self) -> bool:
        result = super()._print_stats()
        print(
            "GPU_V7 "
            f"transport={self.transport_name} mux_sync=0 timeout=50000us "
            f"display_q=2 wall=1920x720 yolo_priority={self.low_cuda_priority} "
            f"input={base.INFER_WIDTH}x{base.INFER_HEIGHT} "
            "manual_rtp_negotiation=0 tracker=0 reid=0 face=0 heatmap=0",
            flush=True,
        )
        return result


def run() -> int:
    return GpuCameraYolo26mV7().run()


if __name__ == "__main__":
    raise SystemExit(run())
