from __future__ import annotations

import multiprocessing as mp
import queue as pyqueue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

from .camera_core import SixCameraCore
from .core_v1_visual_adapter import CoreV1VisualAdapter
from .native_boxes import NativeBoxBridge
from .rfdetr_worker_v2 import rfdetr_worker_v2

ROOT = Path(__file__).resolve().parents[3]
DETECTOR_CONFIG = ROOT / "config" / "vision_v3_detector.yaml"


def _load_detector_config() -> dict:
    with DETECTOR_CONFIG.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    cfg = dict(payload.get("detector") or {})
    cfg["box"] = dict(cfg.get("box") or {})
    return cfg


class FreshFrameMailbox:
    """Exactly one newest detector frame per camera."""

    def __init__(self) -> None:
        self.cv = threading.Condition()
        self.rows: dict[str, tuple[int, float, np.ndarray]] = {}
        self.versions: dict[str, int] = {}
        self.closed = False

    def put(self, camera_id: str, captured: float, frame: np.ndarray) -> None:
        with self.cv:
            version = self.versions.get(camera_id, 0) + 1
            self.versions[camera_id] = version
            self.rows[camera_id] = (version, captured, frame)
            self.cv.notify_all()

    def wait_group(self, camera_ids: list[str], old: dict[str, int], timeout: float):
        deadline = time.monotonic() + timeout
        with self.cv:
            while not self.closed:
                if all(
                    camera_id in self.rows
                    and self.rows[camera_id][0] > old.get(camera_id, 0)
                    for camera_id in camera_ids
                ):
                    return [self.rows[camera_id] for camera_id in camera_ids]
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self.cv.wait(remaining)
        return None

    def close(self) -> None:
        with self.cv:
            self.closed = True
            self.cv.notify_all()


class SixCameraRFDETR(SixCameraCore):
    """Six smooth DeepStream cameras + asynchronous RF-DETR-S person detection.

    Camera ownership remains in :class:`SixCameraCore`.  This subclass only adds a
    latest-only inference tap and OSD metadata.  Detection policy comes from the
    mature Core-v1 implementation: high-recall raw observations, ROI recovery,
    hard masks, full/ROI fusion, and adaptive Kalman/Byte presentation tracking.
    """

    def __init__(self) -> None:
        self.det_cfg = _load_detector_config()
        self.capture_width = max(320, int(self.det_cfg.get("capture_width", 768)))
        self.capture_height = max(192, int(self.det_cfg.get("capture_height", 432)))
        self.micro_batch = max(1, min(2, int(self.det_cfg.get("micro_batch", 1))))
        self.max_result_age_ms = max(0.0, float(self.det_cfg.get("max_result_age_ms", 900.0)))

        self.capture_lock = threading.Lock()
        self.capture_requested: dict[str, bool] = {}
        self.mailbox = FreshFrameMailbox()
        self.camera_index: dict[str, int] = {}
        self.tee_request_pads: list[tuple[object, object]] = []

        self.det_stop = threading.Event()
        self.det_lock = threading.RLock()
        self.det_ready = False
        self.det_error = ""
        self.det_calls = 0
        self.det_inputs = 0
        self.det_batch_ms = 0.0
        self.det_counts: dict[str, int] = {}
        self.det_finish_age_ms: dict[str, float] = {}
        self.capture_timeouts = 0
        self.stale_result_drops = 0
        self.meta_boxes = 0
        self.roi_inputs = 0
        self.roi_variants = 0
        self.hard_rejects = 0

        self.det_duty = float(self.det_cfg.get("gpu_duty", 0.30))
        self.det_duty_min = float(self.det_cfg.get("gpu_duty_min", 0.12))
        self.det_duty_max = float(self.det_cfg.get("gpu_duty_max", 0.45))
        self.wall_intervals_ms: deque[float] = deque(maxlen=240)
        self.wall_last_mono: float | None = None

        self.worker = None
        self.scheduler_thread = None
        self.job_q = None
        self.result_q = None
        self._detector_cleanup_done = False

        super().__init__()
        self.boxes = CoreV1VisualAdapter(
            self.working_width,
            self.working_height,
            self.det_cfg.get("box") or {},
        )
        self.bridge = NativeBoxBridge()
        self._install_osd()

    def _preflight(self) -> None:
        super()._preflight()
        required = ("tee", "nvvideoconvert", "appsink", "capsfilter", "nvdsosd")
        missing = [name for name in required if self.Gst.ElementFactory.find(name) is None]
        if missing:
            raise RuntimeError(
                "Missing detector GStreamer/DeepStream plugins: " + ", ".join(missing)
            )

    def _add_camera(self, index, camera) -> None:
        camera_id = camera.camera_id
        self.camera_index[camera_id] = index
        self.capture_requested[camera_id] = False

        source = self._make("nvurisrcbin", f"v3_source_{index}")
        tee = self._make("tee", f"v3_detect_tee_{index}")
        display_queue = self._make("queue", f"v3_source_queue_{index}")
        infer_queue = self._make("queue", f"v3_detect_queue_{index}")
        converter = self._make("nvvideoconvert", f"v3_detect_convert_{index}")
        capsfilter = self._make("capsfilter", f"v3_detect_caps_{index}")
        appsink = self._make("appsink", f"v3_detect_sink_{index}")

        source.connect("deep-element-added", self._configure_rtsp_child, camera)
        source.set_property("uri", camera.uri)
        self._set_if(source, "disable-audio", True)
        self._set_if(source, "select-rtp-protocol", 0)
        self._set_if(source, "latency", self.rtsp_latency_ms)
        self._set_if(source, "drop-on-latency", True)
        self._set_if(source, "low-latency-mode", self.low_latency_mode)
        self._set_if(source, "num-extra-surfaces", self.extra_surfaces)
        self._set_if(source, "cudadec-memtype", 0)
        self._set_if(source, "udp-buffer-size", self.udp_buffer_size)
        self._set_if(source, "rtsp-reconnect-interval", 2)
        self._set_if(source, "rtsp-reconnect-attempts", -1)
        self._set_if(source, "message-forward", True)
        self._set_if(source, "gpu-id", self.gpu_id)

        self._configure_latest_queue(display_queue)
        self._configure_latest_queue(infer_queue)
        self._set_if(converter, "gpu-id", self.gpu_id)
        capsfilter.set_property(
            "caps",
            self.Gst.Caps.from_string(
                "video/x-raw,format=BGRx,"
                f"width={self.capture_width},height={self.capture_height},"
                "pixel-aspect-ratio=1/1"
            ),
        )
        appsink.set_property("emit-signals", True)
        appsink.set_property("sync", False)
        appsink.set_property("drop", True)
        appsink.set_property("max-buffers", 1)
        self._set_if(appsink, "enable-last-sample", False)
        self._set_if(appsink, "wait-on-eos", False)

        for element in (
            source,
            tee,
            display_queue,
            infer_queue,
            converter,
            capsfilter,
            appsink,
        ):
            self.pipeline.add(element)

        mux_pad = self._request_mux_pad(index)
        if display_queue.get_static_pad("src").link(mux_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{camera_id}: display queue -> nvstreammux failed")

        tee_display = tee.request_pad_simple("src_%u")
        tee_infer = tee.request_pad_simple("src_%u")
        if tee_display is None or tee_infer is None:
            raise RuntimeError(f"{camera_id}: tee request pad failed")
        if (
            tee_display.link(display_queue.get_static_pad("sink"))
            != self.Gst.PadLinkReturn.OK
        ):
            raise RuntimeError(f"{camera_id}: tee -> display queue failed")
        if tee_infer.link(infer_queue.get_static_pad("sink")) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{camera_id}: tee -> detector queue failed")
        self.tee_request_pads.extend([(tee, tee_display), (tee, tee_infer)])

        if (
            not infer_queue.link(converter)
            or not converter.link(capsfilter)
            or not capsfilter.link(appsink)
        ):
            raise RuntimeError(f"{camera_id}: detector capture branch link failed")

        # Ticket gate comes before conversion/CPU mapping.  Unless this camera has
        # been selected for the next detector job, the inference branch drops the
        # buffer immediately and performs no host copy.
        infer_queue.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._infer_gate_probe,
            camera_id,
        )
        appsink.connect("new-sample", self._on_infer_sample, camera_id)
        source.connect("pad-added", self._source_to_tee, tee, camera_id)
        display_queue.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._source_probe,
            camera_id,
        )

        self.sources[camera_id] = source
        self.queues[camera_id] = display_queue

    def _source_to_tee(self, _source, pad, tee, camera_id: str) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            return
        if not str(caps.get_structure(0).get_name()).startswith("video/"):
            return
        sink = tee.get_static_pad("sink")
        if sink.is_linked():
            return
        result = pad.link(sink)
        if result != self.Gst.PadLinkReturn.OK:
            print(
                f"V3_RFDETR {camera_id} source -> tee failed: {result}",
                file=sys.stderr,
                flush=True,
            )

    def _infer_gate_probe(self, _pad, _info, camera_id: str):
        with self.capture_lock:
            if not self.capture_requested.get(camera_id, False):
                return self.Gst.PadProbeReturn.DROP
            self.capture_requested[camera_id] = False
        return self.Gst.PadProbeReturn.OK

    def _on_infer_sample(self, sink, camera_id: str):
        sample = sink.emit("pull-sample")
        if sample is None:
            return self.Gst.FlowReturn.OK
        structure = sample.get_caps().get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            return self.Gst.FlowReturn.OK
        try:
            needed = width * height * 4
            raw = np.frombuffer(mapped.data, dtype=np.uint8, count=needed)
            frame = raw.reshape((height, width, 4))[..., :3].copy()
        finally:
            buffer.unmap(mapped)
        self.mailbox.put(camera_id, time.monotonic(), frame)
        return self.Gst.FlowReturn.OK

    def _install_osd(self) -> None:
        # PyGObject Gst.Element.unlink() returns None even on success.  Verify the
        # actual pad peer instead of treating that return value as a boolean.
        wall_src = self.wall_queue.get_static_pad("src")
        sink_pad = self.sink.get_static_pad("sink")
        if wall_src is None or sink_pad is None:
            raise RuntimeError("camera-core display pads are unavailable")
        peer = wall_src.get_peer()
        if peer is not None:
            if peer != sink_pad:
                raise RuntimeError("camera-core wall queue is linked unexpectedly")
            wall_src.unlink(peer)
        if wall_src.is_linked() or sink_pad.is_linked():
            raise RuntimeError("could not detach camera-core sink for RF-DETR OSD")

        convert = self._make("nvvideoconvert", "v3_detect_wall_convert")
        caps = self._make("capsfilter", "v3_detect_wall_caps")
        osd = self._make("nvdsosd", "v3_detect_osd")
        self._set_if(convert, "gpu-id", self.gpu_id)
        caps.set_property(
            "caps",
            self.Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"),
        )
        self._set_if(osd, "process-mode", 1)
        self._set_if(osd, "display-bbox", True)
        self._set_if(osd, "display-text", False)
        self._set_if(osd, "display-mask", False)
        self._set_if(osd, "gpu-id", self.gpu_id)

        for element in (convert, caps, osd):
            self.pipeline.add(element)
        if not self.wall_queue.link(convert):
            raise RuntimeError("failed wall queue -> nvvideoconvert")
        if not convert.link(caps):
            raise RuntimeError("failed nvvideoconvert -> RGBA caps")
        if not caps.link(osd):
            raise RuntimeError("failed RGBA caps -> nvdsosd")
        if not osd.link(self.sink):
            raise RuntimeError("failed nvdsosd -> nveglglessink")

        mux_src = self.mux.get_static_pad("src")
        osd_src = osd.get_static_pad("src")
        if mux_src is None or osd_src is None:
            raise RuntimeError("RF-DETR OSD probe pads are unavailable")
        mux_src.add_probe(self.Gst.PadProbeType.BUFFER, self._inject_boxes_probe)
        osd_src.add_probe(self.Gst.PadProbeType.BUFFER, self._wall_probe)
        self.osd = osd

    def _inject_boxes_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK
        now = time.monotonic()
        added = 0
        for camera_id, source_id in self.camera_index.items():
            rows = self.boxes.render(camera_id, now)
            if not rows:
                continue
            result = self.bridge.add_person_boxes(buffer, source_id, rows)
            if result > 0:
                added += result
        with self.det_lock:
            self.meta_boxes += added
        return self.Gst.PadProbeReturn.OK

    def _wall_probe(self, _pad, _info):
        now = time.monotonic()
        if self.wall_last_mono is not None:
            delta_ms = (now - self.wall_last_mono) * 1000.0
            if 1.0 < delta_ms < 1000.0:
                self.wall_intervals_ms.append(delta_ms)
        self.wall_last_mono = now
        return self.Gst.PadProbeReturn.OK

    def _request_group(self, camera_ids: list[str]) -> None:
        with self.capture_lock:
            for camera_id in camera_ids:
                self.capture_requested[camera_id] = True

    def _clear_requests(self) -> None:
        with self.capture_lock:
            for camera_id in self.capture_requested:
                self.capture_requested[camera_id] = False

    def _scaled_detections(self, rows):
        sx = self.working_width / float(self.capture_width)
        sy = self.working_height / float(self.capture_height)
        output = []
        for coords, confidence in rows:
            x1, y1, x2, y2 = coords
            output.append(
                (
                    (x1 * sx, y1 * sy, x2 * sx, y2 * sy),
                    float(confidence),
                )
            )
        return output

    def _scheduler(self) -> None:
        assert self.result_q is not None and self.job_q is not None
        startup_timeout = float(self.det_cfg.get("startup_timeout_sec", 90))
        try:
            ready = self.result_q.get(timeout=startup_timeout)
        except pyqueue.Empty:
            with self.det_lock:
                self.det_error = "RF-DETR-S startup timeout"
            return
        if ready.get("type") != "ready":
            with self.det_lock:
                self.det_error = ready.get("error", "RF-DETR-S worker failed")
            return

        with self.det_lock:
            self.det_ready = True
        print(
            "V3_RFDETR ready "
            f"model={ready.get('model')} device={ready.get('device')} cuda={ready.get('cuda')} "
            f"policy={ready.get('policy', 'core-v1')} micro_batch={self.micro_batch} "
            f"capture={self.capture_width}x{self.capture_height} "
            "adaptive_kalman_byte=1 roi_recovery=1 raw_boxes_preserved=1",
            flush=True,
        )

        ids = [camera.camera_id for camera in self.cameras]
        groups = [ids[i : i + self.micro_batch] for i in range(0, len(ids), self.micro_batch)]
        versions = {camera_id: 0 for camera_id in ids}
        group_index = 0
        result_timeout = float(self.det_cfg.get("result_timeout_sec", 12))

        while not self.det_stop.is_set():
            group = groups[group_index % len(groups)]
            group_index += 1
            self._request_group(group)
            rows = self.mailbox.wait_group(group, versions, timeout=1.5)
            if rows is None:
                self._clear_requests()
                with self.det_lock:
                    self.capture_timeouts += 1
                self.det_stop.wait(0.10)
                continue

            frames = []
            captured = []
            for camera_id, row in zip(group, rows):
                version, captured_t, frame = row
                versions[camera_id] = version
                captured.append(captured_t)
                frames.append(frame)
            self._clear_requests()

            try:
                # One in-flight job, Queue(maxsize=1): stale detector backlog cannot
                # form even if RF-DETR or an ROI recovery pass becomes temporarily slow.
                self.job_q.put(
                    {"cameras": group, "frames": frames, "captured": captured},
                    timeout=0.5,
                )
                result = self.result_q.get(timeout=result_timeout)
            except pyqueue.Empty:
                with self.det_lock:
                    self.det_error = "RF-DETR-S result timeout"
                self.det_stop.wait(0.25)
                continue

            if result.get("type") == "fatal":
                with self.det_lock:
                    self.det_error = result.get("error", "RF-DETR-S fatal error")
                return
            if result.get("type") == "batch_error":
                with self.det_lock:
                    self.det_error = result.get("error", "RF-DETR-S batch error")
                self.det_stop.wait(0.50)
                continue
            if result.get("type") != "result":
                continue

            now = time.monotonic()
            counts = {}
            accepted_inputs = 0
            for camera_id, captured_t in zip(result["cameras"], result["captured"]):
                age_ms = max(0.0, (now - float(captured_t)) * 1000.0)
                with self.det_lock:
                    self.det_finish_age_ms[camera_id] = age_ms
                if self.max_result_age_ms > 0.0 and age_ms > self.max_result_age_ms:
                    with self.det_lock:
                        self.stale_result_drops += 1
                    continue
                detections = self._scaled_detections(result["boxes"].get(camera_id, []))
                counts[camera_id] = len(detections)
                self.boxes.update(camera_id, float(captured_t), detections)
                accepted_inputs += 1

            batch_ms = float(result.get("batch_ms") or 0.0)
            with self.det_lock:
                self.det_calls += 1
                self.det_inputs += accepted_inputs
                self.det_batch_ms = batch_ms
                self.det_counts.update(counts)
                self.roi_inputs += int(result.get("roi_inputs") or 0)
                self.roi_variants += int(result.get("roi_variants") or 0)
                self.hard_rejects += int(result.get("hard_rejects") or 0)
                duty = max(self.det_duty_min, min(self.det_duty_max, self.det_duty))
                self.det_error = ""

            active = batch_ms / 1000.0
            idle = max(0.03, active * (1.0 / max(0.05, duty) - 1.0))
            self.det_stop.wait(min(2.0, idle))

    @staticmethod
    def _p95(values: deque[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        p95 = self._p95(self.wall_intervals_ms)
        slow_ms = float(self.det_cfg.get("wall_p95_slow_ms", 72))
        fast_ms = float(self.det_cfg.get("wall_p95_fast_ms", 60))

        with self.det_lock:
            if p95 is not None:
                if p95 > slow_ms:
                    self.det_duty = max(self.det_duty_min, self.det_duty - 0.025)
                elif p95 < fast_ms and self.det_ready:
                    self.det_duty = min(self.det_duty_max, self.det_duty + 0.010)
            calls = self.det_calls
            inputs = self.det_inputs
            batch_ms = self.det_batch_ms
            counts = dict(self.det_counts)
            ages = dict(self.det_finish_age_ms)
            meta = self.meta_boxes
            duty = self.det_duty
            ready = self.det_ready
            error = self.det_error
            timeouts = self.capture_timeouts
            stale = self.stale_result_drops
            roi_inputs = self.roi_inputs
            roi_variants = self.roi_variants
            hard_rejects = self.hard_rejects

        raw_text = " ".join(
            f"{camera_id}:{counts.get(camera_id, 0)}"
            for camera_id in self.camera_index
        )
        visible_text = " ".join(
            f"{camera_id}:{int(self.boxes.metrics(camera_id).get('active_tracks', 0))}"
            for camera_id in self.camera_index
        )
        age_text = " ".join(
            f"{camera_id}:{ages[camera_id]:.0f}ms"
            for camera_id in self.camera_index
            if camera_id in ages
        )
        wall = "?" if p95 is None else f"{p95:.1f}ms"
        print(
            "V3_RFDETR_STATS "
            f"ready={int(ready)} calls={calls} accepted_inputs={inputs} "
            f"micro_batch={self.micro_batch} batch={batch_ms:.1f}ms duty={duty:.0%} "
            f"wall_p95={wall} meta_boxes={meta} timeouts={timeouts} stale={stale} "
            f"roi_inputs={roi_inputs} roi_variants={roi_variants} hard_rejects={hard_rejects} "
            f"raw=[{raw_text}] visual=[{visible_text}] age=[{age_text}]"
            + (f" error={error}" if error else ""),
            flush=True,
        )
        return keep

    def run(self) -> int:
        ctx = mp.get_context("spawn")
        self.job_q = ctx.Queue(maxsize=1)
        self.result_q = ctx.Queue(maxsize=2)
        self.worker = ctx.Process(
            target=rfdetr_worker_v2,
            args=(self.job_q, self.result_q, self.det_cfg),
            daemon=True,
        )
        self.worker.start()
        self.scheduler_thread = threading.Thread(
            target=self._scheduler,
            name="vision-v3-rfdetr-scheduler",
            daemon=True,
        )
        self.scheduler_thread.start()
        print(
            "V3_RFDETR starting: six-camera display hot path + RF-DETR-S + mature Core-v1 detection policy",
            flush=True,
        )
        return super().run()

    def stop(self) -> None:
        if not self._detector_cleanup_done:
            self._detector_cleanup_done = True
            self.det_stop.set()
            self._clear_requests()
            self.mailbox.close()
            if self.job_q is not None:
                try:
                    self.job_q.put_nowait(None)
                except Exception:
                    pass
            if self.scheduler_thread is not None:
                self.scheduler_thread.join(timeout=2.0)
            if self.worker is not None:
                self.worker.join(timeout=3.0)
                if self.worker.is_alive():
                    self.worker.terminate()
                    self.worker.join(timeout=1.0)
            for tee, pad in self.tee_request_pads:
                try:
                    tee.release_request_pad(pad)
                except Exception:
                    pass
        super().stop()


def main() -> int:
    try:
        return SixCameraRFDETR().run()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(
            f"V3_RFDETR_FATAL {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
