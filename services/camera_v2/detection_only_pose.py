from __future__ import annotations

import multiprocessing as mp
import os
import queue as pyqueue
import threading
import time
from collections import deque

# Fixed Pascal/TensorRT-8.6 detector contract. These must be resolved before
# importing detection.py because that module reads them at import time.
os.environ.setdefault("CAMERA_V2_DETECT_WIDTH", "672")
os.environ.setdefault("CAMERA_V2_DETECT_HEIGHT", "384")
os.environ.setdefault("CAMERA_V2_MICRO_BATCH", "1")
os.environ.setdefault("CAMERA_V2_DETECT_CONF", "0.08")
os.environ.setdefault("CAMERA_V2_DETECT_IOU", "0.70")
os.environ.setdefault("CAMERA_V2_MAX_DET", "40")
os.environ.setdefault(
    "CAMERA_V2_DETECT_ACTIVE_CAMERAS",
    "CAM-01,CAM-02,CAM-03,CAM-04,CAM-05,CAM-06",
)
os.environ.setdefault("CAMERA_V2_DETECT_TARGET_HZ", "0.33")
os.environ.setdefault("CAMERA_V2_MAX_DETECT_RESULT_AGE_MS", "600")

# Sparse pose validation only. Strong YOLO boxes bypass pose completely.
os.environ.setdefault("CAMERA_V2_POSE_GATE_MODEL", "yolo26s-pose.pt")
os.environ.setdefault("CAMERA_V2_POSE_GATE_DEVICE", "cpu")
os.environ.setdefault("CAMERA_V2_POSE_GATE_THREADS", "1")
os.environ.setdefault("CAMERA_V2_POSE_GATE_IMGSZ", "224")
os.environ.setdefault("CAMERA_V2_POSE_GATE_MIN_CONF", "0.08")
os.environ.setdefault("CAMERA_V2_POSE_GATE_STRONG_CONF", "0.30")
os.environ.setdefault("CAMERA_V2_POSE_GATE_FALLBACK_CONF", "0.22")
os.environ.setdefault("CAMERA_V2_POSE_GATE_MAX_CANDIDATES", "2")
os.environ.setdefault("CAMERA_V2_POSE_GATE_POSITIVE_TTL_SEC", "12")
os.environ.setdefault("CAMERA_V2_POSE_GATE_NEGATIVE_TTL_SEC", "0")
os.environ.setdefault("CAMERA_V2_POSE_GATE_SOFT_KEEP_CONF", "0.14")
os.environ.setdefault("CAMERA_V2_POSE_GATE_REJECT_HITS", "2")
os.environ.setdefault("CAMERA_V2_POSE_GATE_REJECT_WINDOW_SEC", "10")

from . import detection as detection_module
from .detection import CameraDetectionV2, INFER_HEIGHT, INFER_WIDTH
from .pose_gate_v3 import PoseGateClient
from .secure import SecureCameraWallV2
from .yolo_trt86_fresh_bridge import yolo_trt86_fresh_worker

# CameraDetectionV2.run resolves this module global when spawning the worker.
detection_module._yolo_worker = yolo_trt86_fresh_worker

RESTART_EXIT_CODE = 75


class DetectionOnlyPoseV2(CameraDetectionV2):
    """Golden display + sparse YOLO26 TRT8.6 + conservative S-pose.

    Display graph stays identical to the proven camera-only graph:
      nvurisrcbin/NVDEC -> tee -> display queue -> nvstreammux -> tiler -> EGL

    ML is isolated on a second leaky tee branch. Its gate is placed before the
    expensive conversion/appsink, so frames are only converted when the detector
    scheduler explicitly requests a fresh sample:
      tee -> infer queue -> gate -> nvvideoconvert -> appsink -> TRT8.6 -> S-pose

    Deliberately absent from display: NvDCF, OSD, metadata injection, motion
    prediction, Global ID, ReID and face recognition.

    YOLO26 one-to-one end-to-end output is already duplicate-filtered by the
    model. This runtime therefore does NO external NMS, IoU de-dup, containment
    de-dup or final geometry de-dup. Post-processing is person/confidence filtering
    in the TRT worker followed by the sparse pose validator.
    """

    def __init__(self) -> None:
        self._restart_requested = False
        self._restart_reason = ""
        self._source_started_at: dict[str, float] = {}
        self._last_frames: dict[str, int] = {}
        self._last_progress: dict[str, float] = {}
        self._result_age_samples: deque[float] = deque(maxlen=120)
        self._capture_gate_logged: set[str] = set()
        self._capture_sample_logged: set[str] = set()
        self._letterbox: tuple[int, int, int, int] | None = None
        self._latest_detections: dict[str, tuple[float, list]] = {}
        self._detector_times: dict[str, deque[float]] = {}
        super().__init__()

        # Match the camera-only quality baseline exactly.
        self._set_if(self.mux, "interpolation-method", 4)
        self._set_if(self.tiler, "interpolation-method", 4)
        self._set_if(self.mux, "buffer-pool-size", 12)

        self.pose_gate = PoseGateClient()
        self.detector_target_hz = max(
            0.10,
            float(os.environ.get("CAMERA_V2_DETECT_TARGET_HZ", "0.33")),
        )
        self.max_detector_result_age_ms = max(
            350.0,
            float(os.environ.get("CAMERA_V2_MAX_DETECT_RESULT_AGE_MS", "600")),
        )
        self._stall_s = max(
            8.0,
            float(os.environ.get("CAMERA_V2_DETECTION_ONLY_STALL_SEC", "12")),
        )
        now = time.monotonic()
        self._last_frames = {cid: int(self.stats[cid].frames) for cid in self.sources}
        self._last_progress = {cid: now for cid in self.sources}
        self._detector_times = {cid: deque(maxlen=24) for cid in self.sources}

        for source in self.sources.values():
            self._set_if(source, "rtsp-reconnect-interval", 2)
            self._set_if(source, "rtsp-reconnect-attempts", 3)
            self._set_if(source, "async-handling", True)

        self._audit_detection_only_graph()
        print(
            "CAMERA_DETECTION_ONLY_ARCH "
            "display=NVDEC/tee/queue/nvstreammux/tiler/EGL "
            "ml=tee/leaky-gate/nvvideoconvert/appsink/TRT86/S-pose "
            "nvdcf=0 osd=0 metadata_injection=0 motion_predictor=0 "
            "external_nms=0 geometry_dedup=0",
            flush=True,
        )
        print(
            "CAMERA_DETECTION_ONLY_PROFILE "
            f"mux={self.frame_width}x{self.frame_height}/lanczos "
            f"wall={self.wall_width}x{self.wall_height}/lanczos "
            f"detector={INFER_WIDTH}x{INFER_HEIGHT}/TRT8.6/B1 "
            f"target={self.detector_target_hz:.2f}Hz/cam "
            "pose=yolo26s-pose/cpu/sparse",
            flush=True,
        )

    # CameraDetectionV2 normally rewires wall_queue through OSD and injects
    # metadata. Detection-only intentionally leaves the clean display untouched.
    def _install_osd_and_meta(self) -> None:
        self.osd = None
        self.meta_boxes = 0
        self.wall_queue.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._wall_probe,
        )

    def _add_camera(self, index, camera) -> None:
        super()._add_camera(index, camera)
        cid = camera.camera_id
        source = self.pipeline.get_by_name(f"camera_v2_source_{index}")
        converter = self.pipeline.get_by_name(f"detect_convert_{index}")
        appsink = self.pipeline.get_by_name(f"detect_sink_{index}")
        if source is None or converter is None or appsink is None:
            raise RuntimeError(f"{cid}: detection-only camera branch incomplete")

        # Keep outer nvurisrcbin and inner rtspsrc on TCP.
        self._set_if(source, "select-rtp-protocol", 4)
        self._set_if(source, "async-handling", True)

        # Exact fixed-input geometry: 1280x720 -> 672x378 plus 3px top/bottom.
        scale = min(
            float(INFER_WIDTH) / float(self.frame_width),
            float(INFER_HEIGHT) / float(self.frame_height),
        )
        content_w = max(2, min(INFER_WIDTH, int(round(self.frame_width * scale))))
        content_h = max(2, min(INFER_HEIGHT, int(round(self.frame_height * scale))))
        pad_x = max(0, (INFER_WIDTH - content_w) // 2)
        pad_y = max(0, (INFER_HEIGHT - content_h) // 2)
        converter.set_property("dest-crop", f"{pad_x}:{pad_y}:{content_w}:{content_h}")
        self._letterbox = (pad_x, pad_y, content_w, content_h)

        # No old bootstrap frame: the scheduler opens the gate immediately before
        # it consumes a frame.
        appsink.set_property("async", False)
        appsink.set_property("sync", False)
        self._set_if(appsink, "wait-on-eos", False)
        with self.capture_lock:
            self.capture_requested[cid] = False

    def _infer_gate_probe(self, pad, info, cid: str):
        result = super()._infer_gate_probe(pad, info, cid)
        if result == self.Gst.PadProbeReturn.OK and cid not in self._capture_gate_logged:
            self._capture_gate_logged.add(cid)
            print(f"CAMERA_DETECTION_GATE cid={cid} first_buffer=1", flush=True)
        return result

    def _on_infer_sample(self, sink, cid: str):
        first = cid not in self._capture_sample_logged
        result = super()._on_infer_sample(sink, cid)
        if first:
            self._capture_sample_logged.add(cid)
            print(f"CAMERA_DETECTION_SAMPLE cid={cid} first_sample=1", flush=True)
        return result

    def _scaled_detections(self, rows):
        mapping = self._letterbox
        if mapping is None:
            return super()._scaled_detections(rows)
        pad_x, pad_y, content_w, content_h = mapping
        sx = float(self.frame_width) / float(content_w)
        sy = float(self.frame_height) / float(content_h)
        max_x = float(self.frame_width - 1)
        max_y = float(self.frame_height - 1)
        output = []
        for coords, conf in rows:
            x1, y1, x2, y2 = [float(v) for v in coords]
            x1 = max(0.0, min(max_x, (x1 - pad_x) * sx))
            x2 = max(0.0, min(max_x, (x2 - pad_x) * sx))
            y1 = max(0.0, min(max_y, (y1 - pad_y) * sy))
            y2 = max(0.0, min(max_y, (y2 - pad_y) * sy))
            if x2 > x1 and y2 > y1:
                output.append(((x1, y1, x2, y2), float(conf)))
        return output

    @staticmethod
    def _peer_name(element, pad_name: str) -> str | None:
        pad = element.get_static_pad(pad_name)
        if pad is None:
            return None
        peer = pad.get_peer()
        if peer is None:
            return None
        parent = peer.get_parent_element()
        return parent.get_name() if parent is not None else None

    def _expect_peer(self, element, pad_name: str, expected: str, label: str) -> None:
        actual = self._peer_name(element, pad_name)
        if actual != expected:
            raise RuntimeError(
                f"CAMERA_DETECTION_ONLY_AUDIT {label}: expected={expected} actual={actual}"
            )

    def _audit_detection_only_graph(self) -> None:
        self._expect_peer(self.mux, "src", self.tiler.get_name(), "mux->tiler")
        self._expect_peer(self.tiler, "src", self.wall_caps.get_name(), "tiler->wall_geometry")
        self._expect_peer(self.wall_caps, "src", self.wall_queue.get_name(), "wall_geometry->queue")
        self._expect_peer(self.wall_queue, "src", self.sink.get_name(), "queue->egl")

        forbidden = (
            "person_nvdcf_tracker",
            "track_osd",
            "detect_osd",
            "native_yolo26_pgie",
            "native_nvdcf_tracker",
        )
        present = [name for name in forbidden if self.pipeline.get_by_name(name) is not None]
        if present:
            raise RuntimeError(
                "CAMERA_DETECTION_ONLY_AUDIT inline analytics present: " + ",".join(present)
            )
        for index, camera in enumerate(self.cameras):
            cid = camera.camera_id
            tee = self.pipeline.get_by_name(f"detect_tee_{index}")
            display_q = self.pipeline.get_by_name(f"camera_v2_queue_{index}")
            infer_q = self.pipeline.get_by_name(f"detect_queue_{index}")
            converter = self.pipeline.get_by_name(f"detect_convert_{index}")
            sink = self.pipeline.get_by_name(f"detect_sink_{index}")
            if any(v is None for v in (tee, display_q, infer_q, converter, sink)):
                raise RuntimeError(f"CAMERA_DETECTION_ONLY_AUDIT {cid}: tee branch missing")
            self._expect_peer(infer_q, "src", converter.get_name(), f"{cid}:inferq->convert")
        if (self.frame_width, self.frame_height) != (1280, 720):
            raise RuntimeError("CAMERA_DETECTION_ONLY_AUDIT mux geometry changed")
        if (self.wall_width, self.wall_height) != (1920, 720):
            raise RuntimeError("CAMERA_DETECTION_ONLY_AUDIT wall geometry changed")
        print(
            "CAMERA_DETECTION_ONLY_AUDIT status=OK "
            "display=tee->queue1->nvstreammux->tiler->wallcaps->queue1->EGL "
            "ml=tee->leaky-queue->preconvert-gate->convert->appsink "
            "nvdcf=0 osd=0 dedup=0",
            flush=True,
        )

    def _startup_stagger_seconds(self) -> float:
        configured = float(getattr(self.settings.deepstream, "startup_stagger_sec", 0.5))
        return max(
            0.10,
            min(3.0, float(os.environ.get("CAMERA_V2_STARTUP_STAGGER_SEC", str(configured)))),
        )

    def _prepare_staggered_sources(self) -> None:
        ordered = [camera.camera_id for camera in self.cameras]
        stagger = self._startup_stagger_seconds()
        for cid in ordered:
            source = self.sources[cid]
            source.set_locked_state(True)
            source.set_state(self.Gst.State.NULL)
        print(
            f"CAMERA_DETECTION_SOURCE_STAGGER order={ordered} interval={stagger:.2f}s",
            flush=True,
        )
        for index, cid in enumerate(ordered):
            delay_ms = max(1, int(round(index * stagger * 1000.0)))

            def _start(camera_id=cid, ordinal=index):
                if self._stopping:
                    return False
                source = self.sources[camera_id]
                source.set_locked_state(False)
                sync = bool(source.sync_state_with_parent())
                now = time.monotonic()
                self._source_started_at[camera_id] = now
                self._last_progress[camera_id] = now
                self._last_frames[camera_id] = int(self.stats[camera_id].frames)
                print(
                    f"CAMERA_DETECTION_SOURCE_START cid={camera_id} index={ordinal} sync={int(sync)}",
                    flush=True,
                )
                return False

            self.GLib.timeout_add(delay_ms, _start)

    def _source_watchdog(self) -> bool:
        if self._stopping:
            return False
        now = time.monotonic()
        for cid, started_at in list(self._source_started_at.items()):
            current = int(self.stats[cid].frames)
            if current != self._last_frames[cid]:
                self._last_frames[cid] = current
                self._last_progress[cid] = now
                continue
            if now - started_at < self._stall_s:
                continue
            stalled = now - self._last_progress[cid]
            if stalled < self._stall_s:
                continue
            self._restart_requested = True
            self._restart_reason = f"{cid} no-frames {stalled:.1f}s"
            print(
                f"CAMERA_DETECTION_PROCESS_RESTART reason={self._restart_reason} "
                f"exit_code={RESTART_EXIT_CODE}",
                flush=True,
            )
            self.stop()
            return False
        return True

    def _scheduler(self) -> None:
        assert self.result_q is not None and self.job_q is not None
        try:
            ready = self.result_q.get(timeout=40.0)
        except pyqueue.Empty:
            with self.det_lock:
                self.det_error = "TRT86 worker startup timeout"
            return
        if ready.get("type") != "ready":
            with self.det_lock:
                self.det_error = ready.get("error", "TRT86 worker failed")
            return

        with self.det_lock:
            self.det_ready = True
        ids = [camera.camera_id for camera in self.cameras]
        versions = {cid: 0 for cid in ids}
        group_index = 0
        age_log_n = 0
        print(
            "CAMERA_DETECTION_READY "
            f"backend={ready.get('backend')} model={ready.get('model')} "
            f"input={INFER_WIDTH}x{INFER_HEIGHT} target={self.detector_target_hz:.2f}Hz/cam "
            "postprocess=person+confidence-only pose=sparse-S",
            flush=True,
        )

        while not self.det_stop.is_set():
            cycle_started = time.monotonic()
            cid = ids[group_index % len(ids)]
            group_index += 1
            self._request_group([cid])
            rows = self.mailbox.wait_group([cid], versions, timeout=0.8)
            if rows is None:
                self._clear_requests()
                with self.det_lock:
                    self.capture_timeouts += 1
                self.det_stop.wait(0.025)
                continue

            version, captured_t, frame = rows[0]
            versions[cid] = version
            self._clear_requests()
            try:
                self.job_q.put(
                    {"cameras": [cid], "frames": [frame], "captured": [captured_t]},
                    timeout=0.3,
                )
                result = self.result_q.get(timeout=5.0)
            except pyqueue.Empty:
                with self.det_lock:
                    self.det_error = "TRT86 result timeout"
                continue
            if result.get("type") == "fatal":
                with self.det_lock:
                    self.det_error = result.get("error", "TRT86 fatal error")
                return
            if result.get("type") != "result":
                continue

            raw_rows = list(result.get("boxes", {}).get(cid, []))
            gated_rows, gate = self.pose_gate.filter(
                cid,
                frame,
                raw_rows,
                trusted_boxes=None,
            )
            # IMPORTANT: no geometry de-dup here. YOLO26 one-to-one end-to-end
            # predictions are final model outputs; only pose validation changes
            # the candidate set.
            detections = self._scaled_detections(gated_rows)
            completed_t = time.monotonic()
            age_ms = max(0.0, (completed_t - float(captured_t)) * 1000.0)
            self._result_age_samples.append(age_ms)
            self._latest_detections[cid] = (float(captured_t), detections)
            self._detector_times[cid].append(completed_t)

            with self.det_lock:
                self.det_calls += 1
                self.det_inputs += 1
                self.det_batch_ms = float(result.get("batch_ms") or 0.0)
                self.det_counts[cid] = len(detections)
                self.det_error = ""

            print(
                "CAMERA_DETECTION_RESULT "
                f"cid={cid} raw={len(raw_rows)} direct={gate.direct} "
                f"cache_accept={gate.cache_accept} pose_checked={gate.pose_checked} "
                f"pose_accept={gate.pose_accept} pose_reject={gate.pose_reject} "
                f"soft_hold={int(getattr(gate, 'soft_hold', 0))} "
                f"confirmed_reject={int(getattr(gate, 'confirmed_reject', 0))} "
                f"final={len(detections)} pose_ms={gate.pose_ms:.1f} "
                f"age={age_ms:.1f}ms",
                flush=True,
            )

            age_log_n += 1
            if age_log_n <= 3 or age_log_n % 20 == 0:
                print(
                    "CAMERA_DETECTION_FRESHNESS "
                    f"n={age_log_n} result_age={age_ms:.1f}ms "
                    f"trt_batch={float(result.get('batch_ms') or 0.0):.1f}ms",
                    flush=True,
                )

            desired_interval = 1.0 / max(0.1, self.detector_target_hz * len(ids))
            elapsed = time.monotonic() - cycle_started
            self.det_stop.wait(max(0.02, desired_interval - elapsed))

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        now = time.monotonic()
        with self.det_lock:
            counts = dict(self.det_counts)
            calls = self.det_calls
            batch_ms = self.det_batch_ms
            timeouts = self.capture_timeouts
            ready = self.det_ready
            error = self.det_error
        actual = []
        for cid in self.sources:
            recent = [t for t in self._detector_times.get(cid, ()) if now - t <= 15.0]
            hz = 0.0
            if len(recent) >= 2:
                span = recent[-1] - recent[0]
                if span > 0.0:
                    hz = (len(recent) - 1) / span
            actual.append(f"{cid}:{hz:.2f}")
        persons = " ".join(f"{cid}:{counts.get(cid, 0)}" for cid in self.sources)
        print(
            "CAMERA_DETECTION_ONLY "
            f"ready={int(ready)} calls={calls} batch={batch_ms:.1f}ms "
            f"timeouts={timeouts} persons=[{persons}] actual_hz=[{' '.join(actual)}] "
            "nvdcf=0 osd=0 dedup=0"
            + (f" error={error}" if error else ""),
            flush=True,
        )
        return keep

    def run(self) -> int:
        self._prepare_staggered_sources()
        self.GLib.timeout_add_seconds(1, self._source_watchdog)

        ctx = mp.get_context("spawn")
        self.job_q = ctx.Queue(maxsize=1)
        self.result_q = ctx.Queue(maxsize=2)
        self.worker = ctx.Process(
            target=detection_module._yolo_worker,
            args=(self.job_q, self.result_q),
            daemon=True,
        )
        self.worker.start()
        self.scheduler_thread = threading.Thread(
            target=self._scheduler,
            name="camera-v2-detection-only-scheduler",
            daemon=True,
        )
        self.scheduler_thread.start()
        try:
            # Bypass CameraDetectionV2.run(), which would spawn a second worker.
            result = SecureCameraWallV2.run(self)
        finally:
            self.det_stop.set()
            self._clear_requests()
            self.mailbox.close()
            try:
                self.pose_gate.close()
            except Exception:
                pass
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
        if self._restart_requested:
            return RESTART_EXIT_CODE
        return result


def main() -> int:
    return DetectionOnlyPoseV2().run()


if __name__ == "__main__":
    raise SystemExit(main())
