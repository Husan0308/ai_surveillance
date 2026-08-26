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
            f"detector={INFER_WIDTH}x{INFER_HEIGHT}/TRT8.6/B1/bilinear "
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

        # IMPORTANT: DeepStream nvvideoconvert defaults to interpolation-method=6,
        # which is nearest-neighbour on dGPU. Ultralytics letterbox resizing uses
        # linear interpolation. Small/far persons can disappear under nearest
        # downscaling, which broke production parity for CAM-02/CAM-05. Force
        # GPU bilinear so the TRT input matches the reference preprocessing much
        # more closely while keeping the gated conversion off the display path.
        self._set_if(converter, "compute-hw", 1)
        self._set_if(converter, "interpolation-method", 1)
        self._set_if(converter, "output-buffers", 2)

        # Exact fixed-input geometry: 16:9 source -> 672x378 plus 3px top/bottom.
        # All current camera main streams are 16:9; the TRT worker explicitly
        # fills the 3px bars with Ultralytics padding value 114.
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

        print(
            "CAMERA_DETECTION_PREPROCESS "
            f"cid={cid} convert=gpu-bilinear interpolation=1 "
            f"letterbox={content_w}x{content_h}+{pad_x}+{pad_y}",
            flush=True,
        )

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
            try:
                interpolation = int(converter.get_property("interpolation-method"))
            except Exception:
                interpolation = -1
            if interpolation != 1:
                raise RuntimeError(
                    f"CAMERA_DETECTION_ONLY_AUDIT {cid}: detector interpolation={interpolation}, expected=1"
                )
        if (self.frame_width, self.frame_height) != (1280, 720):
            raise RuntimeError("CAMERA_DETECTION_ONLY_AUDIT mux geometry changed")
        if (self.wall_width, self.wall_height) != (1920, 720):
            raise RuntimeError("CAMERA_DETECTION_ONLY_AUDIT wall geometry changed")

    def _pose_filter(self, cid: str, rows, frame):
        # YOLO26 E2E output is already NMS-free/final. Do not externally collapse
        # overlapping people here. Pose only validates ambiguous detections.
        boxes = []
        for coords, score in rows:
            boxes.append((tuple(float(v) for v in coords), float(score)))

        filtered, pose_diag = self.pose_gate.filter(
            cid,
            frame,
            boxes,
            existing_boxes=None,
        )
        return filtered, pose_diag

    def _process_result(self, message: dict) -> None:
        now = time.monotonic()
        captured = float(message.get("captured") or now)
        age_ms = max(0.0, (now - captured) * 1000.0)
        rows = message.get("boxes") or []
        cid = str(message.get("camera") or "")
        frame = message.get("frame")

        if not cid or cid not in self.sources:
            return

        raw_count = len(rows)
        if frame is not None:
            final_rows, pose_diag = self._pose_filter(cid, rows, frame)
        else:
            final_rows = [(tuple(float(v) for v in coords), float(conf)) for coords, conf in rows]
            pose_diag = {
                "direct": raw_count,
                "cache_accept": 0,
                "cache_reject": 0,
                "tracker_reuse": 0,
                "pose_checked": 0,
                "pose_accept": 0,
                "pose_reject": 0,
                "soft_hold": 0,
                "confirmed_reject": 0,
                "low_reject": 0,
                "overflow": 0,
                "pose_ms": 0.0,
                "fallback": 0,
            }

        with self.det_lock:
            self.det_calls += 1
            self.det_inputs += 1
            self.det_batch_ms = float(message.get("trt_ms") or message.get("batch_ms") or 0.0)
            self.det_counts[cid] = len(final_rows)
            self._latest_detections[cid] = (now, final_rows)
        self._detector_times[cid].append(now)
        self._result_age_samples.append(age_ms)

        print(
            "CAMERA_DETECTION_RESULT "
            f"cid={cid} raw={raw_count} direct={pose_diag.get('direct', 0)} "
            f"cache_accept={pose_diag.get('cache_accept', 0)} "
            f"pose_checked={pose_diag.get('pose_checked', 0)} "
            f"pose_accept={pose_diag.get('pose_accept', 0)} "
            f"pose_reject={pose_diag.get('pose_reject', 0)} "
            f"soft_hold={pose_diag.get('soft_hold', 0)} "
            f"confirmed_reject={pose_diag.get('confirmed_reject', 0)} "
            f"final={len(final_rows)} pose_ms={pose_diag.get('pose_ms', 0.0):.1f} "
            f"age={age_ms:.1f}ms",
            flush=True,
        )

        n = self.det_calls
        if n <= 3 or n % 20 == 0:
            trt_ms = float(message.get("trt_ms") or 0.0)
            print(
                "CAMERA_DETECTION_FRESHNESS "
                f"n={n} result_age={age_ms:.1f}ms trt_batch={trt_ms:.1f}ms",
                flush=True,
            )

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        now = time.monotonic()
        for cid in self.sources:
            current = int(self.stats[cid].frames)
            if current != self._last_frames.get(cid, 0):
                self._last_frames[cid] = current
                self._last_progress[cid] = now
                continue
            started = self._source_started_at.get(cid)
            if started is None or now - started < self._stall_s:
                continue
            stalled_for = now - self._last_progress.get(cid, started)
            if stalled_for >= self._stall_s and not self._restart_requested:
                self._restart_requested = True
                self._restart_reason = f"{cid}-no-frame-{stalled_for:.1f}s"
                print(
                    "CAMERA_DETECTION_PROCESS_RESTART "
                    f"reason={self._restart_reason} exit_code={RESTART_EXIT_CODE}",
                    flush=True,
                )
                try:
                    self.loop.quit()
                except Exception:
                    pass
                return False
        return keep

    def _start_source_at(self, index: int) -> bool:
        keep = super()._start_source_at(index)
        if index < len(self.cameras):
            cid = self.cameras[index].camera_id
            self._source_started_at[cid] = time.monotonic()
            print(
                f"CAMERA_DETECTION_SOURCE_START cid={cid} index={index} sync=1",
                flush=True,
            )
        return keep

    def run(self) -> int:
        code = super().run()
        try:
            self.pose_gate.close()
        except Exception:
            pass
        if self._restart_requested:
            return RESTART_EXIT_CODE
        return code
