from __future__ import annotations

import os
import queue as pyqueue
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _force_runtime_profile() -> None:
    """Install the production CAM-01 low-latency profile before camera modules import."""

    model = ROOT / "yolo26s.pt"
    if not model.is_file():
        raise RuntimeError(f"required detector model not found: {model}")

    forced = {
        "CAMERA_V2_RTSP_TRANSPORT": "tcp",
        "CAMERA_V2_RTSP_LATENCY_MS": "50",
        "CAMERA_V2_LOW_LATENCY_MODE": "1",
        "CAMERA_V2_MUX_TIMEOUT_US": "25000",
        "CAMERA_V2_SOURCE_FPS": "20",
        "CAMERA_V2_EXTRA_SURFACES": "4",
        "CAMERA_V2_FRAME_WIDTH": "960",
        "CAMERA_V2_FRAME_HEIGHT": "540",
        "CAMERA_V2_WALL_WIDTH": "1920",
        "CAMERA_V2_WALL_HEIGHT": "720",
        "CAMERA_V2_YOLO_MODEL": str(model),
        "CAMERA_V2_DETECT_WIDTH": "672",
        "CAMERA_V2_DETECT_HEIGHT": "384",
        "CAMERA_V2_MICRO_BATCH": "1",
        "CAMERA_V2_DETECT_ACTIVE_CAMERAS": "CAM-01",
        "CAMERA_V2_DETECT_CONF": "0.05",
        "CAMERA_V2_DETECT_IOU": "0.70",
        "CAMERA_V2_MAX_DET": "50",
        "CAMERA_V2_DETECT_STARTUP_DELAY": "1.0",
        "CAMERA_V2_DETECT_TARGET_HZ": "3.0",
        "CAMERA_V2_DETECT_MIN_HZ": "2.5",
        "CAMERA_V2_DETECT_MAX_HZ": "3.4",
        "CAMERA_V2_DETECT_GPU_DUTY": "0.34",
        "CAMERA_V2_DETECT_GPU_DUTY_MIN": "0.30",
        "CAMERA_V2_DETECT_GPU_DUTY_MAX": "0.38",
        "CAMERA_V2_MAX_DETECT_RESULT_AGE_MS": "320",
        "CAMERA_V2_TRACKER_WIDTH": "512",
        "CAMERA_V2_TRACKER_HEIGHT": "288",
        "CAMERA_V2_MIN_DISPLAY_TRACK_CONF": "0.05",
        "CAMERA_V2_BOX_RENDER_AGE": "0.45",
        "QWEN_REID_ENABLED": "0",
    }

    for key, value in forced.items():
        os.environ[key] = value

    for key in (
        "CAMERA_V2_POSE_MODEL",
        "CAMERA_V2_POSE_IMGSZ",
        "CAMERA_V2_POSE_CONF",
        "CAMERA_V2_POSE_IOU",
        "CAMERA_V2_YOLO_TRT86_ENGINE",
        "CAMERA_V2_YOLO_TRT86_PYTHON",
        "CAMERA_V2_YOLO_TRT86_WORKER",
    ):
        os.environ.pop(key, None)


_force_runtime_profile()

from . import detection as _det  # noqa: E402
from .person_tracking_reid import CameraPersonTrackingReID  # noqa: E402


class Cam01LowLatencyReID(CameraPersonTrackingReID):
    """CAM-01-only detector with latest-frame capture and no inference prefetch."""

    def __init__(self) -> None:
        self._capture_probe_seen = 0
        self._capture_probe_delivered = 0
        self._capture_probe_last_log = 0.0
        super().__init__()

    def _infer_gate_probe(self, _pad, _info, cid: str):
        # CAM-01 is allowed through the converter continuously. Other cameras are
        # display-only and are dropped before detector conversion/copy work.
        if cid == "CAM-01":
            return self.Gst.PadProbeReturn.OK
        return self.Gst.PadProbeReturn.DROP

    def _on_infer_sample(self, sink, cid: str):
        # The base class connected this signal during construction. V5 disables
        # emit-signals before PLAYING and captures on the converted-buffer pad
        # instead, so this is only a defensive drain if a signal slips through.
        sample = sink.emit("pull-sample")
        return self.Gst.FlowReturn.OK

    def _capture_converted_probe(self, pad, info, cid: str):
        """Copy exactly one requested BGRx frame from the post-convert raw buffer.

        This avoids appsink signal/preroll semantics entirely. The probe sits after
        nvvideoconvert + fixed BGRx caps, so the mapped buffer is normal CPU-visible
        RAW memory. It always returns OK and never blocks the streaming path.
        """
        if cid != "CAM-01":
            return self.Gst.PadProbeReturn.OK

        self._capture_probe_seen += 1

        with self.capture_lock:
            requested = bool(self.capture_requested.get(cid, False))
        if not requested:
            self._maybe_log_capture_probe()
            return self.Gst.PadProbeReturn.OK

        buffer = info.get_buffer()
        if buffer is None:
            self._maybe_log_capture_probe()
            return self.Gst.PadProbeReturn.OK

        caps = pad.get_current_caps()
        if caps is None or caps.get_size() == 0:
            self._maybe_log_capture_probe()
            return self.Gst.PadProbeReturn.OK

        structure = caps.get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        needed = width * height * 4

        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            self._maybe_log_capture_probe()
            return self.Gst.PadProbeReturn.OK

        try:
            if len(mapped.data) < needed:
                return self.Gst.PadProbeReturn.OK
            raw = np.frombuffer(mapped.data, dtype=np.uint8, count=needed)
            frame = raw.reshape((height, width, 4))[..., :3].copy()
        finally:
            buffer.unmap(mapped)

        captured_t = time.monotonic()
        with self.capture_lock:
            if not self.capture_requested.get(cid, False):
                self._maybe_log_capture_probe()
                return self.Gst.PadProbeReturn.OK
            self.capture_requested[cid] = False

        self.mailbox.put(cid, captured_t, frame)
        self._capture_probe_delivered += 1
        self._maybe_log_capture_probe()
        return self.Gst.PadProbeReturn.OK

    def _maybe_log_capture_probe(self) -> None:
        now = time.monotonic()
        if now - self._capture_probe_last_log < 2.0:
            return
        self._capture_probe_last_log = now
        with self.capture_lock:
            armed = int(bool(self.capture_requested.get("CAM-01", False)))
        print(
            "CAM01_CAPTURE_PROBE "
            f"seen={self._capture_probe_seen} "
            f"delivered={self._capture_probe_delivered} armed={armed}",
            flush=True,
        )

    def _scheduler(self) -> None:
        assert self.result_q is not None and self.job_q is not None

        try:
            ready = self.result_q.get(timeout=40.0)
        except pyqueue.Empty:
            with self.det_lock:
                self.det_error = "YOLO worker startup timeout"
            return

        if ready.get("type") != "ready":
            with self.det_lock:
                self.det_error = ready.get("error", "YOLO worker failed")
            return

        configured = [
            value.strip()
            for value in os.environ.get("CAMERA_V2_DETECT_ACTIVE_CAMERAS", "").split(",")
            if value.strip()
        ]
        all_ids = [camera.camera_id for camera in self.cameras]
        ids = [cid for cid in all_ids if cid in set(configured)] if configured else all_ids

        if ids != ["CAM-01"]:
            raise RuntimeError(f"CAM01_LOWLAT scheduler requires exactly CAM-01, got {ids!r}")

        groups = [ids]
        versions = {"CAM-01": 0}

        with self.det_lock:
            self.det_ready = True

        model_name = Path(str(ready.get("model") or _det.MODEL_SPEC)).name
        print(
            "CAMERA_TRACK_FINAL ready: "
            f"model={model_name} micro_batch={_det.MICRO_BATCH} "
            f"input={_det.INFER_WIDTH}x{_det.INFER_HEIGHT} "
            f"conf={os.environ.get('CAMERA_V2_DETECT_CONF')} "
            f"iou={os.environ.get('CAMERA_V2_DETECT_IOU')} "
            f"target={self.detector_target_hz:.1f}Hz/cam "
            f"range={self.detector_min_hz:.1f}-{self.detector_max_hz:.1f}Hz/cam "
            f"tracker={self.tracker_width}x{self.tracker_height} "
            f"display_conf={os.environ.get('CAMERA_V2_MIN_DISPLAY_TRACK_CONF')} "
            f"max_result_age={self.max_detector_result_age_ms:.0f}ms "
            f"device={ready.get('device')} cuda={ready.get('cuda')} "
            "capture_policy=postconvert-buffer-probe-latest",
            flush=True,
        )
        print(
            "CAMERA_DETECT_ACTIVE cameras=['CAM-01'] "
            "policy=postconvert-buffer-probe-latest",
            flush=True,
        )

        while not self.det_stop.is_set():
            cycle_started = time.monotonic()
            group = groups[0]

            # Arm the request only when the worker is ready. The next converted
            # live frame satisfies it; no frame is prefetched while inference runs.
            self._request_group(group)
            rows = self.mailbox.wait_group(group, versions, timeout=0.75)
            if rows is None:
                self._clear_requests()
                with self.det_lock:
                    self.capture_timeouts += 1
                self.det_stop.wait(0.010)
                continue

            frames = []
            captured = []
            for cid, row in zip(group, rows):
                version, captured_t, frame = row
                versions[cid] = version
                captured.append(captured_t)
                frames.append(frame)
            self._clear_requests()

            try:
                self.job_q.put(
                    {"cameras": group, "frames": frames, "captured": captured},
                    timeout=0.20,
                )
                result = self.result_q.get(timeout=5.0)
            except pyqueue.Empty:
                with self.det_lock:
                    self.det_error = "YOLO result timeout"
                self.det_stop.wait(0.025)
                continue

            if result.get("type") == "fatal":
                with self.det_lock:
                    self.det_error = result.get("error", "YOLO fatal error")
                return
            if result.get("type") == "batch_error":
                with self.det_lock:
                    self.det_error = result.get("error", "YOLO batch error")
                self.det_stop.wait(0.050)
                continue
            if result.get("type") != "result":
                continue

            completed_t = time.monotonic()
            counts: dict[str, int] = {}
            ages_ms: list[float] = []

            for cid, captured_t in zip(result["cameras"], result["captured"]):
                detections = self._dedup_and_expand(result["boxes"].get(cid, []))
                prepared = self.latency_compensator.prepare(cid, captured_t, detections)
                self._publish_prepared(cid, captured_t, prepared)
                counts[cid] = len(detections)
                ages_ms.append(max(0.0, (completed_t - captured_t) * 1000.0))
                self.detector_times[cid].append(completed_t)

            batch_ms = float(result.get("batch_ms") or 0.0)
            with self.det_lock:
                self.det_calls += 1
                self.det_inputs += 1
                self.det_batch_ms = batch_ms
                self.det_counts.update(counts)
                if ages_ms:
                    self.detector_result_age_ms = max(ages_ms)
                self.det_error = ""
                target_hz = self.detector_target_hz

            desired_call_interval = 1.0 / max(0.1, target_hz)
            elapsed = time.monotonic() - cycle_started
            idle = max(self.detector_min_idle, desired_call_interval - elapsed)
            self.det_stop.wait(min(0.20, idle))


def _validate_profile() -> None:
    active = os.environ.get("CAMERA_V2_DETECT_ACTIVE_CAMERAS", "")
    expected_model = str(ROOT / "yolo26s.pt")

    errors: list[str] = []
    if str(_det.MODEL_SPEC) != expected_model:
        errors.append(f"model={_det.MODEL_SPEC!r} expected={expected_model!r}")
    if int(_det.INFER_WIDTH) != 672 or int(_det.INFER_HEIGHT) != 384:
        errors.append(
            f"detector_shape={_det.INFER_WIDTH}x{_det.INFER_HEIGHT} expected=672x384"
        )
    if int(_det.MICRO_BATCH) != 1:
        errors.append(f"micro_batch={_det.MICRO_BATCH} expected=1")
    if active != "CAM-01":
        errors.append(f"active={active!r} expected='CAM-01'")

    if errors:
        raise RuntimeError("CAM01_LOWLAT profile invalid: " + "; ".join(errors))

    print(
        "CAM01_LOWLAT_PROFILE "
        f"model={Path(str(_det.MODEL_SPEC)).name} device=cuda:0 "
        f"active={active} detector={_det.INFER_WIDTH}x{_det.INFER_HEIGHT}/micro{_det.MICRO_BATCH} "
        "rtsp=50ms mux_timeout=25ms frame=960x540 wall=1920x720 "
        "tracker=512x288 max_result_age=320ms qwen=0 "
        "capture=postconvert-buffer-probe-latest",
        flush=True,
    )


def main() -> int:
    _validate_profile()
    runtime = Cam01LowLatencyReID()

    runtime._set_if(runtime.mux, "interpolation-method", 1)
    runtime._set_if(runtime.tiler, "interpolation-method", 1)

    capsfilter = runtime.pipeline.get_by_name("detect_caps_0")
    appsink = runtime.pipeline.get_by_name("detect_sink_0")
    if capsfilter is None or appsink is None:
        raise RuntimeError("CAM-01 inference branch elements not found")

    # Capture directly from the RAW BGRx buffer after nvvideoconvert. Disable the
    # signal API so preroll/new-sample behavior cannot gate capture delivery.
    appsink.set_property("emit-signals", False)
    appsink.set_property("sync", False)
    appsink.set_property("drop", True)
    appsink.set_property("max-buffers", 1)
    runtime._set_if(appsink, "async", False)
    runtime._set_if(appsink, "qos", False)

    srcpad = capsfilter.get_static_pad("src")
    if srcpad is None:
        raise RuntimeError("CAM-01 detect caps src pad not found")
    srcpad.add_probe(
        runtime.Gst.PadProbeType.BUFFER,
        runtime._capture_converted_probe,
        "CAM-01",
    )

    print(
        "CAM01_LOWLAT_SCALER mux=bilinear tiler=bilinear "
        "detector_path=postconvert-buffer-probe",
        flush=True,
    )

    return runtime.run()


if __name__ == "__main__":
    raise SystemExit(main())
