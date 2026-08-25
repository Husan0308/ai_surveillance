from __future__ import annotations

import os
import queue as pyqueue
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _force_runtime_profile() -> None:
    """Install the production CAM-01 low-latency profile before camera modules import.

    This deliberately avoids the Pascal-safe pose monkey-patch and the old
    YOLO26m defaults. The detector is YOLO26s on CUDA, CAM-01 only, while NvDCF
    owns per-frame motion between detector refreshes.
    """

    model = ROOT / "yolo26s.pt"
    if not model.is_file():
        raise RuntimeError(f"required detector model not found: {model}")

    forced = {
        # One 20 FPS frame is 50 ms. Keep exactly one-frame RTSP jitter budget on
        # the local NVR path; drop-on-latency remains enabled in the base source.
        "CAMERA_V2_RTSP_TRANSPORT": "tcp",
        "CAMERA_V2_RTSP_LATENCY_MS": "50",
        "CAMERA_V2_LOW_LATENCY_MODE": "1",
        "CAMERA_V2_MUX_TIMEOUT_US": "25000",
        "CAMERA_V2_SOURCE_FPS": "20",
        "CAMERA_V2_EXTRA_SURFACES": "4",
        # Detection taps the decoded source before mux. 960x540 is therefore only
        # the display/tracker working surface, not the detector source resolution.
        "CAMERA_V2_FRAME_WIDTH": "960",
        "CAMERA_V2_FRAME_HEIGHT": "540",
        "CAMERA_V2_WALL_WIDTH": "1920",
        "CAMERA_V2_WALL_HEIGHT": "720",
        # Fast CUDA person detector. Explicitly one camera and one image per job.
        "CAMERA_V2_YOLO_MODEL": str(model),
        "CAMERA_V2_DETECT_WIDTH": "672",
        "CAMERA_V2_DETECT_HEIGHT": "384",
        "CAMERA_V2_MICRO_BATCH": "1",
        "CAMERA_V2_DETECT_ACTIVE_CAMERAS": "CAM-01",
        "CAMERA_V2_DETECT_CONF": "0.05",
        "CAMERA_V2_DETECT_IOU": "0.70",
        "CAMERA_V2_MAX_DET": "50",
        "CAMERA_V2_DETECT_STARTUP_DELAY": "1.0",
        # NvDCF is the visual authority between detector refreshes. 3 Hz detector
        # refresh is enough for correction while leaving decode/display headroom.
        "CAMERA_V2_DETECT_TARGET_HZ": "3.0",
        "CAMERA_V2_DETECT_MIN_HZ": "2.5",
        "CAMERA_V2_DETECT_MAX_HZ": "3.4",
        "CAMERA_V2_DETECT_GPU_DUTY": "0.34",
        "CAMERA_V2_DETECT_GPU_DUTY_MIN": "0.30",
        "CAMERA_V2_DETECT_GPU_DUTY_MAX": "0.38",
        # Normal latest-frame results must remain fresh; never hide latency by
        # accepting 500-800 ms old detections as tracker truth.
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

# These imports MUST stay below _force_runtime_profile(). detection.py snapshots
# model/geometry/batch constants at import time.
from . import detection as _det  # noqa: E402
from .person_tracking_reid import CameraPersonTrackingReID  # noqa: E402


class Cam01LowLatencyReID(CameraPersonTrackingReID):
    """CAM-01 runtime with a true latest-frame detector path.

    The generic path used a one-shot pad gate before nvvideoconvert. That gate can
    consume the request before appsink receives a usable PLAYING-state sample,
    leaving the scheduler in repeated capture timeouts. For CAM-01 we keep the
    inference branch flowing to appsink and apply the one-shot gate *inside the
    appsink callback*. The callback always drains the max-buffers=1 sink, but only
    maps/copies one frame when the scheduler asks for it. This preserves latest
    frame semantics without queue growth or prefetch ageing.
    """

    def _infer_gate_probe(self, _pad, _info, cid: str):
        # CAM-01 must keep flowing all the way to appsink so preroll/PLAYING and
        # subsequent new-sample delivery cannot be starved by the old pad gate.
        if cid == "CAM-01":
            return self.Gst.PadProbeReturn.OK

        # Other cameras are display-only in this tuning phase. Drop them before
        # nvvideoconvert so they consume no detector conversion/copy bandwidth.
        return self.Gst.PadProbeReturn.DROP

    def _on_infer_sample(self, sink, cid: str):
        sample = sink.emit("pull-sample")
        if sample is None:
            return self.Gst.FlowReturn.OK

        # Always drain the CAM-01 appsink, but do no CPU map/copy work unless the
        # scheduler currently wants a fresh detector frame.
        if cid != "CAM-01":
            return self.Gst.FlowReturn.OK

        with self.capture_lock:
            requested = bool(self.capture_requested.get(cid, False))
        if not requested:
            return self.Gst.FlowReturn.OK

        captured_t = time.monotonic()
        caps = sample.get_caps().get_structure(0)
        width = int(caps.get_value("width"))
        height = int(caps.get_value("height"))
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            # Leave the request armed; the next live sample can satisfy it.
            return self.Gst.FlowReturn.OK

        try:
            needed = width * height * 4
            raw = np.frombuffer(mapped.data, dtype=np.uint8, count=needed)
            frame = raw.reshape((height, width, 4))[..., :3].copy()
        finally:
            buffer.unmap(mapped)

        # Clear only after a complete frame is safely copied. If the scheduler
        # cancelled while mapping, avoid publishing an unsolicited sample.
        with self.capture_lock:
            if not self.capture_requested.get(cid, False):
                return self.Gst.FlowReturn.OK
            self.capture_requested[cid] = False

        self.mailbox.put(cid, captured_t, frame)
        return self.Gst.FlowReturn.OK

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
            for value in os.environ.get(
                "CAMERA_V2_DETECT_ACTIVE_CAMERAS",
                "",
            ).split(",")
            if value.strip()
        ]
        all_ids = [camera.camera_id for camera in self.cameras]
        if configured:
            allowed = set(configured)
            ids = [cid for cid in all_ids if cid in allowed]
        else:
            ids = all_ids

        if ids != ["CAM-01"]:
            raise RuntimeError(
                "CAM01_LOWLAT scheduler requires exactly CAM-01, got "
                f"{ids!r}"
            )

        groups = [
            ids[i : i + int(_det.MICRO_BATCH)]
            for i in range(0, len(ids), int(_det.MICRO_BATCH))
        ]
        versions = {cid: 0 for cid in ids}
        group_index = 0

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
            "capture_policy=appsink-latest-no-prefetch",
            flush=True,
        )
        print(
            f"CAMERA_DETECT_ACTIVE cameras={ids} policy=appsink-latest-no-prefetch",
            flush=True,
        )

        while not self.det_stop.is_set():
            cycle_started = time.monotonic()
            group = groups[group_index % len(groups)]
            group_index += 1

            # Arm one capture request. CAM-01's appsink is continuously drained,
            # so the next live sample satisfies this without an upstream pad race.
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
                    {
                        "cameras": group,
                        "frames": frames,
                        "captured": captured,
                    },
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
                prepared = self.latency_compensator.prepare(
                    cid,
                    captured_t,
                    detections,
                )
                self._publish_prepared(cid, captured_t, prepared)
                counts[cid] = len(detections)
                ages_ms.append(max(0.0, (completed_t - captured_t) * 1000.0))
                self.detector_times[cid].append(completed_t)

            batch_ms = float(result.get("batch_ms") or 0.0)
            with self.det_lock:
                self.det_calls += 1
                self.det_inputs += len(group)
                self.det_batch_ms = batch_ms
                self.det_counts.update(counts)
                if ages_ms:
                    self.detector_result_age_ms = max(ages_ms)
                self.det_error = ""
                target_hz = self.detector_target_hz

            desired_call_interval = 1.0 / max(0.1, target_hz * len(groups))
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
        "capture=appsink-latest-no-prefetch",
        flush=True,
    )


def main() -> int:
    _validate_profile()
    runtime = Cam01LowLatencyReID()

    runtime._set_if(runtime.mux, "interpolation-method", 1)
    runtime._set_if(runtime.tiler, "interpolation-method", 1)
    print(
        "CAM01_LOWLAT_SCALER mux=bilinear tiler=bilinear detector_path=independent",
        flush=True,
    )

    return runtime.run()


if __name__ == "__main__":
    raise SystemExit(main())
