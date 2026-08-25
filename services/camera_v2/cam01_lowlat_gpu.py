from __future__ import annotations

import os
import queue as pyqueue
import time
from pathlib import Path


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
        # Local-LAN live profile. Around one 20 FPS frame of RTSP buffering is
        # enough for this NVR; the display path stays latest-frame-only.
        "CAMERA_V2_RTSP_TRANSPORT": "tcp",
        "CAMERA_V2_RTSP_LATENCY_MS": "60",
        "CAMERA_V2_LOW_LATENCY_MODE": "1",
        "CAMERA_V2_MUX_TIMEOUT_US": "25000",
        "CAMERA_V2_SOURCE_FPS": "20",
        "CAMERA_V2_EXTRA_SURFACES": "4",
        # Detection taps the decoded source before mux. 960x540 is therefore only
        # the display/tracker working surface, not the detector source resolution.
        # It cuts six-stream downstream bandwidth versus 1280x720 without reducing
        # detector detail.
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
        # With latest-only capture, a normal result should land below this. Do not
        # stretch this to 500-800 ms; old positions must never become tracker truth.
        "CAMERA_V2_MAX_DETECT_RESULT_AGE_MS": "320",
        "CAMERA_V2_TRACKER_WIDTH": "512",
        "CAMERA_V2_TRACKER_HEIGHT": "288",
        "CAMERA_V2_MIN_DISPLAY_TRACK_CONF": "0.05",
        "CAMERA_V2_BOX_RENDER_AGE": "0.45",
        # Qwen is not allowed to steal GPU/CPU time from the live path.
        "QWEN_REID_ENABLED": "0",
    }

    for key, value in forced.items():
        os.environ[key] = value

    # Explicitly remove experiment selectors that could silently replace the
    # YOLO26s CUDA worker or re-enable a stale TensorRT/pose path.
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
    """CAM-01 runtime with strict latest-only detector scheduling.

    The generic CameraPersonTrackingFinal scheduler historically re-created an
    all-camera round-robin and prefetched the next inference frame while the GPU
    was still busy. With a ~230 ms detector, that prefetched frame was already
    ~200 ms old before inference started, so completion age approached 450 ms and
    the 320 ms freshness gate correctly rejected it. It also ignored
    CAMERA_V2_DETECT_ACTIVE_CAMERAS.

    This scheduler restores the earlier Detection Log Analysis contract:
    wait until the detector is ready for work -> request ONE fresh frame -> infer
    immediately -> publish -> idle. There is never a queued/prefetched camera
    frame waiting behind GPU work.
    """

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

        model_name = Path(
            str(ready.get("model") or _det.MODEL_SPEC)
        ).name
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
            "capture_policy=latest-only-no-prefetch",
            flush=True,
        )
        print(
            f"CAMERA_DETECT_ACTIVE cameras={ids} policy=latest-only-no-prefetch",
            flush=True,
        )

        while not self.det_stop.is_set():
            cycle_started = time.monotonic()
            group = groups[group_index % len(groups)]
            group_index += 1

            # Critical latency rule: request a frame only when the worker is ready
            # for the next inference.  The wait itself does NOT age the detector
            # input: captured_t is stamped only when appsink actually delivers the
            # fresh frame.  Therefore use a generous wait to avoid starving the
            # one-shot gate during decoder/converter scheduling jitter.
            self._request_group(group)
            rows = self.mailbox.wait_group(group, versions, timeout=1.50)
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

            for cid, captured_t in zip(
                result["cameras"],
                result["captured"],
            ):
                detections = self._dedup_and_expand(
                    result["boxes"].get(cid, [])
                )
                prepared = self.latency_compensator.prepare(
                    cid,
                    captured_t,
                    detections,
                )
                self._publish_prepared(cid, captured_t, prepared)
                counts[cid] = len(detections)
                ages_ms.append(
                    max(0.0, (completed_t - captured_t) * 1000.0)
                )
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

            # Idle AFTER the completed inference. The next camera frame is not
            # requested until the next loop, so idle time never makes an input old.
            desired_call_interval = 1.0 / max(
                0.1,
                target_hz * len(groups),
            )
            elapsed = time.monotonic() - cycle_started
            idle = max(
                self.detector_min_idle,
                desired_call_interval - elapsed,
            )
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
        "rtsp=60ms mux_timeout=25ms frame=960x540 wall=1920x720 "
        "tracker=512x288 max_result_age=320ms qwen=0 "
        "capture=latest-only-no-prefetch",
        flush=True,
    )


def main() -> int:
    _validate_profile()
    runtime = Cam01LowLatencyReID()

    # The restored wall uses Lanczos for mux/tiler scaling. Bilinear is sufficient
    # for the live wall and materially cheaper; detector input has an independent
    # conversion path and is not affected by this choice.
    runtime._set_if(runtime.mux, "interpolation-method", 1)
    runtime._set_if(runtime.tiler, "interpolation-method", 1)
    print(
        "CAM01_LOWLAT_SCALER mux=bilinear tiler=bilinear detector_path=independent",
        flush=True,
    )

    return runtime.run()


if __name__ == "__main__":
    raise SystemExit(main())
