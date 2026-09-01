from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import numpy as np

from .step5_global_shadow_v1 import EXPIRED_SHADOW
from .step6_global_shadow_runtime_v1 import V11Step6GlobalShadowRuntimeV1


class V11Step8TwoPersonDebugRuntimeV1(V11Step6GlobalShadowRuntimeV1):
    """Step6 runtime plus a debug-only CAM-01/CAM-04 bbox preview.

    The preview reuses the detector's already-copied BGR frames and the tracker
    snapshots produced by the normal hot path. It opens no extra RTSP streams,
    runs no extra detector/ReID inference, owns no frame backlog, and never mutates
    tracker/global-identity state. It exists only to make the manual Step8 ground
    truth test observable.
    """

    DEBUG_CAMERAS = ("CAM-01", "CAM-04")
    DEBUG_WIDTH = 672
    DEBUG_HEIGHT = 378

    def __init__(self) -> None:
        super().__init__()
        raw = os.environ.get("V11_STEP8_DEBUG_BBOX", "1").strip().lower()
        self.debug_bbox_enabled = raw in {"1", "true", "yes", "on"}
        self.debug_phase_path = Path(
            os.environ.get("V11_STEP8_PHASE_STATE", "/tmp/camera_v11_step8_phase.txt")
        )
        self.debug_pipeline = None
        self.debug_appsrc = None
        self.debug_cv2 = None
        self.debug_last_push_mono = 0.0
        self.debug_push_interval_sec = 0.18
        self.debug_frames = {
            cid: np.zeros((self.DEBUG_HEIGHT, self.DEBUG_WIDTH, 3), dtype=np.uint8)
            for cid in self.DEBUG_CAMERAS
        }
        self.debug_updates = {cid: 0 for cid in self.DEBUG_CAMERAS}
        self.debug_errors = 0
        self._debug_closed = False

        if self.debug_bbox_enabled:
            self._start_debug_preview()
        print(
            "CAMERA_V11_STEP8_DEBUG_BBOX_V1 "
            f"enabled={int(self.debug_bbox_enabled)} cameras=CAM-01+CAM-04 "
            "source=existing_detector_bgr tracker_snapshot=existing no_extra_rtsp=1 "
            "extra_inference=0 queue=latest-only global_state_mutation=0",
            flush=True,
        )

    def _start_debug_preview(self) -> None:
        try:
            import cv2
        except Exception as exc:  # pragma: no cover - environment guard
            raise RuntimeError(f"Step8 debug preview requires cv2: {exc}") from exc
        self.debug_cv2 = cv2

        Gst = self.Gst
        if Gst.ElementFactory.find("appsrc") is None or Gst.ElementFactory.find("videoconvert") is None:
            raise RuntimeError("Step8 debug preview requires appsrc and videoconvert")
        sink_factory = "ximagesink" if Gst.ElementFactory.find("ximagesink") is not None else "autovideosink"
        if Gst.ElementFactory.find(sink_factory) is None:
            raise RuntimeError("Step8 debug preview has no usable video sink")

        wall_width = self.DEBUG_WIDTH * len(self.DEBUG_CAMERAS)
        description = (
            "appsrc name=step8_debug_src is-live=true block=false do-timestamp=true format=time "
            f"caps=video/x-raw,format=BGR,width={wall_width},height={self.DEBUG_HEIGHT},framerate=5/1 "
            "! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream "
            f"! videoconvert ! {sink_factory} sync=false"
        )
        self.debug_pipeline = Gst.parse_launch(description)
        self.debug_appsrc = self.debug_pipeline.get_by_name("step8_debug_src")
        if self.debug_appsrc is None:
            raise RuntimeError("Step8 debug appsrc creation failed")
        result = self.debug_pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Step8 debug preview failed to enter PLAYING")

        for cid in self.DEBUG_CAMERAS:
            frame = self.debug_frames[cid]
            cv2.rectangle(frame, (0, 0), (self.DEBUG_WIDTH - 1, self.DEBUG_HEIGHT - 1), (90, 90, 90), 2)
            cv2.putText(
                frame,
                f"{cid} waiting for detector frame...",
                (24, self.DEBUG_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )
        self._push_debug_wall(force=True)
        print(
            "CAMERA_V11_STEP8_DEBUG_PREVIEW ready=1 "
            f"sink={sink_factory} size={wall_width}x{self.DEBUG_HEIGHT} update=detector-cadence",
            flush=True,
        )

    def _phase_label(self) -> str:
        try:
            value = self.debug_phase_path.read_text(encoding="utf-8").strip()
            return value[:96] if value else "STEP8 PHASE: waiting"
        except Exception:
            return "STEP8 PHASE: waiting"

    def _identity_labels(self, camera_id: str, raw_track_id: str) -> tuple[str, str, str]:
        stable_id = self.camera_tracklet_continuity.canonical_track_id(camera_id, raw_track_id)
        stable_text = stable_id or "CT-pending"
        shadow_id = "GSH-pending"
        verify_state = ""
        if stable_id is None:
            return stable_text, shadow_id, verify_state

        member = (str(camera_id), str(stable_id))
        try:
            records = self.global_shadow_worker.machine.records
            for record in records:
                if record.state == EXPIRED_SHADOW:
                    continue
                if member in record.members:
                    shadow_id = record.shadow_global_id
                    break
        except Exception:
            # Debug-only read races must never perturb the identity worker.
            return stable_text, shadow_id, verify_state

        if shadow_id != "GSH-pending":
            try:
                for row in self.global_shadow_worker.verifier.records:
                    if row.shadow_global_id == shadow_id:
                        verify_state = str(row.state)
                        break
            except Exception:
                verify_state = ""
        return stable_text, shadow_id, verify_state

    def _draw_debug_frame(self, cid: str, snapshots) -> None:
        cv2 = self.debug_cv2
        if cv2 is None or self.detector is None:
            return
        content = getattr(self.detector, "content", None)
        if content is None or content.shape[:2] != (self.DEBUG_HEIGHT, self.DEBUG_WIDTH):
            self.debug_errors += 1
            return
        frame = np.ascontiguousarray(content.copy())
        phase = self._phase_label()

        cv2.rectangle(frame, (0, 0), (self.DEBUG_WIDTH - 1, 31), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"{cid} | {phase}",
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        for snap in snapshots:
            x1, y1, x2, y2 = (int(round(v)) for v in snap.bbox_xyxy)
            x1 = max(0, min(self.DEBUG_WIDTH - 1, x1))
            y1 = max(32, min(self.DEBUG_HEIGHT - 1, y1))
            x2 = max(x1 + 1, min(self.DEBUG_WIDTH - 1, x2))
            y2 = max(y1 + 1, min(self.DEBUG_HEIGHT - 1, y2))

            if snap.predicted or snap.state == "lost":
                color = (0, 190, 255)
            elif snap.confirmed:
                color = (0, 220, 0)
            else:
                color = (180, 180, 180)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            stable_id, shadow_id, verify_state = self._identity_labels(cid, snap.track_id)
            raw_short = snap.track_id.replace(f"{cid}-", "")
            stable_short = stable_id.replace(f"{cid}-", "")
            state = "PRED" if snap.predicted else str(snap.state).upper()
            verify_short = verify_state.replace("_SHADOW", "") if verify_state else ""
            label = f"{raw_short} | {stable_short} | {shadow_id} | {state}"
            if verify_short:
                label += f" | {verify_short}"

            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1
            )
            text_y = y1 - 5
            if text_y - text_h < 32:
                text_y = min(self.DEBUG_HEIGHT - 4, y1 + text_h + 6)
            bg_y1 = max(32, text_y - text_h - baseline - 2)
            bg_y2 = min(self.DEBUG_HEIGHT - 1, text_y + baseline + 2)
            bg_x2 = min(self.DEBUG_WIDTH - 1, x1 + text_w + 6)
            cv2.rectangle(frame, (x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.putText(
                frame,
                label,
                (x1 + 3, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"visible={len(snapshots)}  green=tracked  yellow=pred/lost",
            (8, self.DEBUG_HEIGHT - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        self.debug_frames[cid] = frame
        self.debug_updates[cid] += 1

    def _push_debug_wall(self, *, force: bool = False) -> None:
        if not self.debug_bbox_enabled or self.debug_appsrc is None:
            return
        now = time.monotonic()
        if not force and now - self.debug_last_push_mono < self.debug_push_interval_sec:
            return
        self.debug_last_push_mono = now
        wall = np.ascontiguousarray(
            np.concatenate([self.debug_frames[cid] for cid in self.DEBUG_CAMERAS], axis=1)
        )
        payload = wall.tobytes()
        buffer = self.Gst.Buffer.new_allocate(None, len(payload), None)
        buffer.fill(0, payload)
        buffer.duration = int(self.Gst.SECOND / 5)
        result = self.debug_appsrc.emit("push-buffer", buffer)
        if result != self.Gst.FlowReturn.OK:
            self.debug_errors += 1

    def _consume_tracking(self, cid: str, boxes: list[list[float]], captured_ns: int) -> None:
        # Keep Step3's frozen tracker behavior exactly; this subclass only retains
        # the returned snapshots long enough to draw a debug frame.
        update = self.tracker.update(cid, boxes, captured_ns)
        self.stage_values["tracker"].append(float(update.step_ms))
        ids = tuple(snapshot.track_id for snapshot in update.snapshots)
        if len(ids) != len(set(ids)):
            self.track_duplicate_errors += 1
        prefix = f"{cid}-T"
        self.track_prefix_errors += sum(1 for track_id in ids if not track_id.startswith(prefix))
        self.track_updates[cid] += 1
        self.track_created[cid] += int(update.created)
        self.track_recovered[cid] += int(update.recovered)
        self.track_removed[cid] += int(update.removed)
        self.latest_track_ids[cid] = ids

        if self.debug_bbox_enabled and cid in self.DEBUG_CAMERAS:
            try:
                self._draw_debug_frame(cid, tuple(update.snapshots))
                self._push_debug_wall()
            except Exception as exc:
                self.debug_errors += 1
                if self.debug_errors <= 3:
                    print(
                        "CAMERA_V11_STEP8_DEBUG_ERROR "
                        f"camera={cid} error={type(exc).__name__}:{exc}",
                        flush=True,
                    )

    def _print_stats(self) -> None:
        super()._print_stats()
        if self.debug_bbox_enabled:
            print(
                "CAMERA_V11_STEP8_DEBUG_STATS "
                + " ".join(
                    f"{cid.lower().replace('-', '')}_updates={self.debug_updates[cid]}"
                    for cid in self.DEBUG_CAMERAS
                )
                + f" errors={self.debug_errors} extra_rtsp=0 extra_inference=0",
                flush=True,
            )

    def _close_debug_preview(self) -> None:
        if self._debug_closed:
            return
        self._debug_closed = True
        pipeline = self.debug_pipeline
        appsrc = self.debug_appsrc
        self.debug_pipeline = None
        self.debug_appsrc = None
        if appsrc is not None:
            try:
                appsrc.emit("end-of-stream")
            except Exception:
                pass
        if pipeline is not None:
            try:
                pipeline.set_state(self.Gst.State.NULL)
            except Exception:
                pass

    def close(self) -> None:
        self._close_debug_preview()
        super().close()


def main() -> int:
    service = V11Step8TwoPersonDebugRuntimeV1()

    def stop(_signum, _frame) -> None:
        service.stop_requested = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        return service.run()
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
