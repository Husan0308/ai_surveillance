from __future__ import annotations

import sys
import time

from .detection_only_pose import DetectionOnlyPoseV2
from .secure import SecureCameraWallV2


class DetectionOnlyPoseV3(DetectionOnlyPoseV2):
    """Final detection-only hardening.

    Keeps the V2 clean display / isolated ML graph, while fixing the dynamic-pad
    race that existed in CameraDetectionV2 and making stats detection-only (no
    legacy metadata/tracker counters).
    """

    def _source_to_tee(self, _source, pad, tee, cid: str) -> None:
        # disable-audio=True is set on nvurisrcbin. If caps are already known,
        # reject a non-video pad; if caps are still pending, link once instead of
        # permanently returning and waiting for a pad-added signal that may never
        # be emitted again.
        caps = pad.get_current_caps()
        if caps is None or caps.get_size() == 0:
            try:
                caps = pad.query_caps(None)
            except Exception:
                caps = None

        if caps is not None and caps.get_size() > 0 and not caps.is_any():
            try:
                media = str(caps.get_structure(0).get_name())
            except Exception:
                media = ""
            if media and not media.startswith("video/"):
                return

        sink = tee.get_static_pad("sink")
        if sink is None:
            print(
                f"CAMERA_DETECTION {cid} tee sink missing",
                file=sys.stderr,
                flush=True,
            )
            return
        if sink.is_linked():
            return

        result = pad.link(sink)
        if result != self.Gst.PadLinkReturn.OK:
            caps_text = caps.to_string() if caps is not None else "pending"
            print(
                f"CAMERA_DETECTION {cid} source->tee failed result={result} caps={caps_text}",
                file=sys.stderr,
                flush=True,
            )
            return

        caps_text = caps.to_string() if caps is not None else "pending"
        print(
            f"CAMERA_DETECTION {cid} source->tee linked caps={caps_text}",
            flush=True,
        )

    def _print_stats(self) -> bool:
        # Skip CameraDetectionV2._print_stats(): it contains legacy metadata/OSD
        # counters and adaptive duty logic that are deliberately absent here.
        keep = SecureCameraWallV2._print_stats(self)
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
            "display_inline_ml=0 nvdcf=0 osd=0 dedup=0"
            + (f" error={error}" if error else ""),
            flush=True,
        )
        return keep


def main() -> int:
    return DetectionOnlyPoseV3().run()


if __name__ == "__main__":
    raise SystemExit(main())
