from __future__ import annotations

import sys

from .person_tracking_pascal_trt86 import CameraPersonTrackingPascalTRT86


class CameraPascalRuntime(CameraPersonTrackingPascalTRT86):
    """Final Pascal entrypoint with race-safe nvurisrcbin -> tee linking."""

    def _source_to_tee(self, _source, pad, tee, cid: str) -> None:
        # pad-added can fire before fixed caps are available. Returning in that
        # state is unsafe because the dynamic pad is not guaranteed to be emitted
        # again when caps later become fixed. Audio is disabled on nvurisrcbin, so
        # unknown caps may be linked; known non-video caps are still rejected.
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
            print(f"CAMERA_PASCAL {cid} tee sink pad missing", file=sys.stderr, flush=True)
            return
        if sink.is_linked():
            return
        result = pad.link(sink)
        if result != self.Gst.PadLinkReturn.OK:
            caps_text = caps.to_string() if caps is not None else "pending"
            print(
                f"CAMERA_PASCAL {cid} source->tee link failed result={result} caps={caps_text}",
                file=sys.stderr,
                flush=True,
            )
            return
        caps_text = caps.to_string() if caps is not None else "pending"
        print(f"CAMERA_PASCAL {cid} source->tee linked caps={caps_text}", flush=True)


def main() -> int:
    return CameraPascalRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
