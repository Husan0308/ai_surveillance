from __future__ import annotations

import time

from . import deepstream_torch_batch6_smooth_v3 as smooth_v3


class DeepStreamCameraOnlyProbe(smooth_v3.DeepStreamTorchBatch6SmoothV3):
    """Same six-camera display/decode pipeline with inference fully disabled."""

    def _preprocess_gate_probe(self, _pad, _info, _cid: str):
        # Drop before nvvideoconvert: zero inference resize/color-conversion cost.
        self.preprocess_drops += 1
        return self.Gst.PadProbeReturn.DROP

    def _infer_loop(self) -> None:
        print(
            "CAMERA_ONLY_PROBE inference disabled; all inference buffers dropped "
            "before nvvideoconvert",
            flush=True,
        )
        while not self.stop_event.wait(0.5):
            pass

    def _print_stats(self) -> bool:
        now = time.monotonic()
        parts = []
        for cid in self.camera_ids:
            stat = self.source_stats[cid]
            elapsed = max(0.001, now - stat.last_print)
            fps = (stat.frames - stat.last_frames) / elapsed
            stat.last_frames = stat.frames
            stat.last_print = now
            dq = int(self.display_queues[cid].get_property("current-level-buffers"))
            parts.append(f"{cid}:{fps:.1f}fps dq={dq}")
        print(
            "CAMERA_ONLY_PROBE_V3 " + " | ".join(parts)
            + f" || pre_infer_drops={self.preprocess_drops}",
            flush=True,
        )
        return True


def run() -> int:
    return DeepStreamCameraOnlyProbe().run()


if __name__ == "__main__":
    raise SystemExit(run())
