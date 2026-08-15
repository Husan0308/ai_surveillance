from __future__ import annotations

import os
import sys

from . import deepstream_torch_batch6_smooth_v2 as v2


class DeepStreamTorchBatch6SmoothV3(v2.DeepStreamTorchBatch6SmoothV2):
    """PTS-paced live-RTSP display profile.

    The six-camera detector batch architecture is unchanged. This revision only
    changes live display timing after measurement showed all six decoded sources
    are actually ~20 FPS.

    Smoothness-first choices:
      * mux timeout matches the measured 20 FPS source cadence (~50 ms);
      * sync-inputs=1 asks nvstreammux to timestamp-synchronize live inputs;
      * max-latency gives slightly late RTSP buffers room to arrive;
      * EGL sink sync=1 renders according to PTS instead of burst-as-fast-as-possible;
      * QoS stays disabled so the sink does not feed frame-drop pressure upstream.
    """

    def __init__(self):
        super().__init__()

        source_fps = max(1.0, float(os.environ.get("AI_ACTUAL_SOURCE_FPS", "20.0")))
        mux_timeout_us = max(1, int(round(1_000_000.0 / source_fps)))
        max_latency_ms = max(1, int(os.environ.get("AI_MUX_MAX_LATENCY_MS", "300")))

        self._set_if(self.mux, "batched-push-timeout", mux_timeout_us)
        self._set_if(self.mux, "sync-inputs", True)
        self._set_if(self.mux, "max-latency", max_latency_ms * 1_000_000)
        self._set_if(self.mux, "live-source", True)

        # Smooth wall rendering: honor PTS. Keep QoS disabled to avoid
        # propagating frame-drop decisions upstream on a jittery live source.
        self._set_if(self.sink, "sync", True)
        self._set_if(self.sink, "qos", False)
        self._set_if(self.sink, "max-lateness", -1)

        # Keep only the newest rendered wall buffer if the display ever stalls.
        self._set_if(self.wall_queue, "max-size-buffers", 1)
        self._set_if(self.wall_queue, "max-size-bytes", 0)
        self._set_if(self.wall_queue, "max-size-time", 0)
        self._set_if(self.wall_queue, "leaky", 2)

        print(
            "TORCH_BATCH6_SMOOTH_V3 "
            f"measured_source={source_fps:.1f}fps "
            f"mux_timeout={mux_timeout_us}us "
            "sync_inputs=1 sink_sync=1 "
            f"max_latency={max_latency_ms}ms",
            flush=True,
        )


def run() -> int:
    return DeepStreamTorchBatch6SmoothV3().run()


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as exc:
        print(f"TORCH_BATCH6_SMOOTH_V3 FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
