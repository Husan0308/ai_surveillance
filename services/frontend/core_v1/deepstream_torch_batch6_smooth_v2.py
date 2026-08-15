from __future__ import annotations

import os
import sys

from . import deepstream_torch_batch6_smooth as smooth


class DeepStreamTorchBatch6SmoothV2(smooth.DeepStreamTorchBatch6SmoothWall):
    """NVIDIA-guided RTSP/display tuning on top of strict batch-6 inference.

    The detector architecture is unchanged: six cameras still enter one model
    forward. These changes target the *display* cadence and RTSP jitter only.

    Key tuning:
      * nvstreammux timeout = 1 / 30 fps ~= 33.3 ms instead of 50 ms;
      * larger RTSP jitterbuffer (250 ms) to avoid late-packet frame drops;
      * rtp-multi (UDP preferred, TCP fallback) stays enabled;
      * extra decoder surfaces increased to reduce decoder starvation;
      * all display queues remain latest-only and sink QoS remains disabled.
    """

    def __init__(self):
        super().__init__()

        max_display_fps = max(
            1.0, float(os.environ.get("AI_DISPLAY_MAX_FPS", "30.0"))
        )
        mux_timeout_us = max(1, int(round(1_000_000.0 / max_display_fps)))
        rtsp_latency_ms = max(
            1, int(os.environ.get("AI_RTSP_LATENCY_MS", "250"))
        )
        udp_buffer_bytes = max(
            524288, int(os.environ.get("AI_RTSP_UDP_BUFFER", str(8 * 1024 * 1024)))
        )
        decoder_surfaces = max(
            1, int(os.environ.get("AI_DECODER_EXTRA_SURFACES", "8"))
        )

        self._set_if(self.mux, "batched-push-timeout", mux_timeout_us)
        self._set_if(self.mux, "live-source", True)
        self._set_if(self.mux, "sync-inputs", False)
        self._set_if(self.mux, "buffer-pool-size", 12)
        self._set_if(self.mux, "compute-hw", 1)

        # Keep the wall low-latency: old rendered buffers must not accumulate.
        self._set_if(self.wall_queue, "max-size-buffers", 1)
        self._set_if(self.wall_queue, "max-size-bytes", 0)
        self._set_if(self.wall_queue, "max-size-time", 0)
        self._set_if(self.wall_queue, "leaky", 2)
        self._set_if(self.sink, "sync", False)
        self._set_if(self.sink, "qos", False)

        tuned = 0
        for index in range(len(self.cameras)):
            source = self.pipeline.get_by_name(f"src_{index}")
            if source is None:
                continue
            # 0 = rtp-multi: UDP + UDP multicast + TCP. Do not force TCP.
            self._set_if(source, "select-rtp-protocol", 0)
            self._set_if(source, "latency", rtsp_latency_ms)
            self._set_if(source, "drop-on-latency", True)
            self._set_if(source, "udp-buffer-size", udp_buffer_bytes)
            self._set_if(source, "num-extra-surfaces", decoder_surfaces)
            self._set_if(source, "cudadec-memtype", 0)
            tuned += 1

        self._smooth_v2 = {
            "max_display_fps": max_display_fps,
            "mux_timeout_us": mux_timeout_us,
            "rtsp_latency_ms": rtsp_latency_ms,
            "udp_buffer_bytes": udp_buffer_bytes,
            "decoder_extra_surfaces": decoder_surfaces,
            "sources_tuned": tuned,
        }

        print(
            "TORCH_BATCH6_SMOOTH_V2 "
            f"display_target={max_display_fps:.1f}fps "
            f"mux_timeout={mux_timeout_us}us "
            f"rtsp_latency={rtsp_latency_ms}ms "
            f"udp_buffer={udp_buffer_bytes}B "
            f"decoder_surfaces={decoder_surfaces} "
            f"sources={tuned}",
            flush=True,
        )


def run() -> int:
    return DeepStreamTorchBatch6SmoothV2().run()


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as exc:
        print(f"TORCH_BATCH6_SMOOTH_V2 FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
