from __future__ import annotations

import signal
import time
from collections import deque

from .step1_cam02_lowlat_v7 import V11Step1Cam02LowLatV7


class V11Step1RenderPathDiagV20(V11Step1Cam02LowLatV7):
    """Counter/timestamp-only probes around the frozen render path."""

    def __init__(self) -> None:
        self.diag_queue = {}
        self.diag_convert = {}
        self.diag_sink = {}
        self.diag_last = {}
        self.diag_started = {}
        self.diag_scale_ms = {}
        self.diag_sink_ms = {}
        self.diag_report_at = time.monotonic()
        super().__init__()
        print(
            "CAMERA_V11_STEP1_RENDER_DIAG probes=counter-timestamp-only "
            "topology_changed=0 quality_changed=0",
            flush=True,
        )

    def _build_camera(self, index, camera) -> None:
        super()._build_camera(index, camera)
        cid = camera.camera_id
        self.diag_queue[cid] = 0
        self.diag_convert[cid] = 0
        self.diag_sink[cid] = 0
        self.diag_last[cid] = (0, 0, 0)
        self.diag_started[cid] = {}
        self.diag_scale_ms[cid] = deque(maxlen=2048)
        self.diag_sink_ms[cid] = deque(maxlen=2048)
        self.queues[cid].get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER, self._diag_queue_probe, cid
        )
        self.converters[cid].get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER, self._diag_convert_probe, cid
        )
        self.sinks[cid].get_static_pad("sink").add_probe(
            self.Gst.PadProbeType.BUFFER, self._diag_sink_probe, cid
        )

    @staticmethod
    def _key(buffer) -> int:
        pts = int(buffer.pts)
        return pts if pts >= 0 else id(buffer)

    def _diag_queue_probe(self, _pad, info, cid):
        buffer = info.get_buffer()
        if buffer is not None:
            now = time.monotonic_ns()
            with self.lock:
                self.diag_queue[cid] += 1
                starts = self.diag_started[cid]
                starts[self._key(buffer)] = now
                if len(starts) > 128:
                    for key in list(starts)[: len(starts) - 128]:
                        starts.pop(key, None)
        return self.Gst.PadProbeReturn.OK

    def _diag_convert_probe(self, _pad, info, cid):
        buffer = info.get_buffer()
        if buffer is not None:
            now = time.monotonic_ns()
            with self.lock:
                self.diag_convert[cid] += 1
                started = self.diag_started[cid].get(self._key(buffer))
                if started is not None:
                    self.diag_scale_ms[cid].append((now - started) / 1_000_000.0)
        return self.Gst.PadProbeReturn.OK

    def _diag_sink_probe(self, _pad, info, cid):
        buffer = info.get_buffer()
        if buffer is not None:
            now = time.monotonic_ns()
            with self.lock:
                self.diag_sink[cid] += 1
                started = self.diag_started[cid].pop(self._key(buffer), None)
                if started is not None:
                    self.diag_sink_ms[cid].append((now - started) / 1_000_000.0)
        return self.Gst.PadProbeReturn.OK

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        now = time.monotonic()
        elapsed = max(0.001, now - self.diag_report_at)
        self.diag_report_at = now
        rows = []
        with self.lock:
            for camera in self.cameras:
                cid = camera.camera_id
                current = (self.diag_queue[cid], self.diag_convert[cid], self.diag_sink[cid])
                previous = self.diag_last[cid]
                self.diag_last[cid] = current
                rows.append(
                    (
                        cid,
                        (current[0] - previous[0]) / elapsed,
                        (current[1] - previous[1]) / elapsed,
                        (current[2] - previous[2]) / elapsed,
                        self._pct(self.diag_scale_ms[cid], 0.50),
                        self._pct(self.diag_scale_ms[cid], 0.95),
                        self._pct(self.diag_sink_ms[cid], 0.50),
                        self._pct(self.diag_sink_ms[cid], 0.95),
                    )
                )
        for cid, queue_fps, convert_fps, sink_fps, scale50, scale95, sink50, sink95 in rows:
            print(
                "CAMERA_V11_STEP1_RENDER_PATH "
                f"camera={cid} queue_out={queue_fps:.2f} convert_out={convert_fps:.2f} "
                f"sink_in={sink_fps:.2f} scale_p50={scale50:.2f}ms scale_p95={scale95:.2f}ms "
                f"queue_to_sink_p50={sink50:.2f}ms queue_to_sink_p95={sink95:.2f}ms",
                flush=True,
            )
        return keep


def main() -> int:
    service = V11Step1RenderPathDiagV20()

    def stop(_signum, _frame) -> None:
        service.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    return service.run()


if __name__ == "__main__":
    raise SystemExit(main())
