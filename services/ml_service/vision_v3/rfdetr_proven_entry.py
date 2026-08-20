from __future__ import annotations

"""Compatibility entry for the previously proven RF-DETR-S camera runtime.

Vision V3 keeps its clean six-camera owner, but detector-side details below are
copied from ``agent/rfdetr-s-core-final``: keep the capture gate armed until an
appsink sample really arrives, respect padded BGRx row stride, and use GPU cubic
resize for the detector tap.  These changes do not touch the live display branch.
"""

import numpy as np

from . import rfdetr_runtime as runtime


_original_add_camera = runtime.SixCameraRFDETR._add_camera


def _add_camera_proven(self, index, camera) -> None:
    _original_add_camera(self, index, camera)
    converter = self.pipeline.get_by_name(f"v3_detect_convert_{index}")
    if converter is not None:
        self._set_if(converter, "interpolation-method", 2)
        self._set_if(converter, "compute-hw", 1)


def _infer_gate_until_sample(self, _pad, _info, camera_id: str):
    # The old one-shot gate cleared the request before nvvideoconvert/appsink had
    # necessarily produced a sample. Keep it armed until _on_infer_sample succeeds.
    with self.capture_lock:
        requested = bool(self.capture_requested.get(camera_id, False))
    if not requested:
        return self.Gst.PadProbeReturn.DROP
    return self.Gst.PadProbeReturn.OK


def _on_infer_sample_stride_safe(self, sink, camera_id: str):
    sample = sink.emit("pull-sample")
    if sample is None:
        return self.Gst.FlowReturn.OK

    structure = sample.get_caps().get_structure(0)
    width = int(structure.get_value("width"))
    height = int(structure.get_value("height"))
    buffer = sample.get_buffer()
    ok, mapped = buffer.map(self.Gst.MapFlags.READ)
    if not ok:
        return self.Gst.FlowReturn.OK

    try:
        tight_stride = width * 4
        mapped_size = int(getattr(mapped, "size", len(mapped.data)))
        if mapped_size < tight_stride * height:
            raise RuntimeError(
                f"{camera_id}: BGRx buffer too small: "
                f"{mapped_size} < {tight_stride * height}"
            )

        if mapped_size % height == 0:
            row_stride = mapped_size // height
        else:
            row_stride = tight_stride
        if row_stride < tight_stride:
            raise RuntimeError(
                f"{camera_id}: invalid BGRx stride={row_stride}, tight={tight_stride}"
            )

        needed = row_stride * height
        raw = np.frombuffer(mapped.data, dtype=np.uint8, count=needed)
        rows = raw.reshape((height, row_stride))
        bgrx = rows[:, :tight_stride].reshape((height, width, 4))
        frame = bgrx[..., :3].copy()
    finally:
        buffer.unmap(mapped)

    # Clear only after a real image has been copied into the newest-frame mailbox.
    with self.capture_lock:
        self.capture_requested[camera_id] = False
    self.mailbox.put(camera_id, self.time_monotonic() if hasattr(self, "time_monotonic") else __import__("time").monotonic(), frame)
    return self.Gst.FlowReturn.OK


runtime.SixCameraRFDETR._add_camera = _add_camera_proven
runtime.SixCameraRFDETR._infer_gate_probe = _infer_gate_until_sample
runtime.SixCameraRFDETR._on_infer_sample = _on_infer_sample_stride_safe


def main() -> int:
    return runtime.main()


if __name__ == "__main__":
    raise SystemExit(main())
