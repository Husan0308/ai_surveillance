from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np

from .detection_only_pose_v3 import DetectionOnlyPoseV3


class DetectionResolutionCapture(DetectionOnlyPoseV3):
    """One-shot high-resolution source capture for detector recall diagnosis.

    This keeps the production display and sparse detector branches unchanged.
    Selected source tees get a third diagnostic branch whose gate is placed
    before nvvideoconvert, so each selected camera pays for exactly one
    1920x1080 conversion/copy and then every later buffer is dropped upstream.
    """

    def __init__(self) -> None:
        targets = os.environ.get(
            "CAMERA_V2_RES_CAPTURE_CAMERAS", "CAM-01,CAM-02,CAM-04,CAM-05"
        )
        self._res_targets = {v.strip() for v in targets.split(",") if v.strip()}
        self._res_width = int(os.environ.get("CAMERA_V2_RES_CAPTURE_WIDTH", "1920"))
        self._res_height = int(os.environ.get("CAMERA_V2_RES_CAPTURE_HEIGHT", "1080"))
        self._res_dir = Path(
            os.environ.get(
                "CAMERA_V2_RES_CAPTURE_DIR", ".runtime/yolo26_resolution"
            )
        ).resolve()
        self._res_dir.mkdir(parents=True, exist_ok=True)
        self._res_lock = threading.Lock()
        self._res_inflight: set[str] = set()
        self._res_done: set[str] = set()
        self._res_request_pads: list[tuple[object, object]] = []
        super().__init__()
        print(
            "CAMERA_RES_CAPTURE_PROFILE "
            f"targets={sorted(self._res_targets)} "
            f"frame={self._res_width}x{self._res_height} "
            f"dir={self._res_dir} one_shot=1 gate=preconvert",
            flush=True,
        )

    def _add_camera(self, index, camera) -> None:
        super()._add_camera(index, camera)
        cid = camera.camera_id
        if cid not in self._res_targets:
            return

        tee = self.pipeline.get_by_name(f"detect_tee_{index}")
        if tee is None:
            raise RuntimeError(f"{cid}: resolution capture tee missing")

        queue = self._make("queue", f"res_capture_queue_{index}")
        converter = self._make("nvvideoconvert", f"res_capture_convert_{index}")
        capsfilter = self._make("capsfilter", f"res_capture_caps_{index}")
        appsink = self._make("appsink", f"res_capture_sink_{index}")

        self._set_if(queue, "max-size-buffers", 1)
        self._set_if(queue, "max-size-bytes", 0)
        self._set_if(queue, "max-size-time", 0)
        self._set_if(queue, "leaky", 2)
        self._set_if(queue, "silent", True)
        self._set_if(converter, "gpu-id", self.gpu_id)
        self._set_if(converter, "interpolation-method", 1)
        capsfilter.set_property(
            "caps",
            self.Gst.Caps.from_string(
                "video/x-raw,format=BGRx,"
                f"width={self._res_width},height={self._res_height},"
                "pixel-aspect-ratio=1/1"
            ),
        )
        appsink.set_property("emit-signals", True)
        appsink.set_property("sync", False)
        appsink.set_property("async", False)
        appsink.set_property("drop", True)
        appsink.set_property("max-buffers", 1)
        self._set_if(appsink, "enable-last-sample", False)
        self._set_if(appsink, "wait-on-eos", False)

        for element in (queue, converter, capsfilter, appsink):
            self.pipeline.add(element)

        tee_pad = tee.request_pad_simple("src_%u")
        if tee_pad is None:
            raise RuntimeError(f"{cid}: resolution capture tee pad allocation failed")
        if tee_pad.link(queue.get_static_pad("sink")) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: tee -> resolution capture queue failed")
        self._res_request_pads.append((tee, tee_pad))

        if not queue.link(converter) or not converter.link(capsfilter) or not capsfilter.link(appsink):
            raise RuntimeError(f"{cid}: resolution capture branch link failed")

        queue.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._resolution_gate_probe,
            cid,
        )
        appsink.connect("new-sample", self._on_resolution_sample, cid)

    def _resolution_gate_probe(self, _pad, _info, cid: str):
        with self._res_lock:
            if cid in self._res_done or cid in self._res_inflight:
                return self.Gst.PadProbeReturn.DROP
            self._res_inflight.add(cid)
        print(f"CAMERA_RES_CAPTURE_GATE cid={cid} first_buffer=1", flush=True)
        return self.Gst.PadProbeReturn.OK

    def _on_resolution_sample(self, sink, cid: str):
        sample = sink.emit("pull-sample")
        if sample is None:
            with self._res_lock:
                self._res_inflight.discard(cid)
            return self.Gst.FlowReturn.OK

        structure = sample.get_caps().get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            with self._res_lock:
                self._res_inflight.discard(cid)
            return self.Gst.FlowReturn.OK

        try:
            needed = width * height * 4
            raw = np.frombuffer(mapped.data, dtype=np.uint8, count=needed)
            frame = raw.reshape((height, width, 4))[..., :3].copy()
        finally:
            buffer.unmap(mapped)

        npy_path = self._res_dir / f"{cid}_source_{width}x{height}.npy"
        json_path = npy_path.with_suffix(".json")
        np.save(npy_path, frame, allow_pickle=False)
        json_path.write_text(
            json.dumps(
                {
                    "camera": cid,
                    "width": width,
                    "height": height,
                    "dtype": str(frame.dtype),
                    "shape": list(frame.shape),
                    "bgr_mean": [float(v) for v in frame.mean(axis=(0, 1))],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        with self._res_lock:
            self._res_inflight.discard(cid)
            self._res_done.add(cid)
            complete = self._res_targets.issubset(self._res_done)

        print(
            "CAMERA_RES_CAPTURE_SAVED "
            f"cid={cid} shape={width}x{height} path={npy_path}",
            flush=True,
        )
        if complete:
            print(
                f"CAMERA_RES_CAPTURE complete=1 dir={self._res_dir}",
                flush=True,
            )
        return self.Gst.FlowReturn.OK


def main() -> int:
    return DetectionResolutionCapture().run()


if __name__ == "__main__":
    raise SystemExit(main())
