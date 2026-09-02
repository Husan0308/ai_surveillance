from __future__ import annotations

import os
import time
from pathlib import Path

from services.camera_v11.deepstream_trt86_multi_ui_v1 import V11DeepStreamTRT86MultiCameraUIV1
from services.camera_v11.deepstream_trt86_multi_v1 import map_detector_boxes_to_display


DEFAULT_TRACK_CAMERAS = ("CAM-01",)
DEFAULT_TRACKER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
DEFAULT_TRACKER_CONFIG = str(
    Path(__file__).resolve().parents[2] / "config" / "camera_v11_bbox_nvdcf_cam01_v1.yml"
)


class V11DeepStreamTRT86NvDCFBBoxCam01V1(V11DeepStreamTRT86MultiCameraUIV1):
    """Add CAM-01 local NvDCF tracking without changing the frozen detector runtime.

    The shared TRT detector still runs sparsely. A detector snapshot is injected
    into DeepStream metadata exactly once per detector sequence and explicitly
    marks bInferDone on those frames. NvDCF then owns display-rate bbox updates
    between detector corrections.
    """

    def __init__(self) -> None:
        raw = os.environ.get("V11_BBOX_TRACK_CAMERAS", ",".join(DEFAULT_TRACK_CAMERAS))
        requested = tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
        self.bbox_track_cameras = requested or DEFAULT_TRACK_CAMERAS
        self.tracker_width = max(320, int(os.environ.get("V11_BBOX_TRACKER_WIDTH", "512")))
        self.tracker_height = max(192, int(os.environ.get("V11_BBOX_TRACKER_HEIGHT", "288")))
        self.tracker_ll_lib = os.environ.get("V11_BBOX_TRACKER_LL_LIB", DEFAULT_TRACKER_LIB)
        self.tracker_config = os.environ.get("V11_BBOX_TRACKER_CONFIG", DEFAULT_TRACKER_CONFIG)
        self._nvdcf_last_injected_sequence = {cid: 0 for cid in self.bbox_track_cameras}
        self._nvdcf_detector_corrections = {cid: 0 for cid in self.bbox_track_cameras}
        self._nvdcf_injection_errors = {cid: 0 for cid in self.bbox_track_cameras}
        self._nvdcf_tracker_errors = {cid: 0 for cid in self.bbox_track_cameras}
        self._nvdcf_tracker_frames = {cid: 0 for cid in self.bbox_track_cameras}
        self._nvdcf_visible_last = {cid: -1 for cid in self.bbox_track_cameras}
        self._nvdcf_visible_max = {cid: 0 for cid in self.bbox_track_cameras}
        super().__init__()

        missing = [cid for cid in self.bbox_track_cameras if cid not in self.states]
        if missing:
            raise RuntimeError(f"bbox tracker cameras not configured: {','.join(missing)}")

        print(
            "CAMERA_V11_BBOX_NVDCF_ARCH "
            f"cameras={','.join(self.bbox_track_cameras)} tracker=nvdcf-local-only "
            f"tracker_size={self.tracker_width}x{self.tracker_height} "
            "reid=0 global_id=0 detector=shared-trt86 detector_rtsp=0 "
            "detector_metadata=once-per-sequence infer_done=explicit osd_source=nvtracker",
            flush=True,
        )

    def _preflight(self) -> None:
        super()._preflight()
        if self.Gst.ElementFactory.find("nvtracker") is None:
            raise RuntimeError("missing DeepStream plugin: nvtracker")
        if not Path(self.tracker_ll_lib).is_file():
            raise RuntimeError(f"NvDCF low-level library missing: {self.tracker_ll_lib}")
        if not Path(self.tracker_config).is_file():
            raise RuntimeError(f"NvDCF config missing: {self.tracker_config}")
        if self.tracker_width % 32 != 0 or self.tracker_height % 32 != 0:
            raise RuntimeError(
                f"NvDCF tracker resolution must be multiples of 32: "
                f"{self.tracker_width}x{self.tracker_height}"
            )

    def _build_camera(self, state) -> None:
        super()._build_camera(state)
        cid = state.camera.camera_id
        if cid not in self.bbox_track_cameras:
            return

        safe = cid.lower().replace("-", "_")
        pipeline = state.pipeline
        display_caps = pipeline.get_by_name(f"display_caps_{safe}")
        osd = pipeline.get_by_name(f"osd_{safe}")
        if display_caps is None or osd is None:
            raise RuntimeError(f"{cid}: could not resolve display caps/osd for NvDCF insertion")

        tracker = self._make("nvtracker", f"bbox_nvtracker_{safe}")
        self._set_if(tracker, "tracker-width", self.tracker_width)
        self._set_if(tracker, "tracker-height", self.tracker_height)
        self._set_if(tracker, "gpu-id", self.gpu_id)
        self._set_if(tracker, "ll-lib-file", self.tracker_ll_lib)
        self._set_if(tracker, "ll-config-file", self.tracker_config)
        self._set_if(tracker, "enable-batch-process", True)
        self._set_if(tracker, "display-tracking-id", False)
        self._set_if(tracker, "enable-past-frame", True)

        display_caps.unlink(osd)
        pipeline.add(tracker)
        if not display_caps.link(tracker):
            raise RuntimeError(f"{cid}: display_caps->nvtracker link failed")
        if not tracker.link(osd):
            raise RuntimeError(f"{cid}: nvtracker->osd link failed")

        tracker_src = tracker.get_static_pad("src")
        if tracker_src is None:
            raise RuntimeError(f"{cid}: nvtracker src pad missing")
        tracker_src.add_probe(self.Gst.PadProbeType.BUFFER, self._tracker_output_probe, cid)

        state.bbox_nvtracker = tracker
        print(
            "CAMERA_V11_BBOX_NVDCF_PIPELINE "
            f"camera={cid} path=mux->detector-meta-once->rgba->nvtracker->style->nvdsosd "
            f"tracker_size={self.tracker_width}x{self.tracker_height} config={self.tracker_config}",
            flush=True,
        )

    def _display_meta_probe(self, _pad, info, cid: str):
        if cid not in self.bbox_track_cameras:
            return super()._display_meta_probe(_pad, info, cid)

        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK

        now = time.monotonic()
        with self.lock:
            state = self.states[cid]
            snapshot = state.latest_snapshot
            last_sequence = self._nvdcf_last_injected_sequence.get(cid, 0)
            if snapshot.sequence <= last_sequence:
                return self.Gst.PadProbeReturn.OK
            age = now - snapshot.completed_mono if snapshot.completed_mono > 0 else 999.0
            boxes = snapshot.boxes if age <= self.box_stale_sec else ()
            sequence = snapshot.sequence

        scaled = map_detector_boxes_to_display(boxes, self.width, self.height) if boxes else []

        try:
            # Important: apply_detector_result marks NvDsFrameMeta.bInferDone=TRUE.
            # NvDCF relies on that bit to distinguish a real detector cycle from
            # the display frames where our sparse external detector did not run.
            # This call is required even for an empty detector result.
            added = self.meta_bridge.apply_detector_result(buffer, 0, scaled)
            if added < 0:
                raise RuntimeError(f"detector result metadata apply returned {added}")
            with self.lock:
                state.metadata_added += int(added)
                self._nvdcf_last_injected_sequence[cid] = sequence
                self._nvdcf_detector_corrections[cid] += 1
                corrections = self._nvdcf_detector_corrections[cid]
            if corrections <= 5 or corrections % 20 == 0:
                print(
                    "CAMERA_V11_BBOX_NVDCF_CORRECTION "
                    f"camera={cid} sequence={sequence} raw_boxes={len(boxes)} added={added} "
                    f"infer_done=1 age_ms={age * 1000.0:.1f} corrections={corrections}",
                    flush=True,
                )
        except Exception as exc:
            with self.lock:
                state.meta_errors += 1
                self._nvdcf_injection_errors[cid] += 1
                errors = self._nvdcf_injection_errors[cid]
            if errors <= 5 or errors % 100 == 0:
                print(
                    "CAMERA_V11_BBOX_NVDCF_META "
                    f"camera={cid} warning={type(exc).__name__}:{exc} errors={errors}",
                    flush=True,
                )
        return self.Gst.PadProbeReturn.OK

    def _tracker_output_probe(self, _pad, info, cid: str):
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK
        try:
            visible = self.meta_bridge.style_and_count_tracked(buffer)
            if visible < 0:
                raise RuntimeError(f"style_and_count_tracked returned {visible}")
            with self.lock:
                self._nvdcf_tracker_frames[cid] += 1
                tracker_frames = self._nvdcf_tracker_frames[cid]
                previous = self._nvdcf_visible_last[cid]
                self._nvdcf_visible_last[cid] = int(visible)
                self._nvdcf_visible_max[cid] = max(self._nvdcf_visible_max[cid], int(visible))
            if tracker_frames <= 5 or visible != previous or tracker_frames % 100 == 0:
                print(
                    "CAMERA_V11_BBOX_NVDCF_TRACK "
                    f"camera={cid} visible={visible} tracker_frames={tracker_frames} "
                    f"visible_max={self._nvdcf_visible_max[cid]}",
                    flush=True,
                )
        except Exception as exc:
            with self.lock:
                self._nvdcf_tracker_errors[cid] += 1
                errors = self._nvdcf_tracker_errors[cid]
            if errors <= 5 or errors % 100 == 0:
                print(
                    "CAMERA_V11_BBOX_NVDCF_TRACKER "
                    f"camera={cid} warning={type(exc).__name__}:{exc} errors={errors}",
                    flush=True,
                )
        return self.Gst.PadProbeReturn.OK


def main() -> int:
    return V11DeepStreamTRT86NvDCFBBoxCam01V1().run()


if __name__ == "__main__":
    raise SystemExit(main())
