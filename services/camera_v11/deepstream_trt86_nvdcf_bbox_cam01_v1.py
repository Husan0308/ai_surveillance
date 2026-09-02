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
FRESH_OVERLAP_DIAGNOSTIC_SEC = 0.12
MATCH_IOU_FOR_DIAGNOSTIC = 0.20
NESTED_CENTER_MARGIN = 0.04


def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
    bx1, by1, bx2, by2 = (float(v) for v in b[:4])
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _track_xyxy(row: dict) -> tuple[float, float, float, float]:
    left = float(row["left"])
    top = float(row["top"])
    return (
        left,
        top,
        left + float(row["width"]),
        top + float(row["height"]),
    )


def _center_inside(inner, outer) -> bool:
    ix1, iy1, ix2, iy2 = (float(v) for v in inner[:4])
    ox1, oy1, ox2, oy2 = (float(v) for v in outer[:4])
    ow = max(1.0, ox2 - ox1)
    oh = max(1.0, oy2 - oy1)
    cx = (ix1 + ix2) * 0.5
    cy = (iy1 + iy2) * 0.5
    mx = ow * NESTED_CENTER_MARGIN
    my = oh * NESTED_CENTER_MARGIN
    return (ox1 + mx) <= cx <= (ox2 - mx) and (oy1 + my) <= cy <= (oy2 - my)


def _spatial_match_counts(detector_boxes, tracks) -> tuple[int, int, int]:
    """Greedy one-to-one detector->track diagnostic matching.

    Returns (matched, unmatched, nested_unmatched). This never alters tracker
    metadata; it only tells us whether a fresh detector person has no distinct
    NvDCF output, including the important case where that detector's center is
    inside another already-matched person's tracker rectangle.
    """
    if not detector_boxes:
        return 0, 0, 0
    track_boxes = [_track_xyxy(row) for row in tracks]
    candidates = []
    for di, det in enumerate(detector_boxes):
        for ti, track in enumerate(track_boxes):
            score = _iou_xyxy(det, track)
            if score >= MATCH_IOU_FOR_DIAGNOSTIC:
                candidates.append((score, di, ti))
    candidates.sort(reverse=True)
    used_dets: set[int] = set()
    used_tracks: set[int] = set()
    for _score, di, ti in candidates:
        if di in used_dets or ti in used_tracks:
            continue
        used_dets.add(di)
        used_tracks.add(ti)

    unmatched_indexes = [i for i in range(len(detector_boxes)) if i not in used_dets]
    nested_unmatched = 0
    for di in unmatched_indexes:
        det = detector_boxes[di]
        if any(_center_inside(det, track) for track in track_boxes):
            nested_unmatched += 1
    return len(used_dets), len(unmatched_indexes), nested_unmatched


class V11DeepStreamTRT86NvDCFBBoxCam01V1(V11DeepStreamTRT86MultiCameraUIV1):
    """Add camera-local NvDCF tracking without changing the frozen detector runtime.

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
        self._nvdcf_overlap_gap_events = {cid: 0 for cid in self.bbox_track_cameras}
        self._nvdcf_nested_gap_events = {cid: 0 for cid in self.bbox_track_cameras}
        self._nvdcf_last_ids = {cid: () for cid in self.bbox_track_cameras}
        super().__init__()

        missing = [cid for cid in self.bbox_track_cameras if cid not in self.states]
        if missing:
            raise RuntimeError(f"bbox tracker cameras not configured: {','.join(missing)}")

        print(
            "CAMERA_V11_BBOX_NVDCF_ARCH "
            f"cameras={','.join(self.bbox_track_cameras)} tracker=nvdcf-local-only "
            f"tracker_size={self.tracker_width}x{self.tracker_height} "
            "reid=0 global_id=0 detector=shared-trt86 detector_rtsp=0 "
            "detector_metadata=once-per-sequence infer_done=explicit osd_source=nvtracker "
            "overlap_suppression=0 overlap_diag=fresh-spatial-one-to-one",
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

            inspect_ids = tracker_frames <= 5 or visible != previous or tracker_frames % 20 == 0
            if inspect_ids:
                tracks = self.meta_bridge.copy_tracks(buffer, max_rows=32)
                track_ids = tuple(sorted({int(row["object_id"]) for row in tracks}))
                now = time.monotonic()
                with self.lock:
                    state = self.states[cid]
                    snapshot = state.latest_snapshot
                    detector_age = (
                        now - snapshot.completed_mono if snapshot.completed_mono > 0 else 999.0
                    )
                    detector_fresh = detector_age <= FRESH_OVERLAP_DIAGNOSTIC_SEC
                    detector_visible = len(snapshot.boxes) if detector_fresh else -1
                    detector_boxes = tuple(snapshot.boxes) if detector_fresh else ()

                scaled_detector = (
                    map_detector_boxes_to_display(detector_boxes, self.width, self.height)
                    if detector_boxes
                    else []
                )
                matched, unmatched, nested_unmatched = _spatial_match_counts(
                    scaled_detector, tracks
                )

                with self.lock:
                    if detector_fresh and unmatched > 0:
                        self._nvdcf_overlap_gap_events[cid] += 1
                    if detector_fresh and nested_unmatched > 0:
                        self._nvdcf_nested_gap_events[cid] += 1
                    overlap_gaps = self._nvdcf_overlap_gap_events[cid]
                    nested_gaps = self._nvdcf_nested_gap_events[cid]
                    ids_changed = track_ids != self._nvdcf_last_ids[cid]
                    self._nvdcf_last_ids[cid] = track_ids

                if (
                    ids_changed
                    or (detector_fresh and detector_visible >= 2)
                    or tracker_frames <= 5
                    or tracker_frames % 100 == 0
                ):
                    ids_text = ",".join(str(value) for value in track_ids) if track_ids else "none"
                    print(
                        "CAMERA_V11_BBOX_NVDCF_IDS "
                        f"camera={cid} tracker_visible={visible} detector_visible={detector_visible} "
                        f"detector_fresh={int(detector_fresh)} matched={matched} unmatched={unmatched} "
                        f"nested_unmatched={nested_unmatched} ids={ids_text} "
                        f"overlap_gap_events={overlap_gaps} nested_gap_events={nested_gaps} "
                        f"detector_age_ms={detector_age * 1000.0:.1f} tracker_frames={tracker_frames}",
                        flush=True,
                    )

            if tracker_frames <= 5 or visible != previous or tracker_frames % 100 == 0:
                print(
                    "CAMERA_V11_BBOX_NVDCF_TRACK "
                    f"camera={cid} visible={visible} tracker_frames={tracker_frames} "
                    f"visible_max={self._nvdcf_visible_max[cid]} "
                    f"overlap_gap_events={self._nvdcf_overlap_gap_events[cid]} "
                    f"nested_gap_events={self._nvdcf_nested_gap_events[cid]}",
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
