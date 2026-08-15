from __future__ import annotations

import os
import queue as pyqueue
import threading
import time
from pathlib import Path

from .detection import CameraDetectionV2, INFER_HEIGHT, INFER_WIDTH, MICRO_BATCH


class CameraPersonTrackingV2(CameraDetectionV2):
    """Camera V2 + sparse YOLO26m detections + DeepStream NvDCF tracking.

    The known-good camera path remains GPU/NVMM. YOLO detections are injected only
    once when a fresh detector result arrives. NvDCF then owns per-frame bbox
    propagation on the live video, which avoids the stale custom predictor and
    duplicate local tracks used by the earlier detection-only experiment.
    """

    def __init__(self) -> None:
        self.pending_lock = threading.RLock()
        self.pending: dict[str, tuple[int, list[tuple[float, float, float, float, float]]]] = {}
        self.injected_seq: dict[str, int] = {}
        self.pending_seq = 0
        self.tracked_now = 0
        self.tracker_frames = 0
        self.tracker_width = max(320, int(os.environ.get("CAMERA_V2_TRACKER_WIDTH", str(INFER_WIDTH))))
        self.tracker_height = max(192, int(os.environ.get("CAMERA_V2_TRACKER_HEIGHT", str(INFER_HEIGHT))))
        # NvDCF requires multiples of 32 for the default CUDA crop-scaler path.
        self.tracker_width = ((self.tracker_width + 31) // 32) * 32
        self.tracker_height = ((self.tracker_height + 31) // 32) * 32
        self.box_side_margin = float(os.environ.get("CAMERA_V2_TRACK_BOX_SIDE_MARGIN", "0.06"))
        self.box_top_margin = float(os.environ.get("CAMERA_V2_TRACK_BOX_TOP_MARGIN", "0.04"))
        self.box_bottom_margin = float(os.environ.get("CAMERA_V2_TRACK_BOX_BOTTOM_MARGIN", "0.07"))
        self.dedup_iou = float(os.environ.get("CAMERA_V2_DEDUP_IOU", "0.52"))
        self.dedup_containment = float(os.environ.get("CAMERA_V2_DEDUP_CONTAINMENT", "0.82"))
        self.tracker_lib, self.tracker_config = self._resolve_tracker_files()
        super().__init__()

    @staticmethod
    def _deepstream_roots() -> list[Path]:
        roots = [Path("/opt/nvidia/deepstream/deepstream")]
        roots.extend(sorted(Path("/opt/nvidia/deepstream").glob("deepstream-*"), reverse=True))
        seen: set[str] = set()
        output: list[Path] = []
        for p in roots:
            key = str(p)
            if key not in seen:
                seen.add(key)
                output.append(p)
        return output

    def _resolve_tracker_files(self) -> tuple[Path, Path]:
        lib_env = os.environ.get("CAMERA_V2_TRACKER_LIB", "").strip()
        cfg_env = os.environ.get("CAMERA_V2_TRACKER_CONFIG", "").strip()
        lib = Path(lib_env) if lib_env else None
        cfg = Path(cfg_env) if cfg_env else None

        if lib is None:
            lib = next(
                (
                    root / "lib/libnvds_nvmultiobjecttracker.so"
                    for root in self._deepstream_roots()
                    if (root / "lib/libnvds_nvmultiobjecttracker.so").exists()
                ),
                None,
            )
        if cfg is None:
            cfg = next(
                (
                    root / "samples/configs/deepstream-app/config_tracker_NvDCF_max_perf.yml"
                    for root in self._deepstream_roots()
                    if (root / "samples/configs/deepstream-app/config_tracker_NvDCF_max_perf.yml").exists()
                ),
                None,
            )

        if lib is None or not lib.exists():
            raise RuntimeError(
                "NvDCF low-level tracker library not found: libnvds_nvmultiobjecttracker.so"
            )
        if cfg is None or not cfg.exists():
            raise RuntimeError(
                "NvDCF max-perf config not found: config_tracker_NvDCF_max_perf.yml"
            )
        return lib, cfg

    @staticmethod
    def _area(box) -> float:
        x1, y1, x2, y2 = box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _intersection(a, b) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _det_iou(self, a, b) -> float:
        inter = self._intersection(a, b)
        if inter <= 0:
            return 0.0
        union = self._area(a) + self._area(b) - inter
        return inter / union if union > 0 else 0.0

    def _dedup_and_expand(self, rows):
        """Second-stage person de-dup + conservative full-body margin.

        Ultralytics already applies NMS. This extra pass targets nested/near-identical
        person boxes that can otherwise seed two NvDCF targets for one person.
        """
        scaled = self._scaled_detections(rows)
        ordered = sorted(scaled, key=lambda row: float(row[1]), reverse=True)
        kept: list[tuple[tuple[float, float, float, float], float]] = []
        for box, conf in ordered:
            duplicate = False
            area = max(1.0, self._area(box))
            for other, _ in kept:
                inter = self._intersection(box, other)
                smaller = max(1.0, min(area, self._area(other)))
                if self._det_iou(box, other) >= self.dedup_iou or inter / smaller >= self.dedup_containment:
                    duplicate = True
                    break
            if duplicate:
                continue
            kept.append((box, conf))

        output: list[tuple[float, float, float, float, float]] = []
        for (x1, y1, x2, y2), conf in kept:
            w = max(2.0, x2 - x1)
            h = max(2.0, y2 - y1)
            x1 -= w * self.box_side_margin
            x2 += w * self.box_side_margin
            y1 -= h * self.box_top_margin
            y2 += h * self.box_bottom_margin
            output.append(
                (
                    max(0.0, x1),
                    max(0.0, y1),
                    min(float(self.frame_width - 1), x2),
                    min(float(self.frame_height - 1), y2),
                    float(conf),
                )
            )
        return output

    def _publish_detector_result(self, cid: str, boxes) -> None:
        # Empty result is intentionally not injected. NvDCF shadow tracking keeps
        # an existing target alive until its own termination policy decides it is gone.
        if not boxes:
            return
        with self.pending_lock:
            self.pending_seq += 1
            self.pending[cid] = (self.pending_seq, list(boxes))

    def _install_osd_and_meta(self) -> None:
        if self.Gst.ElementFactory.find("nvtracker") is None:
            raise RuntimeError("DeepStream nvtracker plugin is missing")

        # Insert NvDCF immediately after nvstreammux so it sees detector metadata
        # and the full NVMM video frames before the tiler changes geometry.
        self.mux.unlink(self.tiler)
        tracker = self._make("nvtracker", "person_nvdcf_tracker")
        self._set_if(tracker, "tracker-width", self.tracker_width)
        self._set_if(tracker, "tracker-height", self.tracker_height)
        tracker.set_property("ll-lib-file", str(self.tracker_lib))
        tracker.set_property("ll-config-file", str(self.tracker_config))
        self._set_if(tracker, "gpu-id", self.gpu_id)
        self._set_if(tracker, "compute-hw", 1)
        self._set_if(tracker, "enable-batch-process", True)
        self._set_if(tracker, "display-tracking-id", False)
        self._set_if(tracker, "tracking-id-reset-mode", 1)
        self._set_if(tracker, "user-meta-pool-size", 64)
        self.pipeline.add(tracker)
        if not self.mux.link(tracker):
            raise RuntimeError("failed nvstreammux -> nvtracker")
        if not tracker.link(self.tiler):
            raise RuntimeError("failed nvtracker -> nvmultistreamtiler")

        # Preserve the same final GPU OSD path used by the working detection test.
        self.wall_queue.unlink(self.sink)
        convert = self._make("nvvideoconvert", "track_wall_convert")
        caps = self._make("capsfilter", "track_wall_caps")
        osd = self._make("nvdsosd", "track_osd")
        self._set_if(convert, "gpu-id", self.gpu_id)
        caps.set_property(
            "caps",
            self.Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"),
        )
        self._set_if(osd, "process-mode", 1)
        self._set_if(osd, "display-bbox", True)
        self._set_if(osd, "display-text", False)
        self._set_if(osd, "display-mask", False)
        self._set_if(osd, "gpu-id", self.gpu_id)
        for element in (convert, caps, osd):
            self.pipeline.add(element)
        if not self.wall_queue.link(convert):
            raise RuntimeError("failed wall queue -> nvvideoconvert")
        if not convert.link(caps):
            raise RuntimeError("failed nvvideoconvert -> RGBA caps")
        if not caps.link(osd):
            raise RuntimeError("failed RGBA caps -> nvdsosd")
        if not osd.link(self.sink):
            raise RuntimeError("failed nvdsosd -> nveglglessink")

        self.mux.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._inject_detector_probe,
        )
        tracker.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._tracker_probe,
        )
        osd.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._wall_probe,
        )
        self.tracker = tracker
        self.osd = osd

    def _inject_detector_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK
        added_total = 0
        with self.pending_lock:
            pending = dict(self.pending)
        for cid, source_id in self.camera_index.items():
            row = pending.get(cid)
            if row is None:
                continue
            seq, boxes = row
            if seq <= self.injected_seq.get(cid, 0):
                continue
            added = self.bridge.add_boxes(buffer, source_id, boxes)
            # added==0 can mean this partial mux batch did not contain this source;
            # keep it pending for the next batch instead of losing the detection.
            if added > 0:
                self.injected_seq[cid] = seq
                added_total += added
        if added_total:
            with self.det_lock:
                self.meta_boxes += added_total
        return self.Gst.PadProbeReturn.OK

    def _tracker_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is not None:
            count = self.bridge.count_tracked(buffer)
            if count >= 0:
                with self.det_lock:
                    self.tracked_now = count
                    self.tracker_frames += 1
        return self.Gst.PadProbeReturn.OK

    def _scheduler(self) -> None:
        assert self.result_q is not None and self.job_q is not None
        try:
            ready = self.result_q.get(timeout=40.0)
        except pyqueue.Empty:
            with self.det_lock:
                self.det_error = "YOLO worker startup timeout"
            return
        if ready.get("type") != "ready":
            with self.det_lock:
                self.det_error = ready.get("error", "YOLO worker failed")
            return
        with self.det_lock:
            self.det_ready = True
        print(
            "CAMERA_TRACK ready: "
            f"YOLO26m micro_batch={MICRO_BATCH} input={INFER_WIDTH}x{INFER_HEIGHT} "
            f"NvDCF={self.tracker_width}x{self.tracker_height} "
            f"device={ready.get('device')} cuda={ready.get('cuda')} "
            "detector_once=1 nvdcf_per_frame=1 dedup=1",
            flush=True,
        )

        ids = [camera.camera_id for camera in self.cameras]
        groups = [ids[i : i + MICRO_BATCH] for i in range(0, len(ids), MICRO_BATCH)]
        versions = {cid: 0 for cid in ids}
        group_index = 0

        while not self.det_stop.is_set():
            group = groups[group_index % len(groups)]
            group_index += 1
            self._request_group(group)
            rows = self.mailbox.wait_group(group, versions, timeout=1.2)
            if rows is None:
                self._clear_requests()
                with self.det_lock:
                    self.capture_timeouts += 1
                self.det_stop.wait(0.10)
                continue

            frames = []
            captured = []
            for cid, row in zip(group, rows):
                version, captured_t, frame = row
                versions[cid] = version
                captured.append(captured_t)
                frames.append(frame)
            self._clear_requests()

            try:
                self.job_q.put(
                    {"cameras": group, "frames": frames, "captured": captured},
                    timeout=0.5,
                )
                result = self.result_q.get(timeout=8.0)
            except pyqueue.Empty:
                with self.det_lock:
                    self.det_error = "YOLO result timeout"
                self.det_stop.wait(0.25)
                continue

            if result.get("type") == "fatal":
                with self.det_lock:
                    self.det_error = result.get("error", "YOLO fatal error")
                return
            if result.get("type") == "batch_error":
                with self.det_lock:
                    self.det_error = result.get("error", "YOLO batch error")
                self.det_stop.wait(0.50)
                continue
            if result.get("type") != "result":
                continue

            counts: dict[str, int] = {}
            for cid in result["cameras"]:
                detections = self._dedup_and_expand(result["boxes"].get(cid, []))
                counts[cid] = len(detections)
                self._publish_detector_result(cid, detections)

            batch_ms = float(result.get("batch_ms") or 0.0)
            with self.det_lock:
                self.det_calls += 1
                self.det_inputs += len(group)
                self.det_batch_ms = batch_ms
                self.det_counts.update(counts)
                duty = max(self.det_duty_min, min(self.det_duty_max, self.det_duty))
                self.det_error = ""

            # Camera smoothness remains the priority. NvDCF fills the detector gaps,
            # so there is no need to push YOLO duty aggressively.
            active = batch_ms / 1000.0
            idle = max(0.04, active * (1.0 / max(0.05, duty) - 1.0))
            self.det_stop.wait(min(1.5, idle))

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        with self.det_lock:
            tracked = self.tracked_now
            frames = self.tracker_frames
        print(
            "CAMERA_TRACK "
            f"algorithm=NvDCF tracker={self.tracker_width}x{self.tracker_height} "
            f"tracked_now={tracked} tracker_batches={frames} "
            f"detector_injected={self.meta_boxes} custom_predictor=0",
            flush=True,
        )
        return keep


def main() -> int:
    return CameraPersonTrackingV2().run()


if __name__ == "__main__":
    raise SystemExit(main())
