from __future__ import annotations

import os

# Smoothness-first defaults for GTX 1050 Ti. Explicit env overrides still win.
os.environ.setdefault("AI_YOLO_START_BATCH_FPS", "1.00")
os.environ.setdefault("AI_YOLO_MAX_BATCH_FPS", "1.50")
os.environ.setdefault("AI_YOLO_MAX_GPU_DUTY", "0.25")
os.environ.setdefault("AI_YOLO_CONF", "0.20")
os.environ.setdefault("AI_CAMERA_FPS_FLOOR", "19.0")
os.environ.setdefault("AI_CAMERA_FPS_GOOD", "19.7")

import json
import queue
import re
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import yaml

from . import deepstream_yolo26m_batch6_wall as base

ROOT = Path(__file__).resolve().parents[3]


def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0.0 else 0.0


def _load_core_cfg() -> dict:
    try:
        payload = yaml.safe_load((ROOT / "config/core_v1.yaml").read_text(encoding="utf-8")) or {}
        return dict(payload.get("core_v1") or {})
    except Exception:
        return {}


class NativeCameraYolo26mFull(base.NativeCameraYolo26mBatch6Wall):
    """Camera-smoothness-first full analytics wall.

    Hot display path:
        RTSP -> NVDEC -> tee -> latest-only display queue -> nvstreammux
             -> [custom detector meta] -> NvDCF -> tiler -> GPU nvdsosd -> EGL

    Heavy analytics stay off the display path:
        YOLO26m: strict 6-camera PyTorch CUDA batch, duty capped.
        ReID: OSNet CPU, low-rate crops reused from YOLO sidecar.
        Face: InsightFace CPU, only when a face gallery exists.
        Heatmap: tracker bottom-center accumulation, CPU-light.

    Every optional stage fails open: camera wall stays alive.
    """

    def __init__(self):
        self.pyds = None
        self.tracker = None
        self.osd = None
        self.tracker_enabled = False
        self.osd_enabled = False
        self._last_injected_capture: dict[str, float] = {}
        self._injected_records: dict[tuple[int, int], dict] = {}
        self._injected_lock = threading.Lock()

        self.identity_lock = threading.RLock()
        self.identity_queue: queue.Queue = queue.Queue(maxsize=32)
        self.identity_thread: threading.Thread | None = None
        self._identity_last_enqueued: dict[tuple[str, int], float] = {}
        self._reid_embeddings: dict[tuple[str, int], deque] = defaultdict(lambda: deque(maxlen=5))
        self._global_ids: dict[tuple[str, int], str] = {}
        self._known_names: dict[tuple[str, int], str] = {}
        self._global_names: dict[str, str] = {}
        self._face_pending: dict[tuple[str, int], tuple[str, int, float]] = {}
        self._last_face_attempt: dict[tuple[str, int], float] = {}
        self._reid_samples = 0
        self._reid_errors = 0
        self._face_attempts = 0
        self._face_matches = 0
        self._identity_drops = 0
        self._identity_error = ""

        self._tracked_last: dict[str, int] = {}
        self._track_frames = 0
        self._meta_injections = 0
        self._tracker_matches_for_identity = 0
        self._heatmap = {
            f"CAM-{index:02d}": np.zeros((18, 32), dtype=np.uint64)
            for index in range(1, 7)
        }

        self.identity_sample_interval = max(
            0.5, float(os.environ.get("AI_IDENTITY_SAMPLE_INTERVAL", "1.20"))
        )
        self.face_sample_interval = max(
            1.0, float(os.environ.get("AI_FACE_SAMPLE_INTERVAL", "2.50"))
        )
        self.meta_max_age_ms = max(
            100.0, float(os.environ.get("AI_DETECTION_META_MAX_AGE_MS", "800"))
        )
        self.tracker_width = max(
            160, int(os.environ.get("AI_TRACKER_WIDTH", "480"))
        )
        self.tracker_height = max(
            96, int(os.environ.get("AI_TRACKER_HEIGHT", "288"))
        )
        self.osd_text = os.environ.get("AI_OSD_TEXT", "1").strip().lower() not in {
            "0", "false", "no"
        }

        super().__init__()
        self._setup_metadata_pipeline()

        self.identity_thread = threading.Thread(
            target=self._identity_loop,
            name="identity-sidecar",
            daemon=True,
        )
        self.identity_thread.start()

    def _find_deepstream_file(self, relative: str) -> Path | None:
        candidates = [
            Path("/opt/nvidia/deepstream/deepstream") / relative,
        ]
        candidates.extend(
            sorted(Path("/opt/nvidia/deepstream").glob(f"deepstream-*/{relative}"))
        )
        for path in candidates:
            if path.exists():
                return path
        return None

    def _patched_tracker_config(self) -> Path | None:
        source = self._find_deepstream_file(
            "samples/configs/deepstream-app/config_tracker_NvDCF_max_perf.yml"
        )
        if source is None:
            return None
        try:
            text = source.read_text(encoding="utf-8")
            replacements = {
                "probationAge": "0",
                "maxShadowTrackingAge": "60",
                "earlyTerminationAge": "30",
            }
            for key, value in replacements.items():
                pattern = rf"(?m)^(\s*{re.escape(key)}\s*:\s*)[^#\r\n]+"
                if re.search(pattern, text):
                    text = re.sub(pattern, rf"\g<1>{value} ", text)
            target_dir = ROOT / "data/runtime"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / "config_tracker_NvDCF_sparse_detector.yml"
            target.write_text(text, encoding="utf-8")
            return target
        except Exception as exc:
            print(
                f"FULL_PIPELINE tracker config patch failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return source

    def _setup_metadata_pipeline(self) -> None:
        try:
            import pyds
            self.pyds = pyds
        except Exception as exc:
            print(
                "FULL_PIPELINE pyds unavailable; camera+YOLO stay alive, "
                f"but boxes/tracker are disabled: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return

        # Rewire only before PLAYING: mux -> [tracker] -> tiler -> [GPU OSD] -> queue.
        try:
            self.mux.unlink(self.tiler)
        except Exception:
            pass
        try:
            self.tiler.unlink(self.wall_queue)
        except Exception:
            pass

        tracker = self.Gst.ElementFactory.make("nvtracker", "full_tracker")
        tracker_lib = self._find_deepstream_file("lib/libnvds_nvmultiobjecttracker.so")
        tracker_cfg = self._patched_tracker_config()
        if tracker is not None and tracker_lib is not None and tracker_cfg is not None:
            self.tracker = tracker
            self.pipeline.add(tracker)
            self._set_if(tracker, "tracker-width", self.tracker_width)
            self._set_if(tracker, "tracker-height", self.tracker_height)
            self._set_if(tracker, "ll-lib-file", str(tracker_lib))
            self._set_if(tracker, "ll-config-file", str(tracker_cfg))
            self._set_if(tracker, "gpu-id", 0)
            self._set_if(tracker, "enable-batch-process", True)
            self._set_if(tracker, "display-tracking-id", True)
            self._set_if(tracker, "enable-past-frame", False)
            if not self.mux.link(tracker) or not tracker.link(self.tiler):
                raise RuntimeError("failed to link mux -> nvtracker -> tiler")
            self.tracker_enabled = True
        else:
            if not self.mux.link(self.tiler):
                raise RuntimeError("failed to restore mux -> tiler fallback")
            print(
                "FULL_PIPELINE nvtracker unavailable; detection boxes will be held "
                "without visual tracking",
                flush=True,
            )

        osd = self.Gst.ElementFactory.make("nvdsosd", "full_osd")
        if osd is not None:
            self.osd = osd
            self.pipeline.add(osd)
            self._set_if(osd, "process-mode", 1)  # GPU mode, NV12/RGBA accepted.
            self._set_if(osd, "display-bbox", True)
            self._set_if(osd, "display-text", self.osd_text)
            self._set_if(osd, "display-mask", False)
            self._set_if(osd, "gpu-id", 0)
            if not self.tiler.link(osd) or not osd.link(self.wall_queue):
                raise RuntimeError("failed to link tiler -> nvdsosd -> wall queue")
            self.osd_enabled = True
        else:
            if not self.tiler.link(self.wall_queue):
                raise RuntimeError("failed to restore tiler -> wall queue fallback")
            print(
                "FULL_PIPELINE nvdsosd unavailable; metadata/tracker run but boxes "
                "cannot be rendered",
                file=sys.stderr,
                flush=True,
            )

        mux_src = self.mux.get_static_pad("src")
        mux_src.add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._inject_detector_meta_probe,
        )

        tracker_or_mux = (
            self.tracker.get_static_pad("src")
            if self.tracker_enabled
            else self.mux.get_static_pad("src")
        )
        tracker_or_mux.add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._track_and_style_probe,
        )

        print(
            "FULL_PIPELINE metadata path: "
            f"pyds=1 tracker={int(self.tracker_enabled)} "
            f"osd={int(self.osd_enabled)} tracker_size={self.tracker_width}x{self.tracker_height}",
            flush=True,
        )

    def _frame_list(self, batch_meta):
        pyds = self.pyds
        node = batch_meta.frame_meta_list
        while node is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(node.data)
            except StopIteration:
                break
            yield frame_meta
            try:
                node = node.next
            except StopIteration:
                break

    def _object_list(self, frame_meta):
        pyds = self.pyds
        node = frame_meta.obj_meta_list
        while node is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(node.data)
            except StopIteration:
                break
            yield obj_meta
            try:
                node = node.next
            except StopIteration:
                break

    def _camera_from_frame_meta(self, frame_meta) -> tuple[int, str] | None:
        index = int(getattr(frame_meta, "pad_index", getattr(frame_meta, "source_id", -1)))
        if not (0 <= index < len(self.camera_ids)):
            index = int(getattr(frame_meta, "source_id", -1))
        if not (0 <= index < len(self.camera_ids)):
            return None
        return index, self.camera_ids[index]

    def _snapshot_with_frame(self, cid: str) -> tuple[dict, object | None] | None:
        with self.det_lock:
            snapshot = self.latest_detections.get(cid)
            if not snapshot:
                return None
            snapshot = dict(snapshot)
        captured = float(snapshot.get("captured_mono") or 0.0)
        frame = None
        latest = self.latest
        if latest is not None:
            try:
                with latest._condition:
                    row = latest._frames.get(cid)
                    if row is not None and abs(float(row[1]) - captured) < 0.20:
                        frame = row[2]
            except Exception:
                frame = None
        return snapshot, frame

    def _add_object_meta(self, batch_meta, frame_meta, box, confidence: float, frame_size):
        pyds = self.pyds
        fw, fh = [max(1, int(v)) for v in frame_size]
        sx = float(self.frame_width) / fw
        sy = float(self.frame_height) / fh
        x1, y1, x2, y2 = [float(v) for v in box]
        left = max(0.0, min(float(self.frame_width - 1), x1 * sx))
        top = max(0.0, min(float(self.frame_height - 1), y1 * sy))
        right = max(left + 1.0, min(float(self.frame_width), x2 * sx))
        bottom = max(top + 1.0, min(float(self.frame_height), y2 * sy))
        width = right - left
        height = bottom - top

        obj = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
        obj.unique_component_id = 1
        obj.class_id = 0
        obj.confidence = float(confidence)
        obj.object_id = getattr(pyds, "UNTRACKED_OBJECT_ID", 0xFFFFFFFFFFFFFFFF)
        try:
            obj.obj_label = "Person"
        except Exception:
            pass

        rect = obj.rect_params
        rect.left = left
        rect.top = top
        rect.width = width
        rect.height = height
        rect.border_width = 2
        rect.border_color.set(0.0, 1.0, 0.0, 1.0)
        try:
            rect.has_bg_color = 0
        except Exception:
            pass

        try:
            detector = obj.detector_bbox_info.org_bbox_coords
            detector.left = left
            detector.top = top
            detector.width = width
            detector.height = height
        except Exception:
            pass

        pyds.nvds_add_obj_meta_to_frame(frame_meta, obj, None)

    def _inject_detector_meta_probe(self, _pad, info):
        if self.pyds is None:
            return self.Gst.PadProbeReturn.OK
        gst_buffer = info.get_buffer()
        if gst_buffer is None:
            return self.Gst.PadProbeReturn.OK
        batch_meta = self.pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta is None:
            return self.Gst.PadProbeReturn.OK

        now = time.monotonic()
        for frame_meta in self._frame_list(batch_meta):
            mapped = self._camera_from_frame_meta(frame_meta)
            if mapped is None:
                continue
            source_index, cid = mapped
            value = self._snapshot_with_frame(cid)
            if value is None:
                continue
            snapshot, lowres_frame = value
            captured = float(snapshot.get("captured_mono") or 0.0)
            age_ms = (now - captured) * 1000.0 if captured else 1e9
            last = self._last_injected_capture.get(cid, 0.0)

            if self.tracker_enabled:
                should_inject = captured > last and age_ms <= self.meta_max_age_ms
            else:
                # No tracker fallback: repeat recent detections so OSD boxes do not flicker.
                should_inject = age_ms <= self.meta_max_age_ms
            if not should_inject:
                continue

            frame_size = snapshot.get("frame_size") or [base.INFER_WIDTH, base.INFER_HEIGHT]
            boxes = list(snapshot.get("boxes") or [])
            for item in boxes:
                self._add_object_meta(
                    batch_meta,
                    frame_meta,
                    item.get("xyxy") or [0, 0, 1, 1],
                    float(item.get("confidence") or 0.0),
                    frame_size,
                )
            self._meta_injections += len(boxes)

            if captured > last:
                self._last_injected_capture[cid] = captured
                record = {
                    "cid": cid,
                    "snapshot": snapshot,
                    "frame": lowres_frame,
                    "captured_mono": captured,
                }
                key = (source_index, int(frame_meta.frame_num))
                with self._injected_lock:
                    self._injected_records[key] = record
                    # bounded cleanup
                    if len(self._injected_records) > 64:
                        for old_key in sorted(self._injected_records)[:16]:
                            self._injected_records.pop(old_key, None)

        return self.Gst.PadProbeReturn.OK

    def _track_label(self, cid: str, track_id: int) -> str:
        key = (cid, int(track_id))
        with self.identity_lock:
            gid = self._global_ids.get(key)
            name = self._known_names.get(key)
            if gid and not name:
                name = self._global_names.get(gid)
        if name and gid:
            return f"{name} · {gid}"
        if name:
            return name
        if gid:
            return gid
        return f"Person {track_id}"

    def _style_object(self, obj_meta, cid: str, track_id: int) -> None:
        rect = obj_meta.rect_params
        rect.border_width = 2
        if track_id >= 0:
            rect.border_color.set(0.0, 1.0, 0.15, 1.0)
        else:
            rect.border_color.set(1.0, 0.75, 0.0, 1.0)

        if not self.osd_text:
            return
        try:
            text = obj_meta.text_params
            text.display_text = self._track_label(cid, track_id)
            text.x_offset = max(0, int(rect.left))
            text.y_offset = max(0, int(rect.top) - 20)
            text.font_params.font_name = "Sans"
            text.font_params.font_size = 12
            text.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            text.set_bg_clr = 1
            text.text_bg_clr.set(0.0, 0.0, 0.0, 0.55)
        except Exception:
            pass

    def _heatmap_update(self, cid: str, rect) -> None:
        grid = self._heatmap.get(cid)
        if grid is None:
            return
        x = (float(rect.left) + float(rect.width) * 0.5) / max(1.0, float(self.frame_width))
        y = (float(rect.top) + float(rect.height)) / max(1.0, float(self.frame_height))
        gx = min(grid.shape[1] - 1, max(0, int(x * grid.shape[1])))
        gy = min(grid.shape[0] - 1, max(0, int(y * grid.shape[0])))
        grid[gy, gx] += 1

    def _tracker_objects(self, frame_meta, cid: str) -> list[dict]:
        result = []
        for obj in self._object_list(frame_meta):
            if int(obj.class_id) != 0:
                continue
            track_id = int(obj.object_id)
            untracked = int(getattr(self.pyds, "UNTRACKED_OBJECT_ID", 0xFFFFFFFFFFFFFFFF))
            if track_id == untracked:
                display_id = -1
            else:
                display_id = track_id
            self._style_object(obj, cid, display_id)
            rect = obj.rect_params
            self._heatmap_update(cid, rect)
            result.append(
                {
                    "track_id": display_id,
                    "xyxy": [
                        float(rect.left),
                        float(rect.top),
                        float(rect.left + rect.width),
                        float(rect.top + rect.height),
                    ],
                    "tracker_confidence": float(getattr(obj, "tracker_confidence", -1.0)),
                }
            )
        return result

    def _crop_detection(self, frame, detection) -> object | None:
        if frame is None or not hasattr(frame, "shape"):
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in detection]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 42 or crop.shape[1] < 14:
            return None
        return crop.copy()

    def _enqueue_identity_matches(self, cid: str, tracks: list[dict], record: dict) -> None:
        frame = record.get("frame")
        snapshot = record.get("snapshot") or {}
        detections = list(snapshot.get("boxes") or [])
        frame_size = snapshot.get("frame_size") or [base.INFER_WIDTH, base.INFER_HEIGHT]
        fw, fh = [max(1, int(v)) for v in frame_size]
        sx = float(self.frame_width) / fw
        sy = float(self.frame_height) / fh

        used_tracks = set()
        for item in detections:
            low_box = item.get("xyxy") or [0, 0, 1, 1]
            full_box = [
                float(low_box[0]) * sx,
                float(low_box[1]) * sy,
                float(low_box[2]) * sx,
                float(low_box[3]) * sy,
            ]
            ranked = sorted(
                (
                    (_iou_xyxy(full_box, track["xyxy"]), track)
                    for track in tracks
                    if track["track_id"] >= 0 and track["track_id"] not in used_tracks
                ),
                key=lambda value: value[0],
                reverse=True,
            )
            if not ranked or ranked[0][0] < 0.18:
                continue
            _, track = ranked[0]
            used_tracks.add(track["track_id"])
            crop = self._crop_detection(frame, low_box)
            if crop is None:
                continue
            key = (cid, int(track["track_id"]))
            now = time.monotonic()
            if now - self._identity_last_enqueued.get(key, 0.0) < self.identity_sample_interval:
                continue
            self._identity_last_enqueued[key] = now
            task = {
                "cid": cid,
                "track_id": int(track["track_id"]),
                "crop": crop,
                "det_conf": float(item.get("confidence") or 0.0),
                "captured_mono": float(record.get("captured_mono") or now),
            }
            try:
                self.identity_queue.put_nowait(task)
                self._tracker_matches_for_identity += 1
            except queue.Full:
                self._identity_drops += 1
                try:
                    self.identity_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.identity_queue.put_nowait(task)
                except queue.Full:
                    self._identity_drops += 1

    def _track_and_style_probe(self, _pad, info):
        if self.pyds is None:
            return self.Gst.PadProbeReturn.OK
        gst_buffer = info.get_buffer()
        if gst_buffer is None:
            return self.Gst.PadProbeReturn.OK
        batch_meta = self.pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta is None:
            return self.Gst.PadProbeReturn.OK

        counts = {}
        for frame_meta in self._frame_list(batch_meta):
            mapped = self._camera_from_frame_meta(frame_meta)
            if mapped is None:
                continue
            source_index, cid = mapped
            tracks = self._tracker_objects(frame_meta, cid)
            counts[cid] = len([t for t in tracks if t["track_id"] >= 0])
            key = (source_index, int(frame_meta.frame_num))
            with self._injected_lock:
                record = self._injected_records.pop(key, None)
            if record is not None and tracks:
                self._enqueue_identity_matches(cid, tracks, record)

        self._tracked_last = counts
        self._track_frames += 1
        return self.Gst.PadProbeReturn.OK

    def _identity_loop(self) -> None:
        core = _load_core_cfg()
        reid_cfg = dict(core.get("reid") or {})
        reid_cfg["enabled"] = True
        # One CPU thread protects GStreamer/RTSP scheduling on the i5.
        reid_cfg["cpu_threads"] = 1
        reid_cfg["max_batch"] = min(2, int(reid_cfg.get("max_batch", 2) or 2))

        reid = None
        face_gallery = None
        face_app = None
        face_cfg = dict(core.get("face") or {})

        try:
            from services.ml_service.core_v1.global_reid import GlobalReIdCoordinator
            reid = GlobalReIdCoordinator({}, {}, reid_cfg, ROOT)
        except Exception as exc:
            self._identity_error = f"ReID init: {type(exc).__name__}: {exc}"
            self._reid_errors += 1

        try:
            from services.ml_service.core_v1.face_service_safe import SafeFaceGallery
            face_gallery = SafeFaceGallery(ROOT, face_cfg)
        except Exception as exc:
            self._identity_error = f"Face gallery init: {type(exc).__name__}: {exc}"

        face_has_people = bool(face_gallery and face_gallery.list_people())
        if face_has_people:
            try:
                from insightface.app import FaceAnalysis
                model_root = str(ROOT / str(face_cfg.get("model_root", "models/insightface")))
                face_app = FaceAnalysis(
                    name=str(face_cfg.get("model_pack", "buffalo_m")),
                    root=model_root,
                    allowed_modules=["detection", "recognition"],
                    providers=["CPUExecutionProvider"],
                )
                det_size = int(face_cfg.get("det_size", 320))
                face_app.prepare(
                    ctx_id=-1,
                    det_thresh=float(face_cfg.get("det_thresh", 0.55)),
                    det_size=(det_size, det_size),
                )
                print(
                    f"IDENTITY face=CPU gallery={len(face_gallery.list_people())} "
                    f"reid={'ready-lazy' if reid is not None else 'off'}",
                    flush=True,
                )
            except Exception as exc:
                face_app = None
                self._identity_error = f"Face engine: {type(exc).__name__}: {exc}"
                print(
                    f"IDENTITY face disabled but tracking continues: {self._identity_error}",
                    flush=True,
                )
        else:
            print(
                f"IDENTITY reid={'ready-lazy' if reid is not None else 'off'} "
                "face=idle(no enrolled gallery)",
                flush=True,
            )

        while not self.stop_event.is_set():
            try:
                first = self.identity_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            tasks = [first]
            while len(tasks) < 2:
                try:
                    tasks.append(self.identity_queue.get_nowait())
                except queue.Empty:
                    break

            if reid is not None:
                try:
                    crops = [task["crop"] for task in tasks]
                    features = reid.embedder.embed_batch(crops)
                    now = time.monotonic()
                    for task, feature in zip(tasks, features):
                        key = (task["cid"], int(task["track_id"]))
                        history = self._reid_embeddings[key]
                        history.append(np.asarray(feature, dtype=np.float32))
                        self._reid_samples += 1
                        if len(history) >= 3:
                            prototype = np.mean(np.stack(list(history), axis=0), axis=0)
                            gid = reid.resolve_tracklet(
                                key[0],
                                key[1],
                                prototype,
                                quality=float(task.get("det_conf") or 1.0),
                                now=now,
                            )
                            with self.identity_lock:
                                self._global_ids[key] = gid
                                local_name = self._known_names.get(key)
                                if local_name:
                                    self._global_names[gid] = local_name
                except Exception as exc:
                    self._reid_errors += 1
                    self._identity_error = f"ReID: {type(exc).__name__}: {exc}"

            if face_app is not None and face_gallery is not None:
                # Face is intentionally at most one crop per worker cycle.
                task = tasks[0]
                key = (task["cid"], int(task["track_id"]))
                now = time.monotonic()
                if now - self._last_face_attempt.get(key, 0.0) >= self.face_sample_interval:
                    self._last_face_attempt[key] = now
                    self._face_attempts += 1
                    try:
                        faces = face_app.get(task["crop"])
                        if faces:
                            face = max(
                                faces,
                                key=lambda value: max(
                                    0.0,
                                    float(value.bbox[2] - value.bbox[0])
                                    * float(value.bbox[3] - value.bbox[1]),
                                ),
                            )
                            embedding = getattr(face, "normed_embedding", None)
                            match = face_gallery.match(embedding)
                            if match is not None:
                                pending = self._face_pending.get(key)
                                if pending and pending[0] == match.name and now - pending[2] <= 5.0:
                                    hits = pending[1] + 1
                                else:
                                    hits = 1
                                self._face_pending[key] = (match.name, hits, now)
                                if hits >= 2:
                                    with self.identity_lock:
                                        self._known_names[key] = match.name
                                        gid = self._global_ids.get(key)
                                        if gid:
                                            self._global_names[gid] = match.name
                                    self._face_matches += 1
                                    face_gallery.note_recognition(match.person_id)
                    except Exception as exc:
                        self._identity_error = f"Face: {type(exc).__name__}: {exc}"

    def _adapt_detector_rate(self, min_camera_fps: float) -> None:
        # Keep the base batch-rate controller, then adapt GPU duty as well.
        super()._adapt_detector_rate(min_camera_fps)
        with self.det_lock:
            if not self.detector_ready:
                return
            if min_camera_fps < 19.3:
                self.max_gpu_duty = max(0.16, self.max_gpu_duty * 0.82)
            elif min_camera_fps >= 19.8:
                self.max_gpu_duty = min(0.25, self.max_gpu_duty + 0.01)

    def _print_stats(self) -> bool:
        result = super()._print_stats()
        with self.identity_lock:
            gids = len(set(self._global_ids.values()))
            names = len(set(self._known_names.values()))
        tracked = " ".join(
            f"{cid}:{self._tracked_last.get(cid, 0)}" for cid in self.camera_ids
        )
        heat_total = sum(int(grid.sum()) for grid in self._heatmap.values())
        print(
            "FULL_PIPELINE "
            f"tracker={int(self.tracker_enabled)} osd={int(self.osd_enabled)} "
            f"tracks=[{tracked}] meta={self._meta_injections} "
            f"id_queue={self.identity_queue.qsize()} id_drops={self._identity_drops} "
            f"reid_samples={self._reid_samples} reid_errors={self._reid_errors} "
            f"global_ids={gids} face_attempts={self._face_attempts} "
            f"known_names={names} heat_samples={heat_total}"
            + (f" identity_error={self._identity_error}" if self._identity_error else ""),
            flush=True,
        )
        return result

    def run(self) -> int:
        try:
            return super().run()
        finally:
            self.stop_event.set()
            if self.identity_thread is not None:
                self.identity_thread.join(timeout=2.0)


def run() -> int:
    return NativeCameraYolo26mFull().run()


if __name__ == "__main__":
    raise SystemExit(run())
