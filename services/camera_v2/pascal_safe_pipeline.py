from __future__ import annotations

"""Pascal-safe RF-DETR camera runtime with a display-first graph.

The production display path is intentionally simple and never waits on the
external detector:

    RTSP/NVDEC -> nvstreammux -> tee -> display queue -> display tiler -> OSD -> sink
                                \\-> analysis queue -> analysis tiler -> appsink

The old detector design used a zero-copy per-source split after nvstreammux. Its
child buffers kept the original mux batch alive until all children were returned.
With sparse gated per-camera queues this could retain several different parent
batches at once and exhaust the mux buffer pool. The hardware log stopped at
seven mux batches with an eight-buffer mux pool, exactly matching that failure
mode. The analysis branch below has no zero-copy per-source split.

It produces one temporary 2x3 analysis wall only when the detector scheduler has
armed a capture request. Each tile is exactly RF-DETR's input size, so Python only
copies the requested tile; it does not resize six camera frames or retain DeepStream
batch buffers.
"""

import os
import time
import json
import threading
from pathlib import Path

import numpy as np
import yaml

from .yolo_pose_backend import install as _install_yolo_pose_backend

_install_yolo_pose_backend()

from .detection import CameraDetectionV2, INFER_HEIGHT, INFER_WIDTH
from .secure import SecureCameraWallV2
from .reid_production import ProductionReIdIdentityEngine
from .reid_runtime import CropJob
from .reid_quality import bbox_iou
from .global_identity import EvidenceEmbedding, GlobalIdentity



ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REID_CONFIG = ROOT / "config" / "reid.yaml"


def _load_pascal_reid_config() -> dict:
    path = Path(
        os.environ.get(
            "CAMERA_V2_REID_CONFIG",
            str(DEFAULT_REID_CONFIG),
        )
    )
    raw = yaml.safe_load(path.read_text()) or {}
    return dict(raw.get("reid") or raw)


class CameraPascalSafeRuntime(CameraDetectionV2):
    """RF-DETR + bounded motion prediction, with no DeepStream tracker."""

    ANALYSIS_COLUMNS = 2
    ANALYSIS_ROWS = 3

    # Keep RF-DETR model shape at 672x384, but feed it a higher-resolution
    # source image. RF-DETR performs its own resize internally.
    ANALYSIS_TILE_WIDTH = max(
        INFER_WIDTH,
        int(os.environ.get("CAMERA_V2_ANALYSIS_TILE_WIDTH", "1280")),
    )
    ANALYSIS_TILE_HEIGHT = max(
        INFER_HEIGHT,
        int(os.environ.get("CAMERA_V2_ANALYSIS_TILE_HEIGHT", "720")),
    )

    def __init__(self) -> None:
        backend = os.environ.get("CAMERA_V2_DISPLAY_BACKEND", "egl").strip().lower()
        self.display_backend = backend if backend in {"egl", "x11"} else "egl"
        self.display_failover_requested = False
        self.display_watch_started = 0.0
        self.safe_wall_frames = 0
        self.safe_mux_batches = 0
        self.safe_sink_buffers = 0
        self.analysis_frames = 0
        self.source_track_counts: dict[int, int] = {}
        self.tracked_now = 0
        self._analysis_layout_logged = False
        self._startup_stall_reported = False
        super().__init__()
        self.source_track_counts = {
            int(source_id): 0 for source_id in self.camera_index.values()
        }
        self.tracker_backend = "motion-predictor"
        self.tracker = None

        # Async Daily Global-ID side path.
        # It does not own detection, tracking or display.
        self.daily_identity = None
        self._daily_identity_lock = threading.RLock()
        self._daily_identity_day = time.strftime("%Y-%m-%d")

        self._daily_reid_room_by_camera = {
            camera.camera_id: str(
                getattr(camera, "room", "") or ""
            )
            for camera in self.cameras
        }
        self._daily_reid_last_log = 0.0

        try:
            self.daily_identity = self._build_daily_identity()
            print(
                "CAMERA_DAILY_ID engine=ready "
                f"day={self._daily_identity_day} "
                "backend=production-trt86 "
                "gallery_ttl=90000s",
                flush=True,
            )
        except Exception as exc:
            self.daily_identity = None
            print(
                "CAMERA_DAILY_ID init_warning="
                f"{type(exc).__name__}:{exc}",
                flush=True,
            )

    def _daily_snapshot_path(self, day: str) -> Path:
        directory = ROOT / ".runtime" / "camera_v2" / "daily_identity"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{day}.json"

    def _save_daily_identity_snapshot(
        self,
        identity=None,
        day: str | None = None,
    ) -> bool:
        if identity is None:
            with self._daily_identity_lock:
                identity = self.daily_identity
                day = day or self._daily_identity_day

        if identity is None:
            return False

        day = str(day or self._daily_identity_day)
        core = identity.core
        now_mono = time.monotonic()

        payload = {
            "version": 1,
            "day": day,
            "saved_at_epoch": time.time(),
            "next_global_id": 1,
            "identities": [],
        }

        with core._lock:
            payload["next_global_id"] = int(core._next_global_id)

            for gid, row in sorted(core._globals.items()):
                # Only reliable daily identities survive process restart.
                if not bool(row.confirmed):
                    continue
                if bool(getattr(row, "suspect", False)):
                    continue
                if not row.gallery:
                    continue

                gallery = []

                for sample in row.gallery:
                    gallery.append({
                        "embedding": np.asarray(
                            sample.embedding,
                            dtype=np.float32,
                        ).tolist(),
                        "quality": float(sample.quality),
                        "age_sec": max(
                            0.0,
                            now_mono - float(sample.captured_at),
                        ),
                        "camera_id": str(sample.camera_id),
                        "room_id": str(sample.room_id),
                        "bbox": [
                            float(v) for v in sample.bbox
                        ],
                    })

                if not gallery:
                    continue

                payload["identities"].append({
                    "global_id": int(gid),
                    "created_age_sec": max(
                        0.0,
                        now_mono - float(row.created_at),
                    ),
                    "last_seen_age_sec": max(
                        0.0,
                        now_mono - float(row.last_seen),
                    ),
                    "last_camera": str(row.last_camera),
                    "last_room": str(row.last_room),
                    "gallery": gallery,
                })

        path = self._daily_snapshot_path(day)
        tmp = path.with_suffix(".json.tmp")

        tmp.write_text(
            json.dumps(
                payload,
                separators=(",", ":"),
            )
        )

        os.replace(tmp, path)
        return True

    def _restore_daily_identity_snapshot(
        self,
        identity,
        day: str,
    ) -> int:
        path = self._daily_snapshot_path(day)

        if not path.exists():
            return 0

        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            print(
                "CAMERA_DAILY_ID restore_warning="
                f"{type(exc).__name__}:{exc}",
                flush=True,
            )
            return 0

        if str(payload.get("day", "")) != str(day):
            return 0

        saved_epoch = float(
            payload.get("saved_at_epoch", time.time())
        )
        downtime = max(0.0, time.time() - saved_epoch)
        now_mono = time.monotonic()

        core = identity.core
        restored = 0
        max_gid = 0

        with core._lock:
            # New process: never restore old local track bindings.
            core._tracks.clear()
            core._globals.clear()

            if hasattr(core, "_canonical_by_gid"):
                core._canonical_by_gid.clear()

            for item in payload.get("identities", []):
                try:
                    gid = int(item["global_id"])

                    last_age = (
                        float(item.get("last_seen_age_sec", 0.0))
                        + downtime
                    )

                    if last_age > float(core.gallery_ttl):
                        continue

                    gallery = []

                    for index, sample in enumerate(
                        item.get("gallery", [])
                    ):
                        try:
                            embedding = np.asarray(
                                sample["embedding"],
                                dtype=np.float32,
                            )

                            if (
                                embedding.ndim != 1
                                or embedding.size == 0
                                or not np.all(np.isfinite(embedding))
                            ):
                                continue

                            sample_age = (
                                float(sample.get("age_sec", 0.0))
                                + downtime
                            )

                            if sample_age > float(core.gallery_ttl):
                                continue

                            bbox_raw = sample.get(
                                "bbox",
                                [0.0, 0.0, 1.0, 1.0],
                            )

                            bbox = tuple(
                                float(v) for v in bbox_raw[:4]
                            )

                            if len(bbox) != 4:
                                continue

                            # Negative synthetic local IDs prevent collision
                            # with new process-local tracker IDs (1,2,3...).
                            synthetic_local_id = -(
                                gid * 1000 + index + 1
                            )

                            gallery.append(
                                EvidenceEmbedding(
                                    embedding=embedding,
                                    quality=float(
                                        sample.get("quality", 0.5)
                                    ),
                                    captured_at=(
                                        now_mono - sample_age
                                    ),
                                    camera_id=str(
                                        sample.get(
                                            "camera_id",
                                            "persisted",
                                        )
                                    ),
                                    local_id=synthetic_local_id,
                                    room_id=str(
                                        sample.get("room_id", "")
                                    ),
                                    bbox=bbox,
                                    jpeg=None,
                                )
                            )
                        except Exception:
                            continue

                    if not gallery:
                        continue

                    created_age = (
                        float(
                            item.get(
                                "created_age_sec",
                                last_age,
                            )
                        )
                        + downtime
                    )

                    restored_identity = GlobalIdentity(
                        global_id=gid,
                        created_at=now_mono - created_age,
                        last_seen=now_mono - last_age,
                        last_camera=str(
                            item.get("last_camera", "")
                        ),
                        last_room=str(
                            item.get("last_room", "")
                        ),
                        confirmed=True,
                        gallery=gallery,
                        quarantine=[],
                        prototype=None,
                        active_tracks=set(),
                        suspect=False,
                    )

                    core._rebuild_identity_gallery(
                        restored_identity
                    )

                    core._globals[gid] = restored_identity
                    max_gid = max(max_gid, gid)
                    restored += 1

                except Exception:
                    continue

            saved_next = int(
                payload.get(
                    "next_global_id",
                    max_gid + 1,
                )
            )

            core._next_global_id = max(
                1,
                max_gid + 1,
                saved_next,
            )

        return restored

    def _build_daily_identity(self, day: str | None = None):
        day = str(day or time.strftime("%Y-%m-%d"))

        cfg = _load_pascal_reid_config()

        # Calendar-day rollover owns reset. TTL only prevents
        # an identity disappearing during a long working day.
        cfg["gallery_ttl_sec"] = max(
            90000.0,
            float(cfg.get("gallery_ttl_sec", 21600.0)),
        )

        identity = ProductionReIdIdentityEngine(
            self._daily_reid_room_by_camera,
            cfg,
            root=ROOT,
        )

        restored = self._restore_daily_identity_snapshot(
            identity,
            day,
        )

        if restored:
            print(
                "CAMERA_DAILY_ID restored="
                f"{restored} day={day} "
                f"next_global_id={identity.core._next_global_id}",
                flush=True,
            )

        return identity

    def _daily_identity_persistence_watchdog(self) -> bool:
        if getattr(self, "_stopping", False):
            return False

        now = time.monotonic()
        last = getattr(
            self,
            "_daily_identity_last_save",
            0.0,
        )

        if now - last < 10.0:
            return True

        self._daily_identity_last_save = now

        try:
            self._save_daily_identity_snapshot()
        except Exception as exc:
            print(
                "CAMERA_DAILY_ID save_warning="
                f"{type(exc).__name__}:{exc}",
                flush=True,
            )

        return True

    @staticmethod
    def _stop_old_daily_identity(identity) -> None:
        try:
            identity.stop()
        except Exception as exc:
            print(
                "CAMERA_DAILY_ID old_stop_warning="
                f"{type(exc).__name__}:{exc}",
                flush=True,
            )

    def _daily_identity_watchdog(self) -> bool:
        if getattr(self, "_stopping", False):
            return False

        today = time.strftime("%Y-%m-%d")

        if today == self._daily_identity_day:
            return True

        try:
            new_identity = self._build_daily_identity(today)
            new_identity.start()
        except Exception as exc:
            print(
                "CAMERA_DAILY_ID rollover_warning="
                f"{type(exc).__name__}:{exc}",
                flush=True,
            )
            return True

        with self._daily_identity_lock:
            old_identity = self.daily_identity
            old_day = self._daily_identity_day

            # Swap first. Any old pending jobs can now only modify the
            # detached old engine/core, never the new day's gallery.
            self.daily_identity = new_identity
            self._daily_identity_day = today
            self._daily_reid_last_log = 0.0

        print(
            "CAMERA_DAILY_ID RESET "
            f"old_day={old_day} "
            f"new_day={today} "
            "next_global_id=1",
            flush=True,
        )

        if old_identity is not None:
            try:
                self._save_daily_identity_snapshot(
                    old_identity,
                    old_day,
                )
            except Exception as exc:
                print(
                    "CAMERA_DAILY_ID rollover_save_warning="
                    f"{type(exc).__name__}:{exc}",
                    flush=True,
                )

            threading.Thread(
                target=self._stop_old_daily_identity,
                args=(old_identity,),
                name="camera-v2-old-daily-reid-stop",
                daemon=True,
            ).start()

        return True

    @staticmethod
    def _daily_crop_from_box(
        frame: np.ndarray,
        box: tuple[float, float, float, float],
        native_width: int,
        native_height: int,
    ):
        h, w = frame.shape[:2]

        sx = w / float(max(1, native_width))
        sy = h / float(max(1, native_height))

        x1, y1, x2, y2 = box

        x1 *= sx
        x2 *= sx
        y1 *= sy
        y2 *= sy

        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        # Same small context used by the existing production ReID path.
        x1 -= bw * 0.035
        x2 += bw * 0.035
        y1 -= bh * 0.020
        y2 += bh * 0.015

        ix1 = max(0, int(np.floor(x1)))
        iy1 = max(0, int(np.floor(y1)))
        ix2 = min(w, int(np.ceil(x2)))
        iy2 = min(h, int(np.ceil(y2)))

        if ix2 - ix1 < 8 or iy2 - iy1 < 16:
            return None, None

        return (
            frame[iy1:iy2, ix1:ix2].copy(),
            (
                float(ix1),
                float(iy1),
                float(ix2),
                float(iy2),
            ),
        )

    def _after_motion_tracks_updated(
        self,
        cid: str,
        captured_t: float,
        frame: np.ndarray,
    ) -> None:
        # First test only CAM-01.
        if cid != "CAM-01":
            return

        identity = self.daily_identity
        if identity is None:
            return

        # Current motion tracks.
        tracked = self.boxes.render_with_ids(
            cid,
            captured_t,
        )

        # Use only RAW detections from this exact detector frame.
        # This prevents predicted/stale boxes from poisoning ReID gallery.
        with self.det_lock:
            cached = getattr(
                self,
                "_raw_detector_boxes",
                {},
            ).get(cid)

        if cached is None:
            return

        raw_t, detections = cached

        if abs(float(raw_t) - float(captured_t)) > 0.08:
            return

        # Greedy track <-> fresh detection matching.
        available = set(range(len(detections)))
        matched = []

        for (
            local_id,
            x1,
            y1,
            x2,
            y2,
            track_conf,
        ) in tracked:
            track_box = (
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            )

            best_index = None
            best_iou = 0.0

            for index in available:
                det_box, det_conf = detections[index]
                score = bbox_iou(track_box, det_box)

                if score > best_iou:
                    best_iou = score
                    best_index = index

            # Require a real detector observation, not prediction only.
            if best_index is None or best_iou < 0.20:
                continue

            det_box, det_conf = detections[best_index]
            available.remove(best_index)

            matched.append(
                (
                    int(local_id),
                    track_box,
                    float(det_conf),
                    float(track_conf),
                )
            )

        room_id = self._daily_reid_room_by_camera.get(
            cid,
            "",
        )

        observe_rows = [
            {
                "object_id": local_id,
                "left": box[0],
                "top": box[1],
                "width": box[2] - box[0],
                "height": box[3] - box[1],
                "confidence": det_conf,
                "tracker_confidence": track_conf,
            }
            for (
                local_id,
                box,
                det_conf,
                track_conf,
            ) in matched
        ]

        # Empty snapshot is intentional: it tells GlobalIdentityCore
        # that nobody is currently confirmed by the detector.
        identity.observe_tracks(
            cid,
            room_id,
            observe_rows,
            now=float(captured_t),
        )

        frame_h, frame_w = frame.shape[:2]

        prepared = []

        for (
            local_id,
            box,
            det_conf,
            track_conf,
        ) in matched:
            crop, scaled_bbox = self._daily_crop_from_box(
                frame,
                box,
                self.frame_width,
                self.frame_height,
            )

            if crop is None or scaled_bbox is None:
                continue

            prepared.append(
                (
                    local_id,
                    crop,
                    scaled_bbox,
                    det_conf,
                    track_conf,
                )
            )

        for index, (
            local_id,
            crop,
            scaled_bbox,
            det_conf,
            track_conf,
        ) in enumerate(prepared):
            max_overlap = 0.0

            for other_index, other in enumerate(prepared):
                if other_index == index:
                    continue

                max_overlap = max(
                    max_overlap,
                    bbox_iou(
                        scaled_bbox,
                        other[2],
                    ),
                )

            identity.submit_crop(
                CropJob(
                    camera_id=cid,
                    local_id=local_id,
                    room_id=room_id,
                    crop=crop,
                    source_bbox=scaled_bbox,
                    source_width=frame_w,
                    source_height=frame_h,
                    detector_confidence=det_conf,
                    tracker_confidence=track_conf,
                    max_other_iou=max_overlap,
                    captured_at=float(captured_t),
                )
            )

        # Diagnostic only. Later this Global-ID replaces the displayed local ID.
        now = time.monotonic()

        if now - self._daily_reid_last_log >= 1.0:
            self._daily_reid_last_log = now

            parts = []

            for (
                local_id,
                _box,
                _det_conf,
                _track_conf,
            ) in matched:
                binding = identity.binding_for_track(
                    cid,
                    local_id,
                )

                if binding is None:
                    parts.append(
                        f"local={local_id}->global=?"
                    )
                else:
                    parts.append(
                        f"local={local_id}"
                        f"->global={binding['global_id']}"
                        f":state={binding['state']}"
                        f":score={binding['score']:.3f}"
                    )

            print(
                "CAM01_DAILY "
                + (" ".join(parts) if parts else "none"),
                flush=True,
            )

    def _preflight(self) -> None:
        super()._preflight()
        for plugin in ("tee", "nvmultistreamtiler", "nvvideoconvert", "appsink"):
            if self.Gst.ElementFactory.find(plugin) is None:
                raise RuntimeError(f"required Pascal-safe plugin is unavailable: {plugin}")
        if self.display_backend == "x11" and self.Gst.ElementFactory.find("ximagesink") is None:
            raise RuntimeError("ximagesink is unavailable for X11 display fallback")

    def _make(self, factory: str, name: str):
        if factory == "nveglglessink" and self.display_backend == "x11":
            factory = "ximagesink"
        return super()._make(factory, name)

    def _add_camera(self, index, camera) -> None:
        """Preserve the proven source -> queue -> nvstreammux ingest path."""

        cid = camera.camera_id
        self.camera_index[cid] = int(index)
        self.capture_requested[cid] = False
        SecureCameraWallV2._add_camera(self, index, camera)

    @staticmethod
    def _queue_latest(owner, element, buffers: int = 2) -> None:
        owner._set_if(element, "max-size-buffers", max(1, int(buffers)))
        owner._set_if(element, "max-size-bytes", 0)
        owner._set_if(element, "max-size-time", 0)
        owner._set_if(element, "leaky", 2)
        owner._set_if(element, "silent", True)

    def _request_src_pad(self, element, name: str):
        request_simple = getattr(element, "request_pad_simple", None)
        pad = request_simple(name) if request_simple else None
        if pad is None:
            pad = element.get_request_pad(name)
        if pad is None:
            raise RuntimeError(f"{element.get_name()} could not allocate {name}")
        self.tee_request_pads.append((element, pad))
        return pad

    def _analysis_gate_probe(self, _pad, _info):
        """Drop analysis batches immediately unless RF-DETR requested a sample."""

        with self.capture_lock:
            requested = any(bool(v) for v in self.capture_requested.values())
        return self.Gst.PadProbeReturn.OK if requested else self.Gst.PadProbeReturn.DROP

    def _install_analysis_inference(self) -> None:
        """Attach a non-retaining detector branch after nvstreammux.

        Both branches consume the same batched mux buffer, but the detector branch
        immediately drops unrequested batches. Requested batches are fully consumed
        by a second tiler, which creates its own 2x3 output surface. No retained
        per-source child buffers exist, so the mux buffer can always return to its
        pool after the analysis wall is produced.
        """

        mux_src = self.mux.get_static_pad("src")
        display_tiler_sink = self.tiler.get_static_pad("sink")
        if mux_src is None or display_tiler_sink is None:
            raise RuntimeError("could not inspect nvstreammux -> display tiler link")
        if mux_src.is_linked():
            self.mux.unlink(self.tiler)
        if mux_src.is_linked() or display_tiler_sink.is_linked():
            raise RuntimeError("could not detach nvstreammux -> display tiler")

        tee = self._make("tee", "pascal_mux_tee")
        display_q = self._make("queue", "pascal_display_branch")
        analysis_q = self._make("queue", "pascal_analysis_branch")
        analysis_tiler = self._make("nvmultistreamtiler", "pascal_analysis_tiler")
        analysis_convert = self._make("nvvideoconvert", "pascal_analysis_convert")
        analysis_caps = self._make("capsfilter", "pascal_analysis_caps")
        analysis_sink = self._make("appsink", "pascal_analysis_sink")

        self._queue_latest(self, display_q, 2)
        self._queue_latest(self, analysis_q, 1)

        analysis_width = self.ANALYSIS_TILE_WIDTH * self.ANALYSIS_COLUMNS
        analysis_height = self.ANALYSIS_TILE_HEIGHT * self.ANALYSIS_ROWS
        self._set_if(analysis_tiler, "rows", self.ANALYSIS_ROWS)
        self._set_if(analysis_tiler, "columns", self.ANALYSIS_COLUMNS)
        self._set_if(analysis_tiler, "width", analysis_width)
        self._set_if(analysis_tiler, "height", analysis_height)
        self._set_if(analysis_tiler, "gpu-id", self.gpu_id)
        self._set_if(analysis_tiler, "nvbuf-memory-type", 2)
        self._set_if(analysis_tiler, "compute-hw", 1)
        self._set_if(analysis_tiler, "interpolation-method", 4)
        if analysis_tiler.find_property("show-source") is not None:
            analysis_tiler.set_property("show-source", -1)

        self._set_if(analysis_convert, "gpu-id", self.gpu_id)
        self._set_if(analysis_convert, "compute-hw", 1)
        analysis_caps.set_property(
            "caps",
            self.Gst.Caps.from_string(
                "video/x-raw,format=BGRx,"
                f"width={analysis_width},height={analysis_height},pixel-aspect-ratio=1/1"
            ),
        )
        analysis_sink.set_property("emit-signals", True)
        analysis_sink.set_property("sync", False)
        analysis_sink.set_property("drop", True)
        analysis_sink.set_property("max-buffers", 1)
        self._set_if(analysis_sink, "enable-last-sample", False)
        self._set_if(analysis_sink, "wait-on-eos", False)

        for element in (
            tee,
            display_q,
            analysis_q,
            analysis_tiler,
            analysis_convert,
            analysis_caps,
            analysis_sink,
        ):
            self.pipeline.add(element)

        if not self.mux.link(tee):
            raise RuntimeError("failed nvstreammux -> detector/display tee")

        tee_display = self._request_src_pad(tee, "src_%u")
        tee_analysis = self._request_src_pad(tee, "src_%u")
        if tee_display.link(display_q.get_static_pad("sink")) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError("failed mux tee -> display queue")
        if tee_analysis.link(analysis_q.get_static_pad("sink")) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError("failed mux tee -> analysis queue")
        if not display_q.link(self.tiler):
            raise RuntimeError("failed display queue -> display tiler")
        if not analysis_q.link(analysis_tiler):
            raise RuntimeError("failed analysis queue -> analysis tiler")
        if not analysis_tiler.link(analysis_convert):
            raise RuntimeError("failed analysis tiler -> analysis convert")
        if not analysis_convert.link(analysis_caps):
            raise RuntimeError("failed analysis convert -> BGRx caps")
        if not analysis_caps.link(analysis_sink):
            raise RuntimeError("failed analysis caps -> appsink")

        # BUFFER-only gate: CAPS/SEGMENT events still pass, so negotiation is
        # complete before the first detector sample is requested.
        analysis_q.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._analysis_gate_probe,
        )
        analysis_sink.connect("new-sample", self._on_analysis_sample)

        self.postmux_tee = tee
        self.postmux_display_queue = display_q
        self.analysis_queue = analysis_q
        self.analysis_tiler = analysis_tiler
        self.analysis_convert = analysis_convert
        self.analysis_caps = analysis_caps
        self.analysis_sink = analysis_sink

        print(
            "CAMERA_DETECT_PATH mode=analysis-tiler "
            "source_path=direct-to-nvstreammux demux=disabled "
            "mux_batch_retention=bounded",
            flush=True,
        )

    def _on_analysis_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return self.Gst.FlowReturn.OK

        with self.capture_lock:
            requested = [cid for cid, armed in self.capture_requested.items() if armed]
        if not requested:
            return self.Gst.FlowReturn.OK

        structure = sample.get_caps().get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        expected_width = self.ANALYSIS_TILE_WIDTH * self.ANALYSIS_COLUMNS
        expected_height = self.ANALYSIS_TILE_HEIGHT * self.ANALYSIS_ROWS
        if (width, height) != (expected_width, expected_height):
            with self.det_lock:
                self.det_error = (
                    f"analysis wall geometry {width}x{height} != "
                    f"{expected_width}x{expected_height}"
                )
            return self.Gst.FlowReturn.OK

        buffer = sample.get_buffer()
        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            return self.Gst.FlowReturn.OK

        captured = time.monotonic()
        delivered: list[str] = []
        try:
            tight_stride = width * 4
            mapped_size = int(getattr(mapped, "size", len(mapped.data)))
            if mapped_size < tight_stride * height:
                raise RuntimeError(
                    f"analysis BGRx buffer too small: {mapped_size} < {tight_stride * height}"
                )
            row_stride = (
                mapped_size // height
                if height > 0 and mapped_size % height == 0
                else tight_stride
            )
            if row_stride < tight_stride:
                raise RuntimeError(
                    f"analysis invalid BGRx stride={row_stride}, tight={tight_stride}"
                )

            raw = np.frombuffer(
                mapped.data,
                dtype=np.uint8,
                count=row_stride * height,
            )
            rows = raw.reshape((height, row_stride))
            bgrx = rows[:, :tight_stride].reshape((height, width, 4))

            for cid in requested:
                index = int(self.camera_index[cid])
                row = index // self.ANALYSIS_COLUMNS
                column = index % self.ANALYSIS_COLUMNS
                y1 = row * self.ANALYSIS_TILE_HEIGHT
                y2 = y1 + self.ANALYSIS_TILE_HEIGHT
                x1 = column * self.ANALYSIS_TILE_WIDTH
                x2 = x1 + self.ANALYSIS_TILE_WIDTH
                frame = bgrx[y1:y2, x1:x2, :3].copy()
                if frame.shape != (self.ANALYSIS_TILE_HEIGHT, self.ANALYSIS_TILE_WIDTH, 3):
                    continue
                self.mailbox.put(cid, captured, frame)
                delivered.append(cid)

            if not self._analysis_layout_logged:
                self._analysis_layout_logged = True
                print(
                    "CAMERA_INFER_LAYOUT "
                    f"wall={width}x{height} stride={row_stride} "
                    f"tile={self.ANALYSIS_TILE_WIDTH}x{self.ANALYSIS_TILE_HEIGHT} grid="
                    f"{self.ANALYSIS_COLUMNS}x{self.ANALYSIS_ROWS}",
                    flush=True,
                )
        finally:
            buffer.unmap(mapped)

        if delivered:
            with self.capture_lock:
                for cid in delivered:
                    self.capture_requested[cid] = False
            self.analysis_frames += 1
        return self.Gst.FlowReturn.OK

    def _scaled_detections(self, rows):
        sx = self.frame_width / float(self.ANALYSIS_TILE_WIDTH)
        sy = self.frame_height / float(self.ANALYSIS_TILE_HEIGHT)

        output = []
        for coords, conf in rows:
            x1, y1, x2, y2 = coords
            output.append(
                ((x1 * sx, y1 * sy, x2 * sx, y2 * sy), conf)
            )
        return output

    def _install_osd_and_meta(self) -> None:
        self._install_analysis_inference()

        queue_src = self.wall_queue.get_static_pad("src")
        sink_pad = self.sink.get_static_pad("sink")
        if queue_src is None or sink_pad is None:
            raise RuntimeError("could not inspect baseline wall -> display pads")
        if queue_src.is_linked():
            self.wall_queue.unlink(self.sink)
        if queue_src.is_linked() or sink_pad.is_linked():
            raise RuntimeError("could not detach baseline wall -> display link")

        convert = self._make("nvvideoconvert", "pascal_wall_convert")
        caps = self._make("capsfilter", "pascal_wall_caps")
        osd = self._make("nvdsosd", "pascal_osd")
        self._set_if(convert, "gpu-id", self.gpu_id)
        self._set_if(convert, "compute-hw", 1)
        caps.set_property(
            "caps",
            self.Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"),
        )
        self._set_if(osd, "process-mode", 1)
        self._set_if(osd, "display-bbox", True)
        self._set_if(osd, "display-text", True)
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

        if self.display_backend == "egl":
            if not osd.link(self.sink):
                raise RuntimeError("failed nvdsosd -> nveglglessink")
        else:
            download = self._make("nvvideoconvert", "pascal_x11_download")
            sys_caps = self._make("capsfilter", "pascal_x11_caps")
            self._set_if(download, "gpu-id", self.gpu_id)
            self._set_if(download, "compute-hw", 1)
            sys_caps.set_property(
                "caps",
                self.Gst.Caps.from_string("video/x-raw,format=BGRx"),
            )
            self.pipeline.add(download)
            self.pipeline.add(sys_caps)
            if not osd.link(download):
                raise RuntimeError("failed nvdsosd -> X11 download convert")
            if not download.link(sys_caps):
                raise RuntimeError("failed X11 download convert -> system caps")
            if not sys_caps.link(self.sink):
                raise RuntimeError("failed system BGRx -> ximagesink")
            self.x11_download = download
            self.x11_caps = sys_caps

        mux_src = self.mux.get_static_pad("src")
        overlay_src = caps.get_static_pad("src")
        osd_src = osd.get_static_pad("src")
        final_sink_pad = self.sink.get_static_pad("sink")
        if mux_src is None or overlay_src is None or osd_src is None or final_sink_pad is None:
            raise RuntimeError("could not obtain mux/overlay/OSD/display probe pads")
        mux_src.add_probe(self.Gst.PadProbeType.BUFFER, self._pascal_mux_probe)
        overlay_src.add_probe(self.Gst.PadProbeType.BUFFER, self._pascal_overlay_probe)
        osd_src.add_probe(self.Gst.PadProbeType.BUFFER, self._pascal_wall_probe)
        final_sink_pad.add_probe(self.Gst.PadProbeType.BUFFER, self._pascal_sink_probe)
        self.osd = osd

    def _pascal_mux_probe(self, _pad, _info):
        self.safe_mux_batches += 1
        return self.Gst.PadProbeReturn.OK

    def _pascal_overlay_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK

        now = time.monotonic()
        wall_w = float(self.wall_width)
        wall_h = float(self.wall_height)

        # nvmultistreamtiler show-source >= 0 means fullscreen/focus mode.
        show_source = -1
        try:
            show_source = int(self.tiler.get_property("show-source"))
        except Exception:
            pass

        wall_tracks = []

        for cid, source_id in self.camera_index.items():
            if cid == "CAM-01":
                tracked = self.boxes.render_with_ids(cid, now)

                # Preserve camera-local track ID for post-tiler
                # bbox + ID label rendering.
                rows = tracked

                last = getattr(self, "_cam01_id_debug_last", 0.0)
                if now - last >= 0.50:
                    self._cam01_id_debug_last = now
                    print(
                        "CAM01_IDS "
                        + " ".join(
                            f"id={tid}:"
                            f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})"
                            for tid, x1, y1, x2, y2, conf in tracked
                        ),
                        flush=True,
                    )
            else:
                rows = self.boxes.render_with_ids(cid, now)

            if not rows:
                continue

            if show_source >= 0:
                if int(source_id) != show_source:
                    continue
                tile_x = 0.0
                tile_y = 0.0
                tile_w = wall_w
                tile_h = wall_h
            else:
                column = int(source_id) % self.tiler_columns
                row = int(source_id) // self.tiler_columns
                tile_w = wall_w / float(self.tiler_columns)
                tile_h = wall_h / float(self.tiler_rows)
                tile_x = column * tile_w
                tile_y = row * tile_h

            sx = tile_w / float(self.frame_width)
            sy = tile_h / float(self.frame_height)

            for track_id, x1, y1, x2, y2, conf in rows:
                display_id = int(track_id)

                # CAM-01: once ReID has a confirmed Daily Global-ID,
                # display that ID instead of the short-lived local tracker ID.
                if cid == "CAM-01" and self.daily_identity is not None:
                    binding = self.daily_identity.binding_for_track(
                        cid,
                        int(track_id),
                    )

                    if (
                        binding is not None
                        and str(binding.get("state", "")) == "CONFIRMED"
                    ):
                        display_id = int(binding["global_id"])

                wall_tracks.append((
                    display_id,
                    tile_x + x1 * sx,
                    tile_y + y1 * sy,
                    tile_x + x2 * sx,
                    tile_y + y2 * sy,
                    conf,
                ))

        added = self.bridge.add_wall_tracks(
            buffer,
            wall_tracks,
        )

        if added > 0:
            with self.det_lock:
                self.meta_boxes += added

        return self.Gst.PadProbeReturn.OK

    def _pascal_wall_probe(self, pad, info):
        self.safe_wall_frames += 1
        return CameraDetectionV2._wall_probe(self, pad, info)

    def _pascal_sink_probe(self, _pad, _info):
        self.safe_sink_buffers += 1
        return self.Gst.PadProbeReturn.OK

    def _active_motion_counts(self) -> dict[int, int]:
        now = time.monotonic()
        output = {int(source_id): 0 for source_id in self.camera_index.values()}
        boxes = getattr(self, "boxes", None)
        if boxes is None:
            return output
        with boxes.lock:
            for cid, source_id in self.camera_index.items():
                active = sum(
                    1
                    for track in boxes.tracks.get(cid, {}).values()
                    if now - float(track.last_det_t) <= float(boxes.max_age)
                )
                output[int(source_id)] = active
        return output

    def live_source_counts(self) -> dict[int, int]:
        counts = self._active_motion_counts()
        with self.det_lock:
            self.source_track_counts = counts
            self.tracked_now = sum(counts.values())
        return dict(counts)

    def _display_watchdog(self) -> bool:
        if self.display_backend != "egl" or self._stopping:
            return False
        elapsed = time.monotonic() - self.display_watch_started
        if elapsed < float(os.environ.get("CAMERA_V2_EGL_FAILOVER_SEC", "8.0")):
            return True

        source_frames = sum(int(stat.frames) for stat in self.stats.values())
        rendered, _dropped = self._sink_stats()
        if (
            source_frames >= 60
            and self.safe_mux_batches >= 10
            and self.safe_wall_frames >= 10
            and self.safe_sink_buffers >= 10
            and rendered == 0
        ):
            self.display_failover_requested = True
            print(
                "CAMERA_DISPLAY_FAILOVER "
                f"from=egl to=x11 reason=zero-render source_frames={source_frames} "
                f"mux_batches={self.safe_mux_batches} wall_frames={self.safe_wall_frames} "
                f"sink_buffers={self.safe_sink_buffers}",
                flush=True,
            )
            self.stop()
            return False
        return True

    def _startup_watchdog(self) -> bool:
        if self._stopping:
            return False
        if time.monotonic() - self.display_watch_started < 10.0:
            return True

        source_total = sum(int(stat.frames) for stat in self.stats.values())
        if source_total == 0:
            stage = "source-or-auth"
        elif self.safe_mux_batches == 0:
            stage = "nvstreammux"
        elif self.safe_wall_frames == 0:
            stage = "display-tiler-or-osd"
        elif self.safe_sink_buffers == 0:
            stage = "display-sink-link"
        else:
            return False

        if not self._startup_stall_reported:
            self._startup_stall_reported = True
            print(
                "CAMERA_STARTUP_STALL "
                f"stage={stage} source_frames={source_total} "
                f"mux_batches={self.safe_mux_batches} wall_frames={self.safe_wall_frames} "
                f"sink_buffers={self.safe_sink_buffers}",
                flush=True,
            )
        return True

    def _print_stats(self) -> bool:
        keep = CameraDetectionV2._print_stats(self)
        counts = self.live_source_counts()
        rendered, dropped = self._sink_stats()
        source_total = sum(int(stat.frames) for stat in self.stats.values())
        print(
            "CAMERA_PASCAL_SAFE "
            f"display={self.display_backend} source_frames={source_total} "
            f"mux_batches={self.safe_mux_batches} wall_frames={self.safe_wall_frames} "
            f"sink_buffers={self.safe_sink_buffers} analysis_frames={self.analysis_frames} "
            f"tracked_now={self.tracked_now} source_counts={counts} "
            f"rendered={rendered if rendered is not None else '?'} "
            f"dropped={dropped if dropped is not None else '?'} "
            "nvtracker=0 tracker=motion-predictor detector_path=analysis-tiler",
            flush=True,
        )
        return keep

    def run(self) -> int:
        self.display_watch_started = time.monotonic()
        self.GLib.timeout_add(1000, self._startup_watchdog)

        if self.display_backend == "egl":
            self.GLib.timeout_add(
                1000,
                self._display_watchdog,
            )

        # Check local calendar day once per second.
        # Actual rollover only happens when YYYY-MM-DD changes.
        self.GLib.timeout_add(
            1000,
            self._daily_identity_watchdog,
        )

        self.GLib.timeout_add(
            1000,
            self._daily_identity_persistence_watchdog,
        )

        reid_started = False

        if self.daily_identity is not None:
            try:
                self.daily_identity.start()
                reid_started = True
                print(
                    "CAMERA_DAILY_ID started=1 "
                    "camera=CAM-01",
                    flush=True,
                )
            except Exception as exc:
                print(
                    "CAMERA_DAILY_ID start_warning="
                    f"{type(exc).__name__}:{exc}",
                    flush=True,
                )

        print(
            "CAMERA_PASCAL_SAFE ready backend=YOLO26s-pose "
            f"display={self.display_backend} "
            "tracker=motion-predictor nvtracker=disabled "
            "source_path=direct-to-mux "
            "detector_path=analysis-tiler demux=disabled",
            flush=True,
        )

        try:
            return super().run()
        finally:
            if (
                reid_started
                and self.daily_identity is not None
            ):
                try:
                    self._save_daily_identity_snapshot()
                except Exception as exc:
                    print(
                        "CAMERA_DAILY_ID final_save_warning="
                        f"{type(exc).__name__}:{exc}",
                        flush=True,
                    )

                try:
                    self.daily_identity.stop()
                except Exception as exc:
                    print(
                        "CAMERA_DAILY_ID stop_warning="
                        f"{type(exc).__name__}:{exc}",
                        flush=True,
                    )


def main() -> int:
    enabled = os.environ.get("CAMERA_V2_PASCAL_SAFE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if not enabled:
        raise RuntimeError("CAMERA_V2_PASCAL_SAFE=1 is required")
    return CameraPascalSafeRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
