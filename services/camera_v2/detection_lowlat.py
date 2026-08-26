from __future__ import annotations

import os
import queue as pyqueue
import time

from .detection_only_pose_v4 import DetectionOnlyLowLatencyV4


class DetectionLowLatency(DetectionOnlyLowLatencyV4):
    """Canonical low-latency runtime with async detector + presentation-only bbox OSD.

    Detection remains completely outside the display branch. The only display-side
    addition is a lightweight post-tiler RGBA + nvdsosd stage. Cached detections are
    mapped into wall coordinates and attached as NvDsDisplayMeta, so drawing them
    cannot alter detector/tracker identity state. NvDCF remains deliberately off.
    """

    def _install_osd_and_meta(self) -> None:
        self.meta_boxes = 0
        self._overlay_frames = 0
        self._overlay_last_boxes = 0
        self._overlay_first_logged = False
        self._bbox_overlay_enabled = os.environ.get(
            "CAMERA_V2_DISPLAY_BBOX", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._bbox_overlay_ttl = max(
            0.25,
            min(6.0, float(os.environ.get("CAMERA_V2_DISPLAY_BBOX_TTL_SEC", "3.2"))),
        )

        if not self._bbox_overlay_enabled:
            self.osd = None
            self.wall_queue.get_static_pad("src").add_probe(
                self.Gst.PadProbeType.BUFFER,
                self._wall_probe,
            )
            return

        if self.Gst.ElementFactory.find("nvdsosd") is None:
            raise RuntimeError("DeepStream nvdsosd plugin is missing")

        # Base clean wall is wall_queue -> EGL. Detach only that final presentation
        # link; detector/pose paths stay on their independent tee branches.
        self.wall_queue.unlink(self.sink)
        convert = self._make("nvvideoconvert", "lowlat_wall_convert")
        caps = self._make("capsfilter", "lowlat_wall_rgba")
        osd = self._make("nvdsosd", "lowlat_wall_osd")

        self._set_if(convert, "gpu-id", self.gpu_id)
        self._set_if(convert, "compute-hw", 1)
        self._set_if(convert, "output-buffers", 2)
        caps.set_property(
            "caps",
            self.Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"),
        )
        # GPU bbox-only OSD. Text/masks/clocks stay off to minimize latency.
        self._set_if(osd, "process-mode", 1)
        self._set_if(osd, "display-bbox", True)
        self._set_if(osd, "display-text", False)
        self._set_if(osd, "display-mask", False)
        self._set_if(osd, "display-clock", False)
        self._set_if(osd, "gpu-id", self.gpu_id)

        for element in (convert, caps, osd):
            self.pipeline.add(element)
        self._require_link(self.wall_queue, convert, "wall queue -> bbox convert")
        self._require_link(convert, caps, "bbox convert -> RGBA")
        self._require_link(caps, osd, "RGBA -> bbox OSD")
        self._require_link(osd, self.sink, "bbox OSD -> EGL")

        # The native helper adds NvDsDisplayMeta on the already-composited wall.
        # It is presentation-only and therefore cannot seed duplicate detector or
        # tracker objects. Keep wall latency measurement after OSD so wall_p95
        # includes the actual cost of drawing the boxes.
        self.wall_queue.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._wall_bbox_probe,
        )
        osd.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._wall_probe,
        )
        self.overlay_convert = convert
        self.overlay_caps = caps
        self.osd = osd

        print(
            "CAMERA_DISPLAY_BBOX_SETUP enabled=1 mode=wall-display-meta "
            f"osd=gpu/bbox-only ttl={self._bbox_overlay_ttl:.1f}s nvdcf=0",
            flush=True,
        )

    def _wall_bbox_rows(self, now: float):
        # Tiler preserves mux pad order, so camera index maps deterministically to
        # row/column. Detection boxes are stored in mux coordinates (frame_width x
        # frame_height); map them once into the final wall tile coordinates.
        tile_w = float(self.wall_width) / float(max(1, self.tiler_columns))
        tile_h = float(self.wall_height) / float(max(1, self.tiler_rows))
        sx = tile_w / float(max(1, self.frame_width))
        sy = tile_h / float(max(1, self.frame_height))

        with self.det_lock:
            cached = dict(self._latest_detections)

        output = []
        for index, camera in enumerate(self.cameras):
            cid = camera.camera_id
            entry = cached.get(cid)
            if entry is None:
                continue
            updated, rows = entry
            if now - float(updated) > self._bbox_overlay_ttl:
                continue

            col = index % max(1, self.tiler_columns)
            row = index // max(1, self.tiler_columns)
            ox = float(col) * tile_w
            oy = float(row) * tile_h
            for coords, conf in rows:
                x1, y1, x2, y2 = [float(v) for v in coords]
                wx1 = ox + x1 * sx
                wy1 = oy + y1 * sy
                wx2 = ox + x2 * sx
                wy2 = oy + y2 * sy
                wx1 = max(ox, min(ox + tile_w - 1.0, wx1))
                wy1 = max(oy, min(oy + tile_h - 1.0, wy1))
                wx2 = max(wx1 + 1.0, min(ox + tile_w, wx2))
                wy2 = max(wy1 + 1.0, min(oy + tile_h, wy2))
                if wx2 > wx1 and wy2 > wy1:
                    output.append((wx1, wy1, wx2, wy2, float(conf)))
        return output

    def _wall_bbox_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK

        rows = self._wall_bbox_rows(time.monotonic())
        added = self.bridge.add_wall_rects(buffer, rows) if rows else 0
        self._overlay_frames += 1
        self._overlay_last_boxes = max(0, int(added))
        if added > 0:
            self.meta_boxes += int(added)
            if not self._overlay_first_logged:
                self._overlay_first_logged = True
                print(
                    "CAMERA_DISPLAY_BBOX first=1 "
                    f"boxes={added} wall={self.wall_width}x{self.wall_height}",
                    flush=True,
                )
        if self._overlay_frames % 200 == 0:
            print(
                "CAMERA_DISPLAY_BBOX_STATS "
                f"frames={self._overlay_frames} boxes_now={self._overlay_last_boxes} "
                f"total_drawn={self.meta_boxes}",
                flush=True,
            )
        return self.Gst.PadProbeReturn.OK

    def _audit_detection_only_graph(self) -> None:
        self._expect_peer(self.mux, "src", self.tiler.get_name(), "mux->tiler")
        self._expect_peer(self.tiler, "src", self.wall_caps.get_name(), "tiler->wall_geometry")
        self._expect_peer(self.wall_caps, "src", self.wall_queue.get_name(), "wall_geometry->queue")

        if self._bbox_overlay_enabled:
            self._expect_peer(
                self.wall_queue, "src", self.overlay_convert.get_name(), "queue->bbox-convert"
            )
            self._expect_peer(
                self.overlay_convert, "src", self.overlay_caps.get_name(), "bbox-convert->RGBA"
            )
            self._expect_peer(
                self.overlay_caps, "src", self.osd.get_name(), "RGBA->bbox-OSD"
            )
            self._expect_peer(self.osd, "src", self.sink.get_name(), "bbox-OSD->EGL")
        else:
            self._expect_peer(self.wall_queue, "src", self.sink.get_name(), "queue->EGL")

        forbidden = (
            "person_nvdcf_tracker",
            "track_osd",
            "detect_osd",
            "native_yolo26_pgie",
            "native_nvdcf_tracker",
        )
        present = [name for name in forbidden if self.pipeline.get_by_name(name) is not None]
        if present:
            raise RuntimeError(
                "CAMERA_LOWLAT_AUDIT inline analytics present: " + ",".join(present)
            )

        for index, camera in enumerate(self.cameras):
            cid = camera.camera_id
            tee = self.pipeline.get_by_name(f"detect_tee_{index}")
            display_q = self.pipeline.get_by_name(f"camera_v2_queue_{index}")
            infer_q = self.pipeline.get_by_name(f"detect_queue_{index}")
            converter = self.pipeline.get_by_name(f"detect_convert_{index}")
            infer_sink = self.pipeline.get_by_name(f"detect_sink_{index}")
            if any(v is None for v in (tee, display_q, infer_q, converter, infer_sink)):
                raise RuntimeError(f"CAMERA_LOWLAT_AUDIT {cid}: tee branch missing")
            self._expect_peer(infer_q, "src", converter.get_name(), f"{cid}:inferq->convert")
            try:
                interpolation = int(converter.get_property("interpolation-method"))
            except Exception:
                interpolation = -1
            if interpolation != 1:
                raise RuntimeError(
                    f"CAMERA_LOWLAT_AUDIT {cid}: detector interpolation={interpolation}, expected=1"
                )

        if (self.frame_width, self.frame_height) != (1280, 720):
            raise RuntimeError("CAMERA_LOWLAT_AUDIT mux geometry changed")
        if (self.wall_width, self.wall_height) != (1920, 720):
            raise RuntimeError("CAMERA_LOWLAT_AUDIT wall geometry changed")
        print(
            "CAMERA_LOWLAT_AUDIT status=OK display_ml_inline=0 nvdcf=0 "
            f"bbox_osd={int(self._bbox_overlay_enabled)}",
            flush=True,
        )

    def _pose_filter(self, cid: str, rows, frame):
        boxes = [
            (tuple(float(v) for v in coords), float(score))
            for coords, score in rows
        ]
        # pose_gate_v3 uses trusted_boxes=; older detection-only code used the
        # obsolete existing_boxes= spelling and therefore never exercised the
        # real pose path while it inherited the old scheduler.
        with self._pose_call_lock:
            return self.pose_gate.filter(
                cid,
                frame,
                boxes,
                trusted_boxes=None,
            )

    def _scheduler(self) -> None:
        """Primary S scheduler that never queues an already-stale camera frame.

        The shared GPU slot is acquired first. Only then is the per-camera gate
        opened, so the captured BGR frame is the freshest frame available just
        before TRT inference. Rescue may delay detector cadence, but can never add
        its runtime to the age of a frame that was captured before the wait.
        """
        assert self.result_q is not None and self.job_q is not None
        try:
            ready = self.result_q.get(timeout=40.0)
        except pyqueue.Empty:
            with self.det_lock:
                self.det_error = "primary TRT86 startup timeout"
            return
        if ready.get("type") != "ready":
            with self.det_lock:
                self.det_error = str(ready.get("error") or "primary TRT86 failed")
            return
        with self.det_lock:
            self.det_ready = True

        all_ids = [camera.camera_id for camera in self.cameras]
        configured = [
            x.strip()
            for x in os.environ.get("CAMERA_V2_DETECT_ACTIVE_CAMERAS", "").split(",")
            if x.strip()
        ]
        allowed = set(configured)
        ids = [cid for cid in all_ids if not configured or cid in allowed]
        if not ids:
            with self.det_lock:
                self.det_error = "primary selected no cameras"
            return

        versions = {cid: 0 for cid in ids}
        start = time.monotonic()
        period = 1.0 / max(0.01, self.current_primary_hz)
        due = {
            cid: start + (i * period / max(1, len(ids)))
            for i, cid in enumerate(ids)
        }
        print(
            "CAMERA_LOWLAT_READY "
            f"primary=YOLO26s/672x384 cameras={ids} "
            f"stagger={period / max(1, len(ids)):.3f}s rescue={self.rescue_camera} "
            "capture_after_gpu_slot=1",
            flush=True,
        )

        while not self.det_stop.is_set():
            cid = min(ids, key=lambda x: due[x])
            now = time.monotonic()
            if due[cid] > now:
                if self.det_stop.wait(min(0.20, due[cid] - now)):
                    break
                continue

            captured_t = None
            frame = None
            result = None
            try:
                # Critical freshness rule: wait for any rare M-rescue first, then
                # open the gate and consume a new frame. The wall never takes this
                # lock and remains fully independent.
                with self._gpu_infer_lock:
                    self._request_group([cid])
                    rows = self.mailbox.wait_group([cid], versions, timeout=1.0)
                    if rows is None:
                        self._clear_requests()
                        with self.det_lock:
                            self.capture_timeouts += 1
                        due[cid] = time.monotonic() + 0.25
                        continue
                    version, captured_t, frame = rows[0]
                    versions[cid] = version
                    self._clear_requests()
                    self.job_q.put(
                        {
                            "cameras": [cid],
                            "frames": [frame],
                            "captured": [captured_t],
                        },
                        timeout=0.3,
                    )
                    result = self.result_q.get(timeout=5.0)
            except pyqueue.Empty:
                self._clear_requests()
                with self.det_lock:
                    self.det_error = "primary TRT86 result timeout"
                due[cid] = time.monotonic() + 0.5
                continue
            except Exception as exc:
                self._clear_requests()
                with self.det_lock:
                    self.det_error = f"primary {type(exc).__name__}:{exc}"
                due[cid] = time.monotonic() + 0.5
                continue

            if result is None:
                due[cid] = time.monotonic() + 0.25
                continue
            if result.get("type") == "fatal":
                with self.det_lock:
                    self.det_error = str(result.get("error") or "primary fatal")
                return
            if result.get("type") != "result":
                due[cid] = time.monotonic() + 0.25
                continue

            raw_rows = result.get("boxes", {}).get(cid, [])
            batch_ms = float(result.get("batch_ms") or 0.0)
            primary = self._process_primary(
                cid,
                float(captured_t),
                frame,
                raw_rows,
                batch_ms,
            )
            self._queue_rescue(primary)

            with self._cadence_lock:
                period = 1.0 / max(0.01, self.current_primary_hz)
            due[cid] = max(
                due[cid] + period,
                time.monotonic() + period * 0.45,
            )


def main() -> int:
    return DetectionLowLatency().run()


if __name__ == "__main__":
    raise SystemExit(main())
