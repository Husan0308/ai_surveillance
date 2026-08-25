from __future__ import annotations

import os

from .detection import INFER_HEIGHT, INFER_WIDTH
from .person_tracking_trt86_fresh import CameraPersonTrackingTRT86Fresh


class CameraPersonTrackingTRT86Audited(CameraPersonTrackingTRT86Fresh):
    """Audited CAM-01 TRT86 + NvDCF runtime.

    This keeps the proven JIT capture/sparse detector design, then hardens three
    end-to-end contracts:
      1. nvurisrcbin and its child rtspsrc agree on the requested RTSP transport;
      2. detector preprocessing preserves the 16:9 camera aspect ratio via a
         centered letterbox instead of stretching 16:9 directly to 672x384;
      3. the static DeepStream graph is verified after construction so a future
         refactor cannot silently move NvDCF behind the tiler or bypass OSD.
    """

    def __init__(self) -> None:
        self._detector_letterbox: tuple[int, int, int, int] | None = None
        super().__init__()
        self._audit_pipeline_graph()

    def _add_camera(self, index, camera) -> None:
        super()._add_camera(index, camera)
        cid = camera.camera_id

        # CameraDetectionV2 overrides SecureCameraWallV2._add_camera(), so make
        # the outer nvurisrcbin transport explicit here as well. The inner
        # rtspsrc is already configured by SecureCameraWallV2._configure_rtsp_child.
        source = self.pipeline.get_by_name(f"camera_v2_source_{index}")
        if source is None:
            raise RuntimeError(f"{cid}: nvurisrcbin missing after graph build")
        transport = self._transport()
        self._set_if(source, "select-rtp-protocol", 4 if transport == "tcp" else 0)
        self._set_if(source, "async-handling", True)

        # Fixed-shape TensorRT input must receive the same geometry policy as
        # Ultralytics: scale to fit, preserve aspect ratio, center-pad. The current
        # camera wall works at 16:9, therefore 1280x720 -> 672x378 + 3 px top/bottom.
        converter = self.pipeline.get_by_name(f"detect_convert_{index}")
        if converter is None:
            raise RuntimeError(f"{cid}: detector nvvideoconvert missing")

        scale = min(
            float(INFER_WIDTH) / float(self.frame_width),
            float(INFER_HEIGHT) / float(self.frame_height),
        )
        content_w = max(2, min(INFER_WIDTH, int(round(self.frame_width * scale))))
        content_h = max(2, min(INFER_HEIGHT, int(round(self.frame_height * scale))))
        pad_x = max(0, (INFER_WIDTH - content_w) // 2)
        pad_y = max(0, (INFER_HEIGHT - content_h) // 2)
        converter.set_property(
            "dest-crop",
            f"{pad_x}:{pad_y}:{content_w}:{content_h}",
        )
        self._detector_letterbox = (pad_x, pad_y, content_w, content_h)

        if cid == "CAM-01":
            print(
                "CAM01_TRT86_SOURCE_HARDENED "
                f"outer_transport={transport} async_handling=1",
                flush=True,
            )
            print(
                "CAM01_TRT86_LETTERBOX "
                f"tensor={INFER_WIDTH}x{INFER_HEIGHT} "
                f"content={content_w}x{content_h} pad={pad_x},{pad_y} "
                "padding_value=114",
                flush=True,
            )

    def _scaled_detections(self, rows):
        """Undo detector letterbox and map boxes to nvstreammux frame geometry."""
        mapping = self._detector_letterbox
        if mapping is None:
            return super()._scaled_detections(rows)

        pad_x, pad_y, content_w, content_h = mapping
        sx = float(self.frame_width) / float(content_w)
        sy = float(self.frame_height) / float(content_h)
        max_x = float(self.frame_width - 1)
        max_y = float(self.frame_height - 1)
        output = []

        for coords, conf in rows:
            x1, y1, x2, y2 = [float(v) for v in coords]
            x1 = (x1 - pad_x) * sx
            x2 = (x2 - pad_x) * sx
            y1 = (y1 - pad_y) * sy
            y2 = (y2 - pad_y) * sy
            x1 = max(0.0, min(max_x, x1))
            x2 = max(0.0, min(max_x, x2))
            y1 = max(0.0, min(max_y, y1))
            y2 = max(0.0, min(max_y, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            output.append(((x1, y1, x2, y2), float(conf)))
        return output

    @staticmethod
    def _peer_element_name(element, pad_name: str) -> str | None:
        pad = element.get_static_pad(pad_name)
        if pad is None:
            return None
        peer = pad.get_peer()
        if peer is None:
            return None
        parent = peer.get_parent_element()
        return parent.get_name() if parent is not None else None

    def _require_peer(self, element, pad_name: str, expected: str, label: str) -> None:
        actual = self._peer_element_name(element, pad_name)
        if actual != expected:
            raise RuntimeError(
                f"PIPELINE_AUDIT {label}: expected peer={expected}, got={actual}"
            )

    def _audit_pipeline_graph(self) -> None:
        """Verify the complete static DeepStream topology before PLAYING."""
        tracker = self.pipeline.get_by_name("person_nvdcf_tracker")
        wall_convert = self.pipeline.get_by_name("track_wall_convert")
        wall_caps = self.pipeline.get_by_name("track_wall_caps")
        osd = self.pipeline.get_by_name("track_osd")
        if tracker is None or wall_convert is None or wall_caps is None or osd is None:
            raise RuntimeError("PIPELINE_AUDIT tracking/display elements missing")

        # The critical order: detector metadata is injected on mux.src, then NvDCF
        # consumes it, and only afterwards may tiler change geometry.
        self._require_peer(self.mux, "src", "person_nvdcf_tracker", "mux->tracker")
        self._require_peer(tracker, "src", self.tiler.get_name(), "tracker->tiler")
        self._require_peer(self.tiler, "src", self.wall_caps.get_name(), "tiler->wallcaps")
        self._require_peer(self.wall_caps, "src", self.wall_queue.get_name(), "wallcaps->queue")
        self._require_peer(self.wall_queue, "src", wall_convert.get_name(), "queue->convert")
        self._require_peer(wall_convert, "src", wall_caps.get_name(), "convert->rgba")
        self._require_peer(wall_caps, "src", osd.get_name(), "rgba->osd")
        self._require_peer(osd, "src", self.sink.get_name(), "osd->sink")

        camera_rows = []
        for index, camera in enumerate(self.cameras):
            cid = camera.camera_id
            tee = self.pipeline.get_by_name(f"detect_tee_{index}")
            display_q = self.pipeline.get_by_name(f"camera_v2_queue_{index}")
            infer_q = self.pipeline.get_by_name(f"detect_queue_{index}")
            convert = self.pipeline.get_by_name(f"detect_convert_{index}")
            caps = self.pipeline.get_by_name(f"detect_caps_{index}")
            sink = self.pipeline.get_by_name(f"detect_sink_{index}")
            source = self.pipeline.get_by_name(f"camera_v2_source_{index}")
            if any(v is None for v in (tee, display_q, infer_q, convert, caps, sink, source)):
                raise RuntimeError(f"PIPELINE_AUDIT {cid}: source/tee/infer elements missing")

            # The tee sink must have the dynamic decoded nvurisrcbin pad attached.
            tee_sink = tee.get_static_pad("sink")
            if tee_sink is None or tee_sink.get_peer() is None:
                # Dynamic source pads are linked only after negotiation; this is
                # expected before PLAYING, so record it rather than failing here.
                source_state = "dynamic-pending"
            else:
                source_state = "linked"

            self._require_peer(infer_q, "src", convert.get_name(), f"{cid} inferq->convert")
            self._require_peer(convert, "src", caps.get_name(), f"{cid} convert->caps")
            self._require_peer(caps, "src", sink.get_name(), f"{cid} caps->appsink")

            camera_rows.append(f"{cid}:{source_state}")

        batch_size = int(self.mux.get_property("batch-size"))
        if batch_size != len(self.cameras):
            raise RuntimeError(
                f"PIPELINE_AUDIT mux batch-size={batch_size} cameras={len(self.cameras)}"
            )
        if not bool(self.mux.get_property("live-source")):
            raise RuntimeError("PIPELINE_AUDIT nvstreammux live-source must be true")

        if self.tracker_width % 32 or self.tracker_height % 32:
            raise RuntimeError(
                "PIPELINE_AUDIT NvDCF dimensions must be multiples of 32"
            )

        print(
            "CAMERA_PIPELINE_AUDIT status=OK "
            f"sources={len(self.cameras)} mux_batch={batch_size} "
            f"mux={self.frame_width}x{self.frame_height} "
            f"tracker={self.tracker_width}x{self.tracker_height} "
            f"tiler={self.wall_width}x{self.wall_height} "
            "order=source/tee->mux->detector_meta->nvdcf->tiler->rgba->osd->egl "
            f"dynamic_sources=[{' '.join(camera_rows)}]",
            flush=True,
        )


def main() -> int:
    return CameraPersonTrackingTRT86Audited().run()


if __name__ == "__main__":
    raise SystemExit(main())
