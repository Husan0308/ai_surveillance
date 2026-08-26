from __future__ import annotations

from .detection_lowlat import DetectionLowLatency


class DetectionLowLatencyNv12(DetectionLowLatency):
    """Low-latency detector with presentation-only bbox OSD directly on NV12.

    DeepStream 7.1 nvdsosd GPU mode accepts NV12 directly.  The previous overlay
    path converted the entire 1920x720 wall to RGBA on every frame before OSD,
    which needlessly competed with TensorRT on Pascal.  This variant keeps the
    working NvDsDisplayMeta overlay but removes that full-frame conversion:

        wall_queue (NV12/NVMM) -> nvdsosd GPU -> nveglglessink

    Detection, pose and CAM-05 rescue remain on the isolated ML path.  NvDCF is
    still deliberately absent at this stage.
    """

    def _install_osd_and_meta(self) -> None:
        self.meta_boxes = 0
        self._overlay_frames = 0
        self._overlay_last_boxes = 0
        self._overlay_first_logged = False
        self._bbox_overlay_enabled = self._env_flag("CAMERA_V2_DISPLAY_BBOX", True)
        try:
            ttl = float(__import__("os").environ.get("CAMERA_V2_DISPLAY_BBOX_TTL_SEC", "3.2"))
        except Exception:
            ttl = 3.2
        self._bbox_overlay_ttl = max(0.25, min(6.0, ttl))

        if not self._bbox_overlay_enabled:
            self.osd = None
            self.wall_queue.get_static_pad("src").add_probe(
                self.Gst.PadProbeType.BUFFER,
                self._wall_probe,
            )
            return

        if self.Gst.ElementFactory.find("nvdsosd") is None:
            raise RuntimeError("DeepStream nvdsosd plugin is missing")

        self.wall_queue.unlink(self.sink)
        osd = self._make("nvdsosd", "lowlat_wall_osd")
        self._set_if(osd, "process-mode", 1)  # GPU; DS 7.1 accepts NV12 directly.
        self._set_if(osd, "display-bbox", True)
        self._set_if(osd, "display-text", False)
        self._set_if(osd, "display-mask", False)
        self._set_if(osd, "display-clock", False)
        self._set_if(osd, "gpu-id", self.gpu_id)
        self.pipeline.add(osd)
        self._require_link(self.wall_queue, osd, "wall queue -> NV12 bbox OSD")
        self._require_link(osd, self.sink, "NV12 bbox OSD -> EGL")

        # Attach presentation-only NvDsDisplayMeta immediately before OSD.
        self.wall_queue.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._wall_bbox_probe,
        )
        # Measure the real visible wall after bbox drawing.
        osd.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER,
            self._wall_probe,
        )
        self.osd = osd

        print(
            "CAMERA_DISPLAY_BBOX_SETUP enabled=1 mode=wall-display-meta "
            f"path=NV12-direct osd=gpu/bbox-only ttl={self._bbox_overlay_ttl:.1f}s "
            "rgba_convert=0 nvdcf=0",
            flush=True,
        )

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        import os

        fallback = "1" if default else "0"
        return os.environ.get(name, fallback).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _audit_detection_only_graph(self) -> None:
        self._expect_peer(self.mux, "src", self.tiler.get_name(), "mux->tiler")
        self._expect_peer(
            self.tiler, "src", self.wall_caps.get_name(), "tiler->wall_geometry"
        )
        self._expect_peer(
            self.wall_caps, "src", self.wall_queue.get_name(), "wall_geometry->queue"
        )

        if self._bbox_overlay_enabled:
            self._expect_peer(
                self.wall_queue, "src", self.osd.get_name(), "queue->NV12-bbox-OSD"
            )
            self._expect_peer(
                self.osd, "src", self.sink.get_name(), "NV12-bbox-OSD->EGL"
            )
            # Full-wall RGBA conversion is forbidden in the low-latency path.
            if self.pipeline.get_by_name("lowlat_wall_convert") is not None:
                raise RuntimeError("CAMERA_LOWLAT_AUDIT obsolete wall RGBA converter present")
            if self.pipeline.get_by_name("lowlat_wall_rgba") is not None:
                raise RuntimeError("CAMERA_LOWLAT_AUDIT obsolete wall RGBA caps present")
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
            self._expect_peer(
                infer_q, "src", converter.get_name(), f"{cid}:inferq->convert"
            )
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
            f"bbox_osd={int(self._bbox_overlay_enabled)} osd_path=NV12-direct rgba_convert=0",
            flush=True,
        )


def main() -> int:
    return DetectionLowLatencyNv12().run()


if __name__ == "__main__":
    raise SystemExit(main())
