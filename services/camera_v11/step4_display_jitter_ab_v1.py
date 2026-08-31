from __future__ import annotations

import os

from services.ml_service.app.config import CameraConfig
from .step1_cam02_lowlat_v7 import V11Step1Cam02LowLatV7


class V11Step4DisplayJitterABV1(V11Step1Cam02LowLatV7):
    """Step4 display-only transport A/B without modifying the frozen Step1 files.

    The frozen V7 display policy is preserved except for an explicit per-camera
    RTSP transport experiment. By default CAM-01 and CAM-03 use UDP-only at the
    child rtspsrc while CAM-02/CAM-04/CAM-05/CAM-06 remain TCP-only. CAM-02 keeps
    the previously validated nvv4l2decoder low-latency-mode setting.

    This exists to test the measured symptom where RTP PTS stays at 50 ms while
    wall-clock arrival/render gaps reach about 100 ms. No detector, tracker,
    ReID, sink, queue, resize, or latency setting is changed here.
    """

    def __init__(self) -> None:
        raw = os.environ.get("V11_STEP4_UDP_CAMERAS", "CAM-01,CAM-03")
        self.udp_cameras = {item.strip() for item in raw.split(",") if item.strip()}
        super().__init__()

        known = {camera.camera_id for camera in self.cameras}
        unknown = sorted(self.udp_cameras.difference(known))
        if unknown:
            raise RuntimeError(
                "V11 Step4 display jitter A/B unknown UDP camera ids: " + ",".join(unknown)
            )

        matrix = ",".join(
            f"{camera.camera_id}:{self._transport_for(camera.camera_id)}"
            for camera in self.cameras
        )
        print(
            "CAMERA_V11_STEP4_DISPLAY_JITTER_AB "
            f"transports={matrix} latency_ms={self.latency_ms} "
            f"drop_on_latency={int(self.drop_on_latency)} "
            "queue=latest1 sink_sync=0 sink_qos=0 frozen_step1_mutation=0",
            flush=True,
        )

    def _transport_for(self, camera_id: str) -> str:
        return "udp" if camera_id in self.udp_cameras else "tcp"

    def _configure_rtsp_child(self, bin_obj, sub_bin, element, camera: CameraConfig) -> None:
        # Preserve V7 decoder low-latency handling and all frozen V4 RTSP policy,
        # then override only the transport flag for this A/B experiment.
        super()._configure_rtsp_child(bin_obj, sub_bin, element, camera)

        factory = element.get_factory()
        factory_name = factory.get_name() if factory is not None else ""
        if factory_name != "rtspsrc":
            return

        transport = self._transport_for(camera.camera_id)
        # GstRTSPLowerTrans flags: UDP unicast=1, TCP=4.
        self._set_if(element, "protocols", 1 if transport == "udp" else 4)
        self._set_if(element, "latency", self.latency_ms)
        self._set_if(element, "drop-on-latency", self.drop_on_latency)
        self._set_if(element, "udp-buffer-size", self.udp_buffer_size)
        print(
            "CAMERA_V11_STEP4_DISPLAY_RTSP "
            f"camera={camera.camera_id} transport={transport} "
            f"latency_ms={self.latency_ms} drop_on_latency={int(self.drop_on_latency)}",
            flush=True,
        )

    def _build_camera(self, index: int, camera: CameraConfig) -> None:
        # Let frozen V7 create the exact known display path first. Pipelines are
        # not PLAYING yet, so selecting nvurisrcbin RTP policy here is safe.
        super()._build_camera(index, camera)
        transport = self._transport_for(camera.camera_id)
        source = self.sources[camera.camera_id]
        # nvurisrcbin exposes TCP-only=4. For UDP-only, allow multi at the bin and
        # force UDP-only on its rtspsrc child above.
        self._set_if(source, "select-rtp-protocol", 4 if transport == "tcp" else 0)
        print(
            "CAMERA_V11_STEP4_DISPLAY_SOURCE "
            f"camera={camera.camera_id} transport={transport} "
            f"nvurisrcbin_protocol={4 if transport == 'tcp' else 0}",
            flush=True,
        )


def main() -> int:
    return V11Step4DisplayJitterABV1().run()


if __name__ == "__main__":
    raise SystemExit(main())
