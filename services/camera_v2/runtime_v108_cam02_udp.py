from __future__ import annotations

from .runtime_v105_mux_arrival_audit import PascalTrackerMuxArrivalAuditRuntime


class PascalCam02UdpRuntime(PascalTrackerMuxArrivalAuditRuntime):
    """V10.8: one-variable CAM-02 RTP transport A/B (TCP -> UDP unicast).

    V10.7 showed that increasing only CAM-02 RTSP latency from 60 ms to 120 ms
    did not materially improve the source/tracker tail. Keep the V10.5/V10.4
    tracker, mux, detector, display and bbox behavior unchanged. Only CAM-02 is
    allowed to negotiate UDP at nvurisrcbin level and then forced to UDP unicast
    on the concrete rtspsrc child.
    """

    TARGET_CAMERA = "CAM-02"
    UDP_PROTOCOL = 1

    def __init__(self) -> None:
        super().__init__()
        source = self.sources[self.TARGET_CAMERA]
        # nvurisrcbin documents 0 as the multi-transport mode that includes UDP.
        # The rtspsrc child callback below then narrows the actual lower transport
        # to UDP unicast only (GstRTSPLowerTrans.UDP == 1).
        changed = self._set_if(source, "select-rtp-protocol", 0)
        source_value = None
        if changed:
            try:
                source_value = int(source.get_property("select-rtp-protocol"))
            except Exception:
                source_value = None
        print(
            "CAMERA_V108_ARCH only_change=CAM-02-rtp-transport-tcp-to-udp "
            "rtsp_latency=60ms drop_on_latency=unchanged mux=unchanged "
            "nvdcf=unchanged bbox=unchanged",
            flush=True,
        )
        print(
            "CAMERA_V108_SOURCE "
            f"camera={self.TARGET_CAMERA} select_rtp_protocol={source_value if source_value is not None else 'unknown'} "
            f"property_set={int(changed)} udp_buffer_size={self.udp_buffer_size}",
            flush=True,
        )
        if not changed or source_value != 0:
            raise RuntimeError(
                f"V10.8 expected CAM-02 nvurisrcbin select-rtp-protocol=0, got {source_value}"
            )

    def _configure_rtsp_child(self, bin_, sub_bin, element, camera) -> None:
        super()._configure_rtsp_child(bin_, sub_bin, element, camera)
        if camera.camera_id != self.TARGET_CAMERA:
            return
        factory = element.get_factory()
        factory_name = factory.get_name() if factory is not None else ""
        if factory_name != "rtspsrc":
            return

        changed = self._set_if(element, "protocols", self.UDP_PROTOCOL)
        value = None
        if changed:
            try:
                value = int(element.get_property("protocols"))
            except Exception:
                value = None
        print(
            "CAMERA_V108_RTSP "
            f"camera={camera.camera_id} protocols={value if value is not None else 'unknown'} "
            f"expected_udp={self.UDP_PROTOCOL} latency_ms={self.rtsp_latency_ms} "
            f"drop_on_latency=1 udp_buffer_size={self.udp_buffer_size}",
            flush=True,
        )


def main() -> int:
    return PascalCam02UdpRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
