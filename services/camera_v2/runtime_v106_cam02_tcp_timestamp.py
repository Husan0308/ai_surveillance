from __future__ import annotations

from .runtime_v105_mux_arrival_audit import PascalTrackerMuxArrivalAuditRuntime


class PascalCam02TcpTimestampRuntime(PascalTrackerMuxArrivalAuditRuntime):
    """V10.6: one-variable CAM-02 RTSP burst/jitter A/B.

    V10.5 showed CAM-02 is the outlier at the tracker-mux input path.  Keep the
    entire V10.5/V10.4 pipeline unchanged and disable rtspsrc tcp-timestamp only
    for CAM-02.  GStreamer normally interpolates TCP RTP timestamps because some
    servers deliver data in bursts; tcp-timestamp can turn those receive bursts
    into timestamp bursts.  This A/B tests that single hypothesis without adding
    jitterbuffer latency or changing mux/NvDCF/bbox behavior.
    """

    TARGET_CAMERA = "CAM-02"

    def __init__(self) -> None:
        super().__init__()
        print(
            "CAMERA_V106_ARCH only_change=CAM-02-rtspsrc-tcp-timestamp-off "
            "rtsp_latency=unchanged drop_on_latency=unchanged mux=unchanged "
            "nvdcf=unchanged bbox=unchanged",
            flush=True,
        )

    def _configure_rtsp_child(self, bin_, sub_bin, element, camera) -> None:
        super()._configure_rtsp_child(bin_, sub_bin, element, camera)
        if camera.camera_id != self.TARGET_CAMERA:
            return
        factory = element.get_factory()
        factory_name = factory.get_name() if factory is not None else ""
        if factory_name != "rtspsrc":
            return
        changed = self._set_if(element, "tcp-timestamp", False)
        value = None
        if changed:
            try:
                value = int(bool(element.get_property("tcp-timestamp")))
            except Exception:
                value = None
        print(
            "CAMERA_V106_RTSP "
            f"camera={camera.camera_id} tcp_timestamp={value if value is not None else 'unknown'} "
            f"latency_ms={self.rtsp_latency_ms} drop_on_latency=1 transport={self._transport()}",
            flush=True,
        )


def main() -> int:
    return PascalCam02TcpTimestampRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
