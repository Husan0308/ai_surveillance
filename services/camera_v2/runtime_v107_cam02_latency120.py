from __future__ import annotations

from .runtime_v105_mux_arrival_audit import PascalTrackerMuxArrivalAuditRuntime


class PascalCam02Latency120Runtime(PascalTrackerMuxArrivalAuditRuntime):
    """V10.7: one-variable CAM-02 RTSP jitterbuffer latency A/B.

    V10.6 could not prove its tcp-timestamp A/B was active on the deployed
    GStreamer stack.  Keep the V10.5/V10.4 tracker, mux, detector, display and
    bbox behavior unchanged.  Only CAM-02 gets a 120 ms nvurisrcbin/rtspsrc
    jitterbuffer instead of the 60 ms global baseline.
    """

    TARGET_CAMERA = "CAM-02"
    TARGET_LATENCY_MS = 120

    def __init__(self) -> None:
        super().__init__()
        source = self.sources[self.TARGET_CAMERA]
        changed = self._set_if(source, "latency", self.TARGET_LATENCY_MS)
        value = None
        if changed:
            try:
                value = int(source.get_property("latency"))
            except Exception:
                value = None
        print(
            "CAMERA_V107_ARCH only_change=CAM-02-rtsp-latency-60-to-120 "
            "drop_on_latency=unchanged mux=unchanged nvdcf=unchanged bbox=unchanged",
            flush=True,
        )
        print(
            "CAMERA_V107_SOURCE "
            f"camera={self.TARGET_CAMERA} source_latency_ms={value if value is not None else 'unknown'} "
            f"global_latency_ms={self.rtsp_latency_ms} property_set={int(changed)}",
            flush=True,
        )
        if value != self.TARGET_LATENCY_MS:
            raise RuntimeError(
                f"V10.7 expected CAM-02 nvurisrcbin latency={self.TARGET_LATENCY_MS}, got {value}"
            )

    def _configure_rtsp_child(self, bin_, sub_bin, element, camera) -> None:
        super()._configure_rtsp_child(bin_, sub_bin, element, camera)
        if camera.camera_id != self.TARGET_CAMERA:
            return
        factory = element.get_factory()
        factory_name = factory.get_name() if factory is not None else ""
        if factory_name != "rtspsrc":
            return
        changed = self._set_if(element, "latency", self.TARGET_LATENCY_MS)
        value = None
        if changed:
            try:
                value = int(element.get_property("latency"))
            except Exception:
                value = None
        print(
            "CAMERA_V107_RTSP "
            f"camera={camera.camera_id} latency_ms={value if value is not None else 'unknown'} "
            f"drop_on_latency=1 transport={self._transport()}",
            flush=True,
        )


def main() -> int:
    return PascalCam02Latency120Runtime().run()


if __name__ == "__main__":
    raise SystemExit(main())
