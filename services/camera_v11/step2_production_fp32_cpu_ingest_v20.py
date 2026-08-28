from __future__ import annotations

import signal

from .step2_production_fp32_v18 import V11Step2ProductionFP32V18


class V11Step2ProductionFP32CPUIngestV20(V11Step2ProductionFP32V18):
    """Diagnostic A/B: CPU decode/convert for detector substreams only."""

    def __init__(self) -> None:
        self.cpu_decoders: set[int] = set()
        super().__init__()
        print(
            "CAMERA_V11_STEP2_V20_INGEST decoder=avdec_h264 convert=videoconvertscale "
            "display_topology_changed=0 exact_fp32=1",
            flush=True,
        )

    def _make(self, factory: str, name: str):
        if factory == "nvurisrcbin":
            factory = "uridecodebin"
        elif factory == "nvvideoconvert":
            factory = "videoconvertscale"
        return super()._make(factory, name)

    def _configure_deep_element(self, bin_, sub_bin, element, camera) -> None:
        super()._configure_deep_element(bin_, sub_bin, element, camera)
        factory = element.get_factory()
        factory_name = factory.get_name() if factory is not None else ""
        if factory_name == "nvv4l2decoder":
            raise RuntimeError(f"{camera.camera_id}: CPU-ingest A/B selected NVIDIA decoder")
        if factory_name != "avdec_h264":
            return
        identity = id(element)
        if identity in self.cpu_decoders:
            return
        self.cpu_decoders.add(identity)
        self._set_if(element, "max-threads", 1)
        sink_pad = element.get_static_pad("sink")
        src_pad = element.get_static_pad("src")
        if sink_pad is not None:
            sink_pad.add_probe(self.Gst.PadProbeType.BUFFER, self._rtsp_frame_probe, camera.camera_id)
        if src_pad is not None:
            src_pad.add_probe(self.Gst.PadProbeType.BUFFER, self._decoded_probe, camera.camera_id)
        print(
            f"CAMERA_V11_STEP2_V20_CPU_DECODER camera={camera.camera_id} element={element.get_name()}",
            flush=True,
        )


def main() -> int:
    service = V11Step2ProductionFP32CPUIngestV20()

    def stop(_signum, _frame) -> None:
        service.stop_requested = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        return service.run()
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
