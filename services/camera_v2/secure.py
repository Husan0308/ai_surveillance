from __future__ import annotations

import sys

from .main import CameraWallV2


class SecureCameraWallV2(CameraWallV2):
    """Camera V2 with RTSP credentials applied to nvurisrcbin's child rtspsrc.

    nvurisrcbin does not expose rtspsrc's user-id/user-pw properties directly.
    We keep the configured RTSP URI clean and use GstBin's deep-element-added
    signal to configure the internal rtspsrc as soon as DeepStream creates it.
    """

    def _configure_rtsp_child(self, _bin, _sub_bin, element, camera) -> None:
        factory = element.get_factory()
        factory_name = factory.get_name() if factory is not None else ""
        if factory_name != "rtspsrc":
            return

        # rtspsrc natively handles RTSP Basic/Digest challenge-response with
        # user-id/user-pw. Secrets are never printed or embedded in a loggable URI.
        if camera.username:
            self._set_if(element, "user-id", camera.username)
            self._set_if(element, "user-pw", camera.password)

        # Keep the actual child aligned with the smooth camera profile too.
        self._set_if(element, "latency", self.rtsp_latency_ms)
        self._set_if(element, "drop-on-latency", True)
        self._set_if(element, "udp-buffer-size", self.udp_buffer_size)
        self._set_if(element, "buffer-mode", 3)  # auto

        print(
            f"CAMERA_V2 {camera.camera_id} rtspsrc configured "
            f"auth={'yes' if camera.username else 'no'} latency={self.rtsp_latency_ms}ms",
            flush=True,
        )

    def _add_camera(self, index, camera) -> None:
        cid = camera.camera_id
        source = self._make("nvurisrcbin", f"camera_v2_source_{index}")
        queue = self._make("queue", f"camera_v2_queue_{index}")

        # Connect before assigning URI / entering READY so dynamically-created
        # children cannot appear before our authentication hook is installed.
        source.connect("deep-element-added", self._configure_rtsp_child, camera)

        # Keep credentials OUT of the URI. They are injected into the child
        # rtspsrc using its official user-id/user-pw properties above.
        source.set_property("uri", camera.uri)
        self._set_if(source, "disable-audio", True)
        self._set_if(source, "select-rtp-protocol", 0)  # UDP/UDP multicast/TCP auto
        self._set_if(source, "latency", self.rtsp_latency_ms)
        self._set_if(source, "drop-on-latency", True)
        self._set_if(source, "low-latency-mode", self.low_latency_mode)
        self._set_if(source, "num-extra-surfaces", self.extra_surfaces)
        self._set_if(source, "cudadec-memtype", 0)
        self._set_if(source, "udp-buffer-size", self.udp_buffer_size)
        self._set_if(source, "rtsp-reconnect-interval", 2)
        self._set_if(source, "rtsp-reconnect-attempts", -1)
        self._set_if(source, "message-forward", True)
        self._set_if(source, "gpu-id", self.gpu_id)

        self._set_if(queue, "max-size-buffers", 1)
        self._set_if(queue, "max-size-bytes", 0)
        self._set_if(queue, "max-size-time", 0)
        self._set_if(queue, "leaky", 2)
        self._set_if(queue, "silent", True)

        self.pipeline.add(source)
        self.pipeline.add(queue)

        mux_pad = self._request_mux_pad(index)
        qsrc = queue.get_static_pad("src")
        if qsrc.link(mux_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{cid}: queue -> nvstreammux link failed")

        qsrc.add_probe(self.Gst.PadProbeType.BUFFER, self._source_probe, cid)
        source.connect("pad-added", self._source_pad_added, queue, cid)
        self.sources[cid] = source
        self.queues[cid] = queue


def main() -> int:
    wall = SecureCameraWallV2()
    missing_auth = [c.camera_id for c in wall.cameras if not c.username]
    if missing_auth:
        print(
            "CAMERA_V2 WARNING RTSP username missing for: " + ", ".join(missing_auth),
            file=sys.stderr,
            flush=True,
        )
    return wall.run()


if __name__ == "__main__":
    raise SystemExit(main())
