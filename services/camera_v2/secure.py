from __future__ import annotations

import os
import sys

from .dynamic_wall import DynamicCameraWallV2


class SecureCameraWallV2(DynamicCameraWallV2):
    """Camera V2 with deterministic RTSP authentication and transport."""

    def _transport(self) -> str:
        configured = str(getattr(self.settings.deepstream, "rtsp_transport", "auto") or "auto")
        value = os.environ.get("CAMERA_V2_RTSP_TRANSPORT", configured).strip().lower()
        if value not in {"auto", "tcp", "udp"}:
            raise RuntimeError("CAMERA_V2_RTSP_TRANSPORT must be one of: auto, tcp, udp")
        return value

    @staticmethod
    def _env_bool(name: str, fallback: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return bool(fallback)
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        raise RuntimeError(f"{name} must be boolean (0/1, true/false, yes/no, on/off)")

    def _rtsp_keepalive(self, camera_id: str) -> bool:
        # Per-camera override wins over the global policy. Default remains ON,
        # matching GStreamer rtspsrc. Some RTSP/NVR implementations terminate
        # sessions when keep-alive requests are handled poorly, so this knob lets
        # us test a single problematic channel without weakening the other feeds.
        per_camera = f"{camera_id.replace('-', '_')}_RTSP_KEEPALIVE"
        if per_camera in os.environ:
            return self._env_bool(per_camera, True)
        return self._env_bool("CAMERA_V2_RTSP_KEEPALIVE", True)

    def _configure_rtsp_child(self, _bin, _sub_bin, element, camera) -> None:
        factory = element.get_factory()
        factory_name = factory.get_name() if factory is not None else ""
        if factory_name != "rtspsrc":
            return

        transport = self._transport()
        keepalive = self._rtsp_keepalive(camera.camera_id)
        if camera.username:
            self._set_if(element, "user-id", camera.username)
            self._set_if(element, "user-pw", camera.password)

        # rtspsrc protocols is GstRTSPLowerTrans flags: UDP=1, TCP=4.
        if transport == "tcp":
            self._set_if(element, "protocols", 4)
        elif transport == "udp":
            self._set_if(element, "protocols", 1)

        self._set_if(element, "latency", self.rtsp_latency_ms)
        self._set_if(element, "drop-on-latency", True)
        self._set_if(element, "udp-buffer-size", self.udp_buffer_size)
        self._set_if(element, "buffer-mode", 3)  # auto
        self._set_if(element, "do-rtsp-keep-alive", keepalive)

        print(
            f"CAMERA_V2 {camera.camera_id} rtspsrc configured "
            f"auth={'yes' if camera.username else 'no'} "
            f"transport={transport} latency={self.rtsp_latency_ms}ms "
            f"keepalive={int(keepalive)}",
            flush=True,
        )

    def _add_camera(self, index, camera) -> None:
        cid = camera.camera_id
        transport = self._transport()
        source = self._make("nvurisrcbin", f"camera_v2_source_{index}")
        queue = self._make("queue", f"camera_v2_queue_{index}")

        source.connect("deep-element-added", self._configure_rtsp_child, camera)
        source.set_property("uri", camera.uri)
        self._set_if(source, "disable-audio", True)

        # DeepStream nvurisrcbin: 0 = UDP + multicast + TCP, 4 = TCP only.
        self._set_if(source, "select-rtp-protocol", 4 if transport == "tcp" else 0)
        self._set_if(source, "latency", self.rtsp_latency_ms)
        self._set_if(source, "drop-on-latency", True)
        self._set_if(source, "low-latency-mode", self.low_latency_mode)
        self._set_if(source, "num-extra-surfaces", self.extra_surfaces)
        self._set_if(source, "cudadec-memtype", 0)
        self._set_if(source, "udp-buffer-size", self.udp_buffer_size)
        self._set_if(source, "rtsp-reconnect-interval", 2)
        self._set_if(source, "rtsp-reconnect-attempts", -1)
        self._set_if(source, "message-forward", True)
        self._set_if(source, "async-handling", True)
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

        print(
            f"CAMERA_V2 {cid} source configured transport={transport} "
            f"latency={self.rtsp_latency_ms}ms keepalive={int(self._rtsp_keepalive(cid))}",
            flush=True,
        )

    def _source_pad_added(self, _source, pad, queue, cid: str) -> None:
        caps = pad.get_current_caps()
        if caps is None or caps.get_size() == 0:
            try:
                caps = pad.query_caps(None)
            except Exception:
                caps = None

        if caps is not None and caps.get_size() > 0 and not caps.is_any():
            try:
                media = str(caps.get_structure(0).get_name())
            except Exception:
                media = ""
            if media and not media.startswith("video/"):
                return

        sink_pad = queue.get_static_pad("sink")
        if sink_pad is None:
            print(f"CAMERA_V2 {cid} source queue sink pad missing", file=sys.stderr, flush=True)
            return
        if sink_pad.is_linked():
            return

        result = pad.link(sink_pad)
        if result != self.Gst.PadLinkReturn.OK:
            caps_text = caps.to_string() if caps is not None else "pending"
            print(
                f"CAMERA_V2 {cid} source link failed result={result} caps={caps_text}",
                file=sys.stderr,
                flush=True,
            )
            return

        caps_text = caps.to_string() if caps is not None else "pending"
        print(
            f"CAMERA_V2 {cid} source linked transport={self._transport()} caps={caps_text}",
            flush=True,
        )


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
