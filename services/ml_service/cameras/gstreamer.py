"""NVIDIA/GStreamer RTSP capture adapter with latest-sample semantics."""
from __future__ import annotations

from urllib.parse import quote, urlsplit, urlunsplit
import threading

SUPPORTED_CODECS = {"h264": ("rtph264depay", "h264parse"), "h265": ("rtph265depay", "h265parse"), "hevc": ("rtph265depay", "h265parse")}
_GST_INIT_LOCK=threading.Lock()
_GST=None

def _gstreamer():
    global _GST
    with _GST_INIT_LOCK:
        if _GST is None:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
            Gst.init(None);_GST=Gst
        return _GST

def authenticated_source(config: dict) -> str | int:
    source = config.get("source", config.get("rtsp_url"))
    if isinstance(source, str) and source.isdigit(): return int(source)
    user, password = config.get("username"), config.get("password")
    if not isinstance(source, str) or not user or not password: return source
    parsed = urlsplit(source)
    if "@" in parsed.netloc: return source
    credentials = f"{quote(str(user), safe='')}:{quote(str(password), safe='')}@"
    return urlunsplit((parsed.scheme, credentials + parsed.netloc, parsed.path, parsed.query, parsed.fragment))

def _gst_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

def nvidia_rtsp_pipeline(config: dict) -> str:
    codec = str(config.get("codec", "")).lower()
    if codec not in SUPPORTED_CODECS: raise ValueError(f"camera {config.get('id')} requires codec=h264 or codec=h265")
    source = authenticated_source(config)
    if not isinstance(source, str) or not source.lower().startswith(("rtsp://", "rtsps://")): raise ValueError(f"camera {config.get('id')} NVIDIA backend requires an RTSP source")
    depay, parser = SUPPORTED_CODECS[codec]; latency = max(0, int(config.get("latency_ms", 50)))
    decoder_backend=str(config.get("decoder_backend","nvv4l2decoder")).lower()
    if decoder_backend=="nvcodec":
        decoder="nvh264dec max-display-delay=0" if codec=="h264" else "nvh265dec max-display-delay=0"
        conversion="video/x-raw,format=NV12 ! videoconvert"
    elif decoder_backend=="nvv4l2decoder":decoder="nvv4l2decoder low-latency-mode=true";conversion="nvvideoconvert ! video/x-raw,format=BGRx ! videoconvert"
    else:raise ValueError(f"unsupported NVIDIA decoder backend: {decoder_backend}")
    encoding="H264" if codec=="h264" else "H265"
    return (f"rtspsrc location={_gst_quote(source)} protocols=tcp latency={latency} drop-on-latency=true ! "
            f"application/x-rtp,media=video,encoding-name={encoding} ! "
            f"{depay} ! {parser} ! {decoder} ! {conversion} ! video/x-raw,format=BGR ! "
            "appsink name=sink drop=true max-buffers=1 sync=false")

class GStreamerCapture:
    """VideoCapture-compatible native GStreamer wrapper used by CameraReader."""
    backend = "gstreamer-nvdec"
    def __init__(self, config: dict):
        Gst=_gstreamer();self.Gst=Gst
        self.pipeline = nvidia_rtsp_pipeline(config)
        self._pipeline=Gst.parse_launch(self.pipeline);self._sink=self._pipeline.get_by_name("sink")
        if self._sink is None:self._sink=self._pipeline.get_by_name("appsink0")
        result=self._pipeline.set_state(Gst.State.PLAYING)
        # Live sources transition asynchronously. Waiting here would serialize camera startup.
        self._opened=result!=Gst.StateChangeReturn.FAILURE
    def isOpened(self): return self._opened
    def read(self):
        import numpy as np
        if not self._opened or self._sink is None:return False,None
        sample=self._sink.emit("try-pull-sample",self.Gst.SECOND)
        if sample is None:
            message=self._pipeline.get_bus().pop_filtered(self.Gst.MessageType.ERROR|self.Gst.MessageType.EOS)
            if message is not None:self._opened=False
            return False,None
        caps=sample.get_caps().get_structure(0);width=caps.get_value("width");height=caps.get_value("height")
        buffer=sample.get_buffer();ok,mapped=buffer.map(self.Gst.MapFlags.READ)
        if not ok:return False,None
        try:frame=np.frombuffer(mapped.data,dtype=np.uint8).reshape((height,width,3)).copy()
        finally:buffer.unmap(mapped)
        return True,frame
    def interrupt(self):
        # CameraReader will leave try-pull-sample within one second and perform
        # the state transition from the owning reader thread.
        self._opened=False
    def release(self):
        if getattr(self,"_pipeline",None) is not None:self._pipeline.set_state(self.Gst.State.NULL)
        self._opened=False
