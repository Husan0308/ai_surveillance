"""Local-file GStreamer source adapter for the existing Entrance appsrc path.

This module decodes the original camera recordings to native BGR frames. It
does not perform detection or tracking; callers pass these frames through the
same native scale-only transform and DeepStream appsrc path used by RTSP mode.
Source PTS comes from the decoded GstBuffer and is cross-checked against the
FFprobe best-effort timestamp table indexed by decoded source-frame ordinal.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE_BY_CAMERA = {
    "CAM-02": {
        "default_path": ROOT / "calibration_recordings/entrance/entrance-20261009T130526Z/camera_1.mp4",
        "sha256": "0216d2f8cf42996ec603a52882c2d12ee62974788057ffc7290c3636009a01b0",
        "width": 3200,
        "height": 1800,
        "frames": 3598,
        "nominal_fps": 20.0,
        "codec": "h264",
    },
    "CAM-05": {
        "default_path": ROOT / "calibration_recordings/entrance/entrance-20261009T130526Z/camera_2.mp4",
        "sha256": "0cd0ae6ddd31272d1ab049f7d81f12d199f0653ab2d0cd953fa6de87453f4370",
        "width": 2560,
        "height": 1440,
        "frames": 3600,
        "nominal_fps": 20.0,
        "codec": "h264",
    },
}
GST_CLOCK_TIME_NONE = (1 << 64) - 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rescale_time(value: Any, numerator: int, denominator: int) -> int | None:
    if value in (None, "N/A", ""):
        return None
    ticks = int(value)
    return (ticks * int(numerator) * 1_000_000_000) // int(denominator)


def inspect_recording(camera_id: str, path: Path) -> dict[str, Any]:
    """Hash- and stream-verify one original recording and retain its PTS map."""
    if camera_id not in SOURCE_BY_CAMERA:
        raise ValueError(f"unsupported recorded source {camera_id}")
    path = Path(path).expanduser().resolve(strict=True)
    expected = SOURCE_BY_CAMERA[camera_id]
    actual_hash = sha256_file(path)
    if actual_hash != expected["sha256"]:
        raise ValueError(f"{camera_id}: original recording SHA-256 mismatch")
    base = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,time_base,r_frame_rate,avg_frame_rate,nb_frames,duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=20,
    )
    document = json.loads(base.stdout)
    streams = document.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"{camera_id}: expected exactly one video stream")
    stream = streams[0]
    if (
        stream.get("codec_name") != expected["codec"]
        or int(stream.get("width", -1)) != expected["width"]
        or int(stream.get("height", -1)) != expected["height"]
        or int(stream.get("nb_frames", -1)) != expected["frames"]
    ):
        raise ValueError(f"{camera_id}: recording stream contract differs from the original source manifest")
    time_base = stream.get("time_base", "")
    try:
        time_num, time_den = (int(value) for value in time_base.split("/", 1))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{camera_id}: invalid source time_base") from exc
    if time_num <= 0 or time_den <= 0:
        raise ValueError(f"{camera_id}: invalid source time_base")

    frames_result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
         "-show_entries", "packet=pts,dts", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    frames_doc = json.loads(frames_result.stdout)
    frames = frames_doc.get("packets", [])
    if len(frames) != expected["frames"]:
        raise ValueError(f"{camera_id}: FFprobe video-packet/frame count mismatch")
    packets = []
    pts_ns: list[int] = []
    dts_ns: list[int] = []
    for packet_index, packet in enumerate(frames):
        pts = _rescale_time(packet.get("pts"), time_num, time_den)
        if pts is None:
            raise ValueError(f"{camera_id}: source packet PTS missing at packet {packet_index}")
        packets.append((pts, _rescale_time(packet.get("dts"), time_num, time_den) or -1))
    packets.sort(key=lambda row: row[0])
    for index, (pts, dts) in enumerate(packets):
        if pts_ns and pts <= pts_ns[-1]:
            raise ValueError(f"{camera_id}: source PTS is duplicated or nonmonotonic at decoded frame {index}")
        pts_ns.append(pts)
        dts_ns.append(dts)
    return {
        "camera_id": camera_id,
        "source_id": 0 if camera_id == "CAM-02" else 1,
        "path": str(path),
        "sha256": actual_hash,
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_count": len(frames),
        "nominal_fps": expected["nominal_fps"],
        "reported_r_frame_rate": stream.get("r_frame_rate"),
        "reported_avg_frame_rate": stream.get("avg_frame_rate"),
        "duration_sec": float(stream["duration"]),
        "time_base": time_base,
        "pts_ns_by_decoded_frame": pts_ns,
        "dts_ns_by_decoded_frame": dts_ns,
        "timestamp_provenance": "FFprobe compressed video packet PTS/DTS sorted into presentation order in the original MP4 stream time_base; packet count equals original declared frame count",
    }


@dataclass(frozen=True)
class RecordedFrame:
    frame_index: int
    source_pts_ns: int
    source_dts_ns: int
    duration_ns: int
    bgr: np.ndarray


class GstRecordedFileCapture:
    """Bounded uridecodebin → videoconvert → native-BGR appsink reader."""

    def __init__(self, source: dict[str, Any], *, appsink_buffers: int = 1):
        if appsink_buffers != 1:
            raise ValueError("recorded decoder appsink is intentionally bounded to one buffer")
        if source.get("camera_id") not in SOURCE_BY_CAMERA:
            raise ValueError("recorded source camera identity is invalid")
        self.source = source
        self.camera_id = source["camera_id"]
        self.path = Path(source["path"]).resolve(strict=True)
        self.width = int(source["width"])
        self.height = int(source["height"])
        self.expected_pts = source["pts_ns_by_decoded_frame"]
        self.expected_dts = source["dts_ns_by_decoded_frame"]
        self.decoded_frames = 0
        self.eos_seen = False
        self.closed = False
        self.close_state = "NOT_CLOSED"
        self.last_error = ""

        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstVideo
        Gst.init(None)
        self.Gst = Gst
        self.GstVideo = GstVideo

        self.pipeline = Gst.Pipeline.new(f"recorded-{self.camera_id.lower()}")
        self.filesrc = Gst.ElementFactory.make("filesrc", None)
        self.demux = Gst.ElementFactory.make("qtdemux", None)
        self.queue = Gst.ElementFactory.make("queue", None)
        self.parser = Gst.ElementFactory.make("h264parse", None)
        self.decoder = Gst.ElementFactory.make("avdec_h264", None)
        self.convert = Gst.ElementFactory.make("videoconvert", None)
        self.capsfilter = Gst.ElementFactory.make("capsfilter", None)
        self.appsink = Gst.ElementFactory.make("appsink", f"frames-{self.camera_id.lower()}")
        if not all((self.pipeline, self.filesrc, self.demux, self.queue, self.parser,
                    self.decoder, self.convert, self.capsfilter, self.appsink)):
            raise RuntimeError("host GStreamer is missing a required local-file decode element")
        self.filesrc.set_property("location", str(self.path))
        self.queue.set_property("max-size-buffers", 1)
        self.queue.set_property("max-size-bytes", 0)
        self.queue.set_property("max-size-time", 0)
        self.capsfilter.set_property(
            "caps", Gst.Caps.from_string(
                f"video/x-raw,format=BGR,width={self.width},height={self.height}"
            )
        )
        self.appsink.set_property("emit-signals", False)
        self.appsink.set_property("max-buffers", 1)
        self.appsink.set_property("drop", False)
        self.appsink.set_property("sync", False)
        self.appsink.set_property("async", False)
        self.appsink.set_property("wait-on-eos", False)
        self.pipeline.add(self.filesrc)
        for element in (self.demux, self.queue, self.parser, self.decoder,
                        self.convert, self.capsfilter, self.appsink):
            self.pipeline.add(element)
        if not self.filesrc.link(self.demux) or not self.queue.link(self.parser) \
                or not self.parser.link(self.decoder) or not self.decoder.link(self.convert) \
                or not self.convert.link(self.capsfilter) \
                or not self.capsfilter.link(self.appsink):
            raise RuntimeError("could not link local GStreamer video conversion path")
        self._video_pad_linked = False
        self._pts_to_frame = {int(pts): index for index, pts in enumerate(self.expected_pts)}
        if len(self._pts_to_frame) != len(self.expected_pts):
            raise ValueError(f"{self.camera_id}: source packet PTS is not a unique frame key")
        self.last_source_frame_index = -1
        self.demux.connect("pad-added", self._on_demux_pad)
        bus = self.pipeline.get_bus()
        self.bus = bus
        state_result = self.pipeline.set_state(Gst.State.PLAYING)
        if state_result == Gst.StateChangeReturn.FAILURE:
            self.close()
            raise RuntimeError("local GStreamer recorded-file pipeline refused PLAYING")
        self.pipeline.get_state(3 * Gst.SECOND)

    def seek_to_frame(self, source_frame_index: int) -> None:
        """Seeking is intentionally unsupported; decode ordinal remains authoritative."""
        if self.closed or source_frame_index < 0 or source_frame_index >= len(self.expected_pts):
            raise ValueError("recorded source seek index is outside the verified file")
        raise RuntimeError("recorded reader does not seek; source frame ordinals are decoded from frame zero")

    def _on_demux_pad(self, _demux, pad) -> None:
        if self._video_pad_linked:
            return
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            return
        if not caps.get_structure(0).get_name().startswith("video/x-h264"):
            return
        result = pad.link(self.queue.get_static_pad("sink"))
        if result == self.Gst.PadLinkReturn.OK:
            self._video_pad_linked = True
        else:
            self.last_error = f"demuxed H.264 video pad link failed: {result.value_nick}"

    def _check_bus(self) -> None:
        message = self.bus.pop_filtered(self.Gst.MessageType.ERROR | self.Gst.MessageType.EOS)
        if message is None:
            return
        if message.type == self.Gst.MessageType.EOS:
            self.eos_seen = True
            return
        error, _debug = message.parse_error()
        # Debug strings can contain full file paths and plugin detail; retain
        # only the GStreamer error's stable domain/message for private logs.
        self.last_error = f"{error.domain}: {error.message}"
        raise RuntimeError(f"{self.camera_id}: local GStreamer decode failed ({error.message})")

    def pull(self, timeout_ms: int = 500) -> RecordedFrame | None:
        if self.closed:
            raise RuntimeError("recorded decoder is already closed")
        timeout_ns = max(1, int(timeout_ms)) * self.Gst.MSECOND
        sample = self.appsink.emit("try-pull-sample", timeout_ns)
        if sample is None:
            self._check_bus()
            return None
        self._check_bus()
        caps = sample.get_caps()
        info = self.GstVideo.VideoInfo.new_from_caps(caps)
        if info is None:
            raise RuntimeError(f"{self.camera_id}: could not inspect decoded GstVideoInfo")
        if (int(info.width), int(info.height)) != (self.width, self.height):
            raise RuntimeError(
                f"{self.camera_id}: local decode geometry changed to {info.width}x{info.height}"
            )
        if caps.get_structure(0).get_string("format") != "BGR":
            raise RuntimeError(f"{self.camera_id}: appsink negotiated a non-BGR pixel format")
        buffer = sample.get_buffer()
        pts = int(buffer.pts)
        if pts == GST_CLOCK_TIME_NONE or self.decoded_frames >= len(self.expected_pts):
            raise RuntimeError(f"{self.camera_id}: decoded frame has no valid source PTS mapping")
        frame_index = self._pts_to_frame.get(pts)
        if frame_index is None:
            raise RuntimeError(f"{self.camera_id}: decoded PTS has no exact FFprobe packet crosswalk")
        if frame_index <= self.last_source_frame_index:
            raise RuntimeError(f"{self.camera_id}: decoded source-frame ordinal regressed")
        mapped, map_info = buffer.map(self.Gst.MapFlags.READ)
        if not mapped:
            raise RuntimeError(f"{self.camera_id}: could not map decoded BGR buffer")
        try:
            stride = int(info.stride[0])
            offset = int(info.offset[0])
            if stride < self.width * 3 or offset + stride * self.height > map_info.size:
                raise RuntimeError(f"{self.camera_id}: decoded BGR stride/layout is invalid")
            view = np.ndarray(
                (self.height, self.width, 3), dtype=np.uint8, buffer=map_info.data,
                offset=offset, strides=(stride, 3, 1),
            )
            # Detach unconditionally before unmapping the GstBuffer. Using
            # np.ascontiguousarray here is unsafe: when the mapped row stride
            # is already width*3 it returns `view` without copying, leaving
            # Qt/UI and DeepStream callers with a pointer into an unmapped or
            # recycled GStreamer buffer.
            bgr = view.copy(order="C")
        finally:
            buffer.unmap(map_info)
        dts = int(buffer.dts)
        if dts == GST_CLOCK_TIME_NONE:
            dts = int(self.expected_dts[frame_index])
        if dts < 0:
            dts = -1
        duration = int(buffer.duration)
        if duration == GST_CLOCK_TIME_NONE:
            duration = -1
        self.decoded_frames += 1
        self.last_source_frame_index = frame_index
        return RecordedFrame(frame_index, pts, dts, duration, bgr)

    def close(self, timeout_ms: int = 3000) -> dict[str, Any]:
        if self.closed:
            return {"closed": True, "close_state": self.close_state, "eos_seen": self.eos_seen}
        self.closed = True
        result = self.pipeline.set_state(self.Gst.State.NULL)
        if result == self.Gst.StateChangeReturn.FAILURE:
            self.close_state = "NULL_STATE_FAILED"
            raise RuntimeError(f"{self.camera_id}: recorded GStreamer pipeline failed transition to NULL")
        self.pipeline.get_state(max(1, int(timeout_ms)) * self.Gst.MSECOND)
        self.close_state = "NULL"
        self._check_bus()
        return {"closed": True, "close_state": self.close_state, "eos_seen": self.eos_seen}


def playback_deadline_ns(playback_start_mono_ns: int, first_source_pts_ns: int,
                         source_pts_ns: int, playback_speed: float = 1.0) -> int:
    """Map original media PTS deltas onto a local monotonic playback clock.

    Source PTS is never modified. A speed of 2.0 paces a 30-second source
    segment into 15 seconds of wall time.
    """
    if playback_start_mono_ns < 0 or first_source_pts_ns < 0 or source_pts_ns < first_source_pts_ns:
        raise ValueError("recorded playback requires ordered nonnegative source PTS")
    if not math.isfinite(float(playback_speed)) or float(playback_speed) <= 0.0:
        raise ValueError("recorded playback speed must be finite and positive")
    return int(playback_start_mono_ns + (source_pts_ns - first_source_pts_ns) / float(playback_speed))
