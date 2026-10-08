"""Capture one synchronized, stream-copy RTSP pass for CAM-01 and CAM-04."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.ml_service.app.config import load_settings

CAMERAS = ("CAM-01", "CAM-04")


def capture(output: Path, duration: int) -> dict:
    Gst.init(None)
    settings = load_settings(ROOT / "config/cameras.yaml")
    camera_rows = {camera.camera_id: camera for camera in settings.cameras}
    if set(CAMERAS) - camera_rows.keys():
        raise RuntimeError("CAM-01 and CAM-04 must be configured")
    output.mkdir(parents=True, exist_ok=False)
    pipeline = Gst.Pipeline.new("priority-rtsp-streamcopy")
    loop = GLib.MainLoop()
    state = {camera: {"rtp_buffers": 0, "rtp_first_pts": None, "rtp_last_pts": None,
                      "rtp_caps": None, "pad_linked": False} for camera in CAMERAS}
    errors: list[str] = []
    start_ns = time.monotonic_ns()
    stopped_ns: int | None = None

    for index, camera_id in enumerate(CAMERAS):
        camera = camera_rows[camera_id]
        codec = "h264" if camera_id == "CAM-01" else "h265"
        source = Gst.ElementFactory.make("rtspsrc", f"{camera_id}-rtsp")
        depay = Gst.ElementFactory.make(f"rtp{codec}depay", f"{camera_id}-depay")
        parser = Gst.ElementFactory.make(f"{codec}parse", f"{camera_id}-parse")
        mux = Gst.ElementFactory.make("qtmux", f"{camera_id}-qtmux")
        sink = Gst.ElementFactory.make("filesink", f"{camera_id}-file")
        if any(element is None for element in (source, depay, parser, mux, sink)):
            raise RuntimeError(f"required GStreamer stream-copy element missing for {camera_id}")
        source.set_property("location", camera.uri)
        source.set_property("user-id", camera.username)
        source.set_property("user-pw", camera.password)
        source.set_property("protocols", 4)
        source.set_property("latency", camera.effective_latency_ms(settings.deepstream.latency_ms))
        source.set_property("drop-on-latency", bool(settings.deepstream.drop_on_latency))
        sink.set_property("location", str(output / ("cam_00.mp4" if index == 0 else "cam_01.mp4")))
        sink.set_property("sync", False)
        for element in (source, depay, parser, mux, sink):
            pipeline.add(element)
        if not depay.link(parser) or not parser.link(mux) or not mux.link(sink):
            raise RuntimeError(f"stream-copy link failed for {camera_id}")

        def on_pad_added(_source, pad, *, cid=camera_id, depay_element=depay):
            caps = pad.get_current_caps() or pad.query_caps(None)
            caps_text = caps.to_string() if caps else "unknown"
            structure = caps.get_structure(0) if caps and caps.get_size() else None
            media = structure.get_string("media") if structure else None
            if media != "video":
                return
            encoding = structure.get_string("encoding-name") if structure else None
            if encoding and encoding.upper() not in ("H264", "H265"):
                errors.append(f"{cid}: unexpected RTSP encoding {encoding}")
                return
            state[cid]["rtp_caps"] = caps_text.split(", codec_data", 1)[0]
            sink_pad = depay_element.get_static_pad("sink")
            if not sink_pad.is_linked():
                result = pad.link(sink_pad)
                state[cid]["pad_linked"] = result == Gst.PadLinkReturn.OK
                if result != Gst.PadLinkReturn.OK:
                    errors.append(f"{cid}: RTP pad link failed: {result.value_nick}")

                def packet_probe(_pad, info, *, camera=cid):
                    buffer = info.get_buffer()
                    if buffer is not None:
                        current = state[camera]
                        current["rtp_buffers"] += 1
                        pts = int(buffer.pts) if buffer.pts != Gst.CLOCK_TIME_NONE else None
                        if current["rtp_first_pts"] is None:
                            current["rtp_first_pts"] = pts
                        current["rtp_last_pts"] = pts
                    return Gst.PadProbeReturn.OK
                pad.add_probe(Gst.PadProbeType.BUFFER, packet_probe)

        source.connect("pad-added", on_pad_added)

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(_bus, message):
        nonlocal stopped_ns
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            # Do not include a URI or element property in diagnostic output.
            errors.append(f"{message.src.get_name()}: {err.message}; debug={debug or ''}")
            loop.quit()
        elif message.type == Gst.MessageType.EOS:
            stopped_ns = time.monotonic_ns()
            loop.quit()

    bus.connect("message", on_message)
    change = pipeline.set_state(Gst.State.PLAYING)
    if change == Gst.StateChangeReturn.FAILURE:
        raise RuntimeError("capture pipeline failed to enter PLAYING")

    def stop_after_duration():
        nonlocal stopped_ns
        stopped_ns = time.monotonic_ns()
        pipeline.send_event(Gst.Event.new_eos())
        return GLib.SOURCE_REMOVE

    timer = GLib.timeout_add_seconds(duration, stop_after_duration)
    try:
        loop.run()
    finally:
        GLib.source_remove(timer)
        pipeline.set_state(Gst.State.NULL)
        bus.remove_signal_watch()

    summary = {
        "duration_requested_sec": duration,
        "started_monotonic_ns": start_ns,
        "stopped_monotonic_ns": stopped_ns,
        "elapsed_sec": ((stopped_ns - start_ns) / 1e9) if stopped_ns else None,
        "capture_method": "RTSP RTP depay + codec parser + qtmux; no video re-encode",
        "rtsp_reader_count": 2,
        "sources": state,
        "errors": errors,
    }
    (output / "capture-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if errors or not all(item["pad_linked"] and item["rtp_buffers"] for item in state.values()):
        raise RuntimeError("capture incomplete; see capture-summary.json")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=60)
    args = parser.parse_args()
    if args.duration < 20:
        parser.error("duration must be >=20 seconds")
    print(json.dumps(capture(args.output.resolve(), args.duration), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
