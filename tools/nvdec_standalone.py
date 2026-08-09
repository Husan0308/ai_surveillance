#!/usr/bin/env python3
"""Standalone RTSP/NVDEC concurrency probe; imports no surveillance code."""
from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import time
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import gi
import yaml

gi.require_version("Gst", "1.0")
from gi.repository import Gst

ROOT = Path(__file__).resolve().parents[1]


def _yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_cameras():
    bootstrap = {str(item["id"]): item for item in (_yaml(ROOT / "config/cameras.yaml") or {}).get("cameras", [])}
    local = _yaml(ROOT / "config/cameras.local.yaml") or {}
    defaults = local.get("defaults", {})
    local_items = {str(item["id"]): item for item in local.get("cameras", []) if item.get("id")}
    database = ROOT / "data/surveillance.db"
    runtime = {}
    if database.exists():
        with sqlite3.connect(database) as db:
            rows = db.execute("SELECT id,data FROM api_resources WHERE resource='cameras'").fetchall()
        runtime = {str(camera_id): json.loads(data or "{}") for camera_id, data in rows}
    result = {}
    for camera_id in sorted(set(bootstrap) | set(runtime)):
        item = {**bootstrap.get(camera_id, {}), **runtime.get(camera_id, {}), **defaults, **local_items.get(camera_id, {})}
        item["id"] = camera_id
        item["source"] = runtime.get(camera_id, {}).get("rtsp_url", runtime.get(camera_id, {}).get("source", item.get("source")))
        result[camera_id] = item
    return result


def authenticated_uri(item):
    parsed = urlsplit(str(item["source"]))
    if "@" in parsed.netloc or not item.get("username"):
        return item["source"]
    auth = f"{quote(str(item['username']), safe='')}:{quote(str(item.get('password', '')), safe='')}@"
    return urlunsplit((parsed.scheme, auth + parsed.netloc, parsed.path, parsed.query, parsed.fragment))


class Probe:
    def __init__(self, item, decoder, extra_surfaces, memtype, output="fakesink"):
        self.item, self.frames, self.first_frame_at = item, 0, None
        codec = str(item.get("codec", "")).lower()
        depay, parser = (("rtph264depay", "h264parse") if codec == "h264" else ("rtph265depay", "h265parse"))
        encoding = "H264" if codec == "h264" else "H265"
        if decoder == "nvv4l2decoder":
            properties = [f"num-extra-surfaces={extra_surfaces}"]
            if memtype is not None: properties.append(f"cudadec-memtype={memtype}")
            decoder_desc = "nvv4l2decoder name=decoder " + " ".join(properties)
        else:
            decoder_desc = ("nvh264dec" if codec == "h264" else "nvh265dec") + " name=decoder max-display-delay=0"
        uri = authenticated_uri(item).replace("\\", "\\\\").replace('"', '\\"')
        tail=("fakesink sync=false async=false" if output=="fakesink" else
              "nvvideoconvert ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false")
        description = (f'rtspsrc location="{uri}" protocols=tcp latency={int(item.get("latency_ms", 20))} drop-on-latency=true ! '
                       f"application/x-rtp,media=video,encoding-name={encoding} ! "
                       f"{depay} ! {parser} ! {decoder_desc} ! {tail}")
        self.pipeline = Gst.parse_launch(description)
        self.decoder = self.pipeline.get_by_name("decoder")
        self.decoder.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, self._buffer)
        self.errors, self.warnings, self.started_at = [], [], 0.0

    def _buffer(self, _pad, _info):
        self.frames += 1
        if self.first_frame_at is None: self.first_frame_at = time.time()
        return Gst.PadProbeReturn.OK

    def start(self):
        self.started_at = time.time()
        change = self.pipeline.set_state(Gst.State.PLAYING)
        if change == Gst.StateChangeReturn.FAILURE: self.errors.append("set_state PLAYING failed")

    def poll_bus(self):
        bus = self.pipeline.get_bus()
        while True:
            message = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.WARNING | Gst.MessageType.EOS)
            if message is None: return
            if message.type == Gst.MessageType.ERROR:
                error, debug = message.parse_error(); self.errors.append(f"{error.domain}:{error.code}: {error.message} | {debug or ''}")
            elif message.type == Gst.MessageType.WARNING:
                warning, debug = message.parse_warning(); self.warnings.append(f"{warning.domain}:{warning.code}: {warning.message} | {debug or ''}")
            else: self.errors.append("unexpected EOS")

    def stop(self):
        self.poll_bus(); self.pipeline.set_state(Gst.State.NULL); self.pipeline.get_state(3 * Gst.SECOND)

    def result(self):
        return {"camera_id": self.item["id"], "codec": self.item.get("codec"), "frames": self.frames,
                "first_frame_ms": None if self.first_frame_at is None else round((self.first_frame_at-self.started_at)*1000, 1),
                "success": self.frames > 0 and not self.errors, "first_error": self.errors[0] if self.errors else None,
                "first_warning": self.warnings[0] if self.warnings else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("camera_ids", nargs="+")
    parser.add_argument("--decoder", choices=("nvv4l2decoder", "nvcodec"), default="nvv4l2decoder")
    parser.add_argument("--duration", type=float, default=12)
    parser.add_argument("--extra-surfaces", type=int, default=0)
    parser.add_argument("--memtype", type=int, choices=(0, 1, 2))
    parser.add_argument("--output", choices=("fakesink","bgr-appsink"), default="fakesink")
    args = parser.parse_args(); Gst.init(None); cameras = load_cameras()
    missing = [camera_id for camera_id in args.camera_ids if camera_id not in cameras]
    if missing: raise SystemExit(f"unknown cameras: {','.join(missing)}")
    probes = [Probe(cameras[camera_id], args.decoder, args.extra_surfaces, args.memtype,args.output) for camera_id in args.camera_ids]
    threads = [threading.Thread(target=probe.start) for probe in probes]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    deadline = time.monotonic() + args.duration
    while time.monotonic() < deadline:
        for probe in probes: probe.poll_bus()
        time.sleep(.1)
    for probe in probes: probe.stop()
    print(json.dumps({"decoder": args.decoder, "duration": args.duration, "extra_surfaces": args.extra_surfaces,
                      "memtype": args.memtype,"output":args.output,"results": [probe.result() for probe in probes]}, indent=2))


if __name__ == "__main__": main()
