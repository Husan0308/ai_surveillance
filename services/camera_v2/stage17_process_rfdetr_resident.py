from __future__ import annotations

"""Stage 17: proven Stage 16 graph plus the real RF-DETR-S worker resident on CUDA.

Exact diagnostic intent:
- keep the six-camera Stage 16 media/display graph and drop-all analysis BUFFER gate;
- spawn the production rfdetr_worker before PLAYING;
- let it do its normal startup delay, RFDETRSmall CUDA load and one warmup;
- send ZERO camera jobs after READY, so no live-frame inference/copy is involved;
- keep the model resident while the display pipeline continues.

This isolates detector-process CUDA/model memory pressure from detector scheduling,
frame delivery, metadata injection and tracking.
"""

from pathlib import Path


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"stage17 transform expected exactly one match for {old!r}, got {count}"
        )
    return text.replace(old, new, 1)


source_path = Path(__file__).with_name("stage12_process_analysis_convert.py")
source = source_path.read_text(encoding="utf-8")

source = source.replace("STAGE12", "STAGE17")
source = source.replace("Stage12", "Stage17")
source = source.replace("Stage 12", "Stage 17")
source = source.replace("stage12", "stage17")

# Stage 16 gate state.
source = _replace_once(source, "import time\n", "import time\nimport threading\n")
source = _replace_once(
    source,
    '    source_counts = {camera.camera_id: 0 for camera in cameras}\n    mux_request_pads = []\n    started = time.monotonic()',
    '''    source_counts = {camera.camera_id: 0 for camera in cameras}\n    capture_lock = threading.Lock()\n    capture_requested = {camera.camera_id: False for camera in cameras}\n    gate_counts = {"seen": 0, "drop": 0, "pass": 0}\n\n    # Production RF-DETR worker, but no jobs are ever submitted in Stage 17.\n    from services.camera_v2.rfdetr_backend import rfdetr_worker\n\n    detector_ctx = mp.get_context("spawn")\n    detector_job_q = detector_ctx.Queue(maxsize=1)\n    detector_result_q = detector_ctx.Queue(maxsize=2)\n    detector_worker = detector_ctx.Process(\n        target=rfdetr_worker,\n        args=(detector_job_q, detector_result_q),\n        name="stage17-rfdetr-resident",\n        daemon=True,\n    )\n    detector_state = {\n        "state": "STARTING",\n        "detail": "",\n        "ready": 0,\n        "fatal": 0,\n        "messages": 0,\n    }\n    detector_worker.start()\n    print(\n        f"STAGE17_RFDETR state=STARTING worker_pid={detector_worker.pid} "\n        f"startup_delay={os.environ.get('CAMERA_V2_DETECT_STARTUP_DELAY', '3.0')}s jobs=0",\n        flush=True,\n    )\n\n    mux_request_pads = []\n    started = time.monotonic()''',
)

# Same production-style drop-all startup gate as proven Stage 16.
source = _replace_once(
    source,
    '    analysis_q.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("analysis_q"))\n    analysis_tiler.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("analysis_tiler"))',
    '''    analysis_q.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("analysis_q"))\n\n    def analysis_gate_probe(_pad, _info):\n        gate_counts["seen"] += 1\n        with capture_lock:\n            requested = any(bool(v) for v in capture_requested.values())\n        if requested:\n            gate_counts["pass"] += 1\n            return Gst.PadProbeReturn.OK\n        gate_counts["drop"] += 1\n        return Gst.PadProbeReturn.DROP\n\n    analysis_q.get_static_pad("src").add_probe(\n        Gst.PadProbeType.BUFFER, analysis_gate_probe\n    )\n    analysis_tiler.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("analysis_tiler"))''',
)

source = source.replace(
    'analysis=tiler-{ANALYSIS_COLUMNS}x{ANALYSIS_ROWS}-BGRx-fakesink ',
    'analysis=gate-drop-all->tiler-{ANALYSIS_COLUMNS}x{ANALYSIS_ROWS}-BGRx-fakesink ',
)
source = source.replace(
    'f"parent={os.getppid()} child={os.getpid()} appsink=0 detector=0 tracker=0 gate=0"',
    'f"parent={os.getppid()} child={os.getpid()} appsink=0 detector=1 detector_jobs=0 tracker=0 gate=1 requests=0"',
)
source = source.replace(
    '"cameras=6 tee=1 analysis_tiler=1 bgrx=1 appsink=0 detector=0"',
    '"cameras=6 tee=1 analysis_tiler=1 bgrx=1 appsink=0 gate=1 requests=0 detector=1 detector_jobs=0"',
)

# Poll only worker status. Never place a job on detector_job_q.
source = _replace_once(
    source,
    '''    def stop_tick() -> bool:\n        if stop_event.is_set():\n            loop.quit()\n            return False\n        return True\n\n    GLib.timeout_add_seconds(2, stats_tick)\n    GLib.timeout_add(100, stop_tick)''',
    '''    def detector_tick() -> bool:\n        while True:\n            try:\n                message = detector_result_q.get_nowait()\n            except queue.Empty:\n                break\n            detector_state["messages"] += 1\n            kind = str(message.get("type") or "unknown")\n            if kind == "ready":\n                detector_state["state"] = "READY"\n                detector_state["ready"] = 1\n                detector_state["detail"] = (\n                    f"backend={message.get('backend')} model={message.get('model')} "\n                    f"device={message.get('device')} cuda={message.get('cuda')} "\n                    f"shape={message.get('shape')} threshold={message.get('threshold')}"\n                )\n                print(\n                    f"STAGE17_RFDETR state=READY worker_pid={detector_worker.pid} "\n                    f"{detector_state['detail']} jobs=0",\n                    flush=True,\n                )\n            elif kind == "fatal":\n                detector_state["state"] = "FATAL"\n                detector_state["fatal"] = 1\n                detector_state["detail"] = str(message.get("error") or "unknown")\n                print(\n                    f"STAGE17_RFDETR state=FATAL worker_pid={detector_worker.pid} "\n                    f"error={detector_state['detail']}",\n                    file=sys.stderr,\n                    flush=True,\n                )\n            else:\n                detector_state["state"] = kind.upper()\n                detector_state["detail"] = str(message)\n                print(\n                    f"STAGE17_RFDETR state={detector_state['state']} "\n                    f"worker_pid={detector_worker.pid} detail={message}",\n                    flush=True,\n                )\n        if not detector_worker.is_alive() and detector_state["state"] == "STARTING":\n            detector_state["state"] = "EXITED"\n            detector_state["detail"] = f"exitcode={detector_worker.exitcode}"\n            print(\n                f"STAGE17_RFDETR state=EXITED worker_pid={detector_worker.pid} "\n                f"exitcode={detector_worker.exitcode}",\n                flush=True,\n            )\n        return True\n\n    def stop_tick() -> bool:\n        if stop_event.is_set():\n            loop.quit()\n            return False\n        return True\n\n    GLib.timeout_add_seconds(2, stats_tick)\n    GLib.timeout_add(100, detector_tick)\n    GLib.timeout_add(100, stop_tick)''',
)

source = _replace_once(
    source,
    '''            f"analysis_sink={counters['analysis_sink']} fps={counters['display_sink']/elapsed:.1f} "\n            f"analysis_caps={analysis_caps_text['value']} child_pid={os.getpid()}",''',
    '''            f"analysis_sink={counters['analysis_sink']} gate_seen={gate_counts['seen']} "\n            f"gate_drop={gate_counts['drop']} gate_pass={gate_counts['pass']} "\n            f"detector_state={detector_state['state']} detector_ready={detector_state['ready']} "\n            f"detector_alive={int(detector_worker.is_alive())} detector_pid={detector_worker.pid} jobs=0 "\n            f"fps={counters['display_sink']/elapsed:.1f} "\n            f"analysis_caps={analysis_caps_text['value']} child_pid={os.getpid()}",''',
)
source = _replace_once(
    source,
    '_put_latest(status_q, ("STATS", dict(counters)))',
    '''_put_latest(status_q, ("STATS", {\n            **counters,\n            **{f"gate_{k}": v for k, v in gate_counts.items()},\n            "detector_state": detector_state["state"],\n            "detector_ready": detector_state["ready"],\n            "detector_alive": int(detector_worker.is_alive()),\n        }))''',
)

# Ensure the resident CUDA worker is released when the GStreamer child stops.
source = _replace_once(
    source,
    '''    finally:\n        pipeline.set_state(Gst.State.NULL)\n        bus.remove_signal_watch()''',
    '''    finally:\n        pipeline.set_state(Gst.State.NULL)\n        try:\n            detector_job_q.put_nowait(None)\n        except Exception:\n            pass\n        detector_worker.join(timeout=3.0)\n        if detector_worker.is_alive():\n            detector_worker.terminate()\n            detector_worker.join(timeout=1.0)\n        print(\n            f"STAGE17_RFDETR state=STOPPED worker_pid={detector_worker.pid} "\n            f"exitcode={detector_worker.exitcode}",\n            flush=True,\n        )\n        bus.remove_signal_watch()''',
)

# Execute in real module globals so both nested multiprocessing spawn targets are
# import-resolvable exactly like the proven diagnostics and production worker.
exec(compile(source, str(source_path), "exec"), globals())
