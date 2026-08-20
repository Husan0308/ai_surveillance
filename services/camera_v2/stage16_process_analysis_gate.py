from __future__ import annotations

"""Stage 16 exact-delta diagnostic: production-style analysis BUFFER gate.

Derives from hardware-proven Stage 12 and adds only the production startup gate
behavior on analysis_q.src:

    with capture_lock:
        requested = any(capture_requested.values())
    return OK if requested else DROP

All six capture_requested entries stay False in this stage, intentionally
matching production startup before the detector scheduler arms a capture.
Display must continue to flow while analysis downstream receives zero buffers.
No appsink, map/copy, metadata injection, detector, or tracker is present.
"""

from pathlib import Path


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"stage16 transform expected exactly one match for {old!r}, got {count}"
        )
    return text.replace(old, new, 1)


source_path = Path(__file__).with_name("stage12_process_analysis_convert.py")
source = source_path.read_text(encoding="utf-8")

source = source.replace("STAGE12", "STAGE16")
source = source.replace("Stage12", "Stage16")
source = source.replace("Stage 12", "Stage 16")
source = source.replace("stage12", "stage16")

# Stage 12 does not need threading; Stage 16 needs only a production-shaped lock
# around capture_requested while evaluating the BUFFER gate.
source = _replace_once(source, "import time\n", "import time\nimport threading\n")

# Add the gate state after source_counts exists so camera ids exactly match the
# six active inputs. Requests intentionally remain false for the whole run.
source = _replace_once(
    source,
    '    source_counts = {camera.camera_id: 0 for camera in cameras}\n    mux_request_pads = []',
    '''    source_counts = {camera.camera_id: 0 for camera in cameras}\n    capture_lock = threading.Lock()\n    capture_requested = {camera.camera_id: False for camera in cameras}\n    gate_counts = {"seen": 0, "drop": 0, "pass": 0}\n    mux_request_pads = []''',
)

# Install the production-style BUFFER-only gate after the ordinary diagnostic
# counter probe. Events/CAPS/SEGMENT continue downstream; only buffers drop.
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
    'f"parent={os.getppid()} child={os.getpid()} appsink=0 detector=0 tracker=0 gate=1 requests=0"',
)
source = source.replace(
    '"cameras=6 tee=1 analysis_tiler=1 bgrx=1 appsink=0 detector=0"',
    '"cameras=6 tee=1 analysis_tiler=1 bgrx=1 appsink=0 gate=1 requests=0 detector=0"',
)

source = _replace_once(
    source,
    '''            f"analysis_sink={counters['analysis_sink']} fps={counters['display_sink']/elapsed:.1f} "\n            f"analysis_caps={analysis_caps_text['value']} child_pid={os.getpid()}",''',
    '''            f"analysis_sink={counters['analysis_sink']} gate_seen={gate_counts['seen']} "\n            f"gate_drop={gate_counts['drop']} gate_pass={gate_counts['pass']} "\n            f"fps={counters['display_sink']/elapsed:.1f} "\n            f"analysis_caps={analysis_caps_text['value']} child_pid={os.getpid()}",''',
)
source = _replace_once(
    source,
    '_put_latest(status_q, ("STATS", dict(counters)))',
    '_put_latest(status_q, ("STATS", {**counters, **{f"gate_{k}": v for k, v in gate_counts.items()}}))',
)

# Execute in real module globals so multiprocessing spawn resolves _child_main.
exec(compile(source, str(source_path), "exec"), globals())
