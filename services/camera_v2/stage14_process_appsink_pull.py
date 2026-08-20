from __future__ import annotations

"""Stage 14 exact-delta diagnostic.

Derive from hardware-proven Stage 12 and change only the analysis sink behavior:

    fakesink -> appsink(drop=true,max-buffers=1,emit-signals=true)
    new-sample callback -> pull-sample -> immediately release Python reference

There is still no Gst.Buffer map, memory copy, NumPy conversion, gate probe,
detector or tracker. This isolates the appsink signal/callback/pull lifecycle.
"""

from pathlib import Path


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"stage14 transform expected exactly one match for {old!r}, got {count}"
        )
    return text.replace(old, new, 1)


source_path = Path(__file__).with_name("stage12_process_analysis_convert.py")
source = source_path.read_text(encoding="utf-8")

source = source.replace("STAGE12", "STAGE14")
source = source.replace("Stage12", "Stage14")
source = source.replace("Stage 12", "Stage 14")
source = source.replace("stage12", "stage14")

source = _replace_once(
    source,
    'analysis_sink = make("fakesink", "stage14_analysis_sink")',
    'analysis_sink = make("appsink", "stage14_analysis_sink")',
)

source = _replace_once(
    source,
    '''    _set_if(analysis_sink, "sync", False)\n    _set_if(analysis_sink, "async", False)\n    _set_if(analysis_sink, "enable-last-sample", False)''',
    '''    _set_if(analysis_sink, "sync", False)\n    _set_if(analysis_sink, "async", False)\n    analysis_sink.set_property("emit-signals", True)\n    analysis_sink.set_property("drop", True)\n    analysis_sink.set_property("max-buffers", 1)\n    _set_if(analysis_sink, "enable-last-sample", False)\n    _set_if(analysis_sink, "wait-on-eos", False)\n\n    appsink_counts = {"callbacks": 0, "pulled": 0, "null": 0}\n\n    def on_new_sample(appsink):\n        appsink_counts["callbacks"] += 1\n        sample = appsink.emit("pull-sample")\n        if sample is None:\n            appsink_counts["null"] += 1\n            return Gst.FlowReturn.OK\n        appsink_counts["pulled"] += 1\n        # Deliberately do not inspect/map/copy the buffer in Stage 14.\n        del sample\n        return Gst.FlowReturn.OK\n\n    analysis_sink.connect("new-sample", on_new_sample)''',
)

source = _replace_once(
    source,
    '(analysis_caps, analysis_sink, "analysis BGRx -> fakesink"),',
    '(analysis_caps, analysis_sink, "analysis BGRx -> appsink pull-only"),',
)

source = _replace_once(
    source,
    'analysis=tiler-{ANALYSIS_COLUMNS}x{ANALYSIS_ROWS}-BGRx-fakesink ',
    'analysis=tiler-{ANALYSIS_COLUMNS}x{ANALYSIS_ROWS}-BGRx-appsink-pull-only ',
)

source = source.replace(
    "-> nvvideoconvert -> system-memory BGRx caps -> fakesink",
    "-> nvvideoconvert -> system-memory BGRx caps -> appsink -> pull-sample only",
)
source = source.replace(
    "No gate probe, appsink callback, frame mapping/copy, detector or tracker is present.",
    "Appsink callback/pull is present; no gate probe, frame mapping/copy, detector or tracker is present.",
)
source = source.replace(
    '"cameras=6 tee=1 analysis_tiler=1 bgrx=1 appsink=0 detector=0"',
    '"cameras=6 tee=1 analysis_tiler=1 bgrx=1 appsink=1 callback=1 pull=1 map=0 copy=0 detector=0"',
)
source = source.replace(
    'f"parent={os.getppid()} child={os.getpid()} appsink=0 detector=0 tracker=0 gate=0"',
    'f"parent={os.getppid()} child={os.getpid()} appsink=1 callback=1 pull=1 map=0 copy=0 detector=0 tracker=0 gate=0"',
)

source = _replace_once(
    source,
    '''            f"analysis_sink={counters['analysis_sink']} fps={counters['display_sink']/elapsed:.1f} "\n            f"analysis_caps={analysis_caps_text['value']} child_pid={os.getpid()}",''',
    '''            f"analysis_sink={counters['analysis_sink']} callback={appsink_counts['callbacks']} "\n            f"pulled={appsink_counts['pulled']} null={appsink_counts['null']} "\n            f"fps={counters['display_sink']/elapsed:.1f} "\n            f"analysis_caps={analysis_caps_text['value']} child_pid={os.getpid()}",''',
)
source = _replace_once(
    source,
    '_put_latest(status_q, ("STATS", dict(counters)))',
    '_put_latest(status_q, ("STATS", {**counters, **appsink_counts}))',
)

# Execute in the real module globals so multiprocessing spawn can pickle
# _child_main by the __main__ module/name exactly like the proven Stage 12/13.
exec(compile(source, str(source_path), "exec"), globals())
