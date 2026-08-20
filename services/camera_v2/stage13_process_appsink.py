from __future__ import annotations

"""Stage 13 exact-delta diagnostic.

This intentionally derives from the already hardware-proven Stage 12 source at
runtime and changes only the analysis sink contract:

    fakesink -> appsink(drop=true,max-buffers=1,emit-signals=false)

There is still no new-sample callback, pull-sample, buffer map/copy, gate probe,
detector or tracker. Keeping Stage 12 as the source makes the test resistant to
accidentally introducing unrelated graph differences while isolating appsink.
"""

from pathlib import Path


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"stage13 transform expected exactly one match for {old!r}, got {count}"
        )
    return text.replace(old, new, 1)


source_path = Path(__file__).with_name("stage12_process_analysis_convert.py")
source = source_path.read_text(encoding="utf-8")

# Rename diagnostic labels/module-local symbols only. This does not alter graph
# semantics and makes the hardware log unambiguous.
source = source.replace("STAGE12", "STAGE13")
source = source.replace("Stage12", "Stage13")
source = source.replace("Stage 12", "Stage 13")
source = source.replace("stage12", "stage13")

source = _replace_once(
    source,
    'analysis_sink = make("fakesink", "stage13_analysis_sink")',
    'analysis_sink = make("appsink", "stage13_analysis_sink")',
)
source = _replace_once(
    source,
    '''    _set_if(analysis_sink, "sync", False)\n    _set_if(analysis_sink, "async", False)\n    _set_if(analysis_sink, "enable-last-sample", False)''',
    '''    _set_if(analysis_sink, "sync", False)\n    _set_if(analysis_sink, "async", False)\n    analysis_sink.set_property("emit-signals", False)\n    analysis_sink.set_property("drop", True)\n    analysis_sink.set_property("max-buffers", 1)\n    _set_if(analysis_sink, "enable-last-sample", False)\n    _set_if(analysis_sink, "wait-on-eos", False)''',
)
source = _replace_once(
    source,
    '(analysis_caps, analysis_sink, "analysis BGRx -> fakesink"),',
    '(analysis_caps, analysis_sink, "analysis BGRx -> appsink"),',
)
source = _replace_once(
    source,
    'analysis=tiler-{ANALYSIS_COLUMNS}x{ANALYSIS_ROWS}-BGRx-fakesink ',
    'analysis=tiler-{ANALYSIS_COLUMNS}x{ANALYSIS_ROWS}-BGRx-appsink-no-pull ',
)
source = source.replace(
    "-> nvvideoconvert -> system-memory BGRx caps -> fakesink",
    "-> nvvideoconvert -> system-memory BGRx caps -> appsink (no pull)",
)
source = source.replace(
    "No gate probe, appsink callback, frame mapping/copy, detector or tracker is present.",
    "No gate probe, appsink callback/pull, frame mapping/copy, detector or tracker is present.",
)
source = source.replace(
    '"cameras=6 tee=1 analysis_tiler=1 bgrx=1 appsink=0 detector=0"',
    '"cameras=6 tee=1 analysis_tiler=1 bgrx=1 appsink=1 pull=0 detector=0"',
)
source = source.replace(
    'f"parent={os.getppid()} child={os.getpid()} appsink=0 detector=0 tracker=0 gate=0"',
    'f"parent={os.getppid()} child={os.getpid()} appsink=1 pull=0 callback=0 detector=0 tracker=0 gate=0"',
)

# Execute the transformed diagnostic exactly as a module entry point.
exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})
