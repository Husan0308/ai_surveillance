from __future__ import annotations

"""Stage 22: final wall-geometry confirmation using aligned NVMM height.

Derives directly from hardware-proven Stage 12. Relative to failed Stage 19 the
ONLY caps difference is height=1352 instead of height=1350:

    Stage 19: NVMM, 1600x1350, PAR=1/1  -> not-negotiated
    Stage 22: NVMM, 1600x1352, PAR=1/1  -> expected aligned geometry

Stage 21 proved that the display tiler actually negotiates 1600x1352 when its
requested presentation height is 1350. No wall queue, detector, gate, metadata,
or tracker is present here.
"""

from pathlib import Path


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"stage22 transform expected exactly one match for {old!r}, got {count}"
        )
    return text.replace(old, new, 1)


source_path = Path(__file__).with_name("stage12_process_analysis_convert.py")
source = source_path.read_text(encoding="utf-8")

source = source.replace("STAGE12", "STAGE22")
source = source.replace("Stage12", "Stage22")
source = source.replace("Stage 12", "Stage 22")
source = source.replace("stage12", "stage22")

source = _replace_once(
    source,
    '    display_tiler = make("nvmultistreamtiler", "stage22_display_tiler")\n    display_convert = make("nvvideoconvert", "stage22_display_convert")',
    '    display_tiler = make("nvmultistreamtiler", "stage22_display_tiler")\n    wall_caps = make("capsfilter", "stage22_wall_geometry")\n    display_convert = make("nvvideoconvert", "stage22_display_convert")',
)

source = _replace_once(
    source,
    '    _set_if(display_tiler, "gpu-id", gpu_id)\n\n    _set_if(display_convert, "gpu-id", gpu_id)',
    '''    _set_if(display_tiler, "gpu-id", gpu_id)\n    _set_if(display_tiler, "interpolation-method", 4)\n    wall_caps.set_property(\n        "caps",\n        Gst.Caps.from_string(\n            "video/x-raw(memory:NVMM),width=1600,height=1352,pixel-aspect-ratio=1/1"\n        ),\n    )\n\n    _set_if(display_convert, "gpu-id", gpu_id)''',
)

source = _replace_once(
    source,
    '        display_tiler,\n        display_convert,',
    '        display_tiler,\n        wall_caps,\n        display_convert,',
)

source = _replace_once(
    source,
    '        (display_tiler, display_convert, "display_tiler -> display_convert"),\n        (display_convert, display_caps, "display_convert -> RGBA"),',
    '        (display_tiler, wall_caps, "display_tiler -> wall_caps"),\n        (wall_caps, display_convert, "wall_caps -> display_convert"),\n        (display_convert, display_caps, "display_convert -> RGBA"),',
)

source = _replace_once(
    source,
    '        "display_tiler": 0,\n        "display_sink": 0,',
    '        "display_tiler": 0,\n        "wall_caps": 0,\n        "display_sink": 0,',
)

source = _replace_once(
    source,
    '    display_tiler.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("display_tiler"))\n    sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe("display_sink"))',
    '''    display_tiler.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("display_tiler"))\n    wall_caps.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("wall_caps"))\n    sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe("display_sink"))\n\n    caps_logged = set()\n\n    def caps_event_probe(label: str):\n        def _caps_event(_pad, info):\n            event = info.get_event()\n            if event is not None and event.type == Gst.EventType.CAPS and label not in caps_logged:\n                caps_logged.add(label)\n                try:\n                    caps_value = event.parse_caps()\n                    print(\n                        f"STAGE22_CAPS point={label} caps={caps_value.to_string()}",\n                        flush=True,\n                    )\n                except Exception as exc:\n                    print(\n                        f"STAGE22_CAPS point={label} error={type(exc).__name__}:{exc}",\n                        flush=True,\n                    )\n            return Gst.PadProbeReturn.OK\n        return _caps_event\n\n    display_tiler.get_static_pad("src").add_probe(\n        Gst.PadProbeType.EVENT_DOWNSTREAM, caps_event_probe("display_tiler_src")\n    )\n    wall_caps.get_static_pad("src").add_probe(\n        Gst.PadProbeType.EVENT_DOWNSTREAM, caps_event_probe("wall_caps_src")\n    )\n    display_convert.get_static_pad("src").add_probe(\n        Gst.PadProbeType.EVENT_DOWNSTREAM, caps_event_probe("display_convert_src")\n    )''',
)

source = source.replace(
    'f"parent={os.getppid()} child={os.getpid()} appsink=0 detector=0 tracker=0 gate=0"',
    'f"parent={os.getppid()} child={os.getpid()} display_interp=4 wall_caps=NVMM-1600x1352-PAR1/1 appsink=0 detector=0 tracker=0 gate=0"',
)
source = source.replace(
    '"cameras=6 tee=1 analysis_tiler=1 bgrx=1 appsink=0 detector=0"',
    '"cameras=6 tee=1 display_interp=4 wall_caps=1 wall_geometry=1600x1352 wall_par=1 analysis_tiler=1 bgrx=1 appsink=0 detector=0"',
)

exec(compile(source, str(source_path), "exec"), globals())
