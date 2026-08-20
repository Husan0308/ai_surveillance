from __future__ import annotations

"""Stage 20: production wall geometry caps without pixel-aspect-ratio.

Derives directly from hardware-proven Stage 12 to avoid transform-on-transform
wrapper failures. It keeps production display interpolation-method=4 and adds:

    display tiler -> NVMM wall caps(1600x1350, no PAR) -> display convert

No wall queue, detector, gate, metadata injection, or tracker is present.
One-shot downstream CAPS probes expose actual negotiation at the display tiler,
wall capsfilter and display convert.
"""

from pathlib import Path


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"stage20 transform expected exactly one match for {old!r}, got {count}"
        )
    return text.replace(old, new, 1)


source_path = Path(__file__).with_name("stage12_process_analysis_convert.py")
source = source_path.read_text(encoding="utf-8")

source = source.replace("STAGE12", "STAGE20")
source = source.replace("Stage12", "Stage20")
source = source.replace("Stage 12", "Stage 20")
source = source.replace("stage12", "stage20")

# Add the geometry capsfilter immediately after the display tiler.
source = _replace_once(
    source,
    '    display_tiler = make("nvmultistreamtiler", "stage20_display_tiler")\n    display_convert = make("nvvideoconvert", "stage20_display_convert")',
    '    display_tiler = make("nvmultistreamtiler", "stage20_display_tiler")\n    wall_caps = make("capsfilter", "stage20_wall_geometry")\n    display_convert = make("nvvideoconvert", "stage20_display_convert")',
)

# Preserve Stage 18's production Lanczos property. Stage 20 intentionally omits
# pixel-aspect-ratio so it is the only caps delta from failed Stage 19.
source = _replace_once(
    source,
    '    _set_if(display_tiler, "gpu-id", gpu_id)\n\n    _set_if(display_convert, "gpu-id", gpu_id)',
    '''    _set_if(display_tiler, "gpu-id", gpu_id)\n    _set_if(display_tiler, "interpolation-method", 4)\n    wall_caps.set_property(\n        "caps",\n        Gst.Caps.from_string(\n            "video/x-raw(memory:NVMM),width=1600,height=1350"\n        ),\n    )\n\n    _set_if(display_convert, "gpu-id", gpu_id)''',
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
    '''    display_tiler.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("display_tiler"))\n    wall_caps.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("wall_caps"))\n    sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe("display_sink"))\n\n    caps_logged = set()\n\n    def caps_event_probe(label: str):\n        def _caps_event(_pad, info):\n            event = info.get_event()\n            if (\n                event is not None\n                and event.type == Gst.EventType.CAPS\n                and label not in caps_logged\n            ):\n                caps_logged.add(label)\n                try:\n                    caps_value = event.parse_caps()\n                    print(\n                        f"STAGE20_CAPS point={label} caps={caps_value.to_string()}",\n                        flush=True,\n                    )\n                except Exception as exc:\n                    print(\n                        f"STAGE20_CAPS point={label} "\n                        f"error={type(exc).__name__}:{exc}",\n                        flush=True,\n                    )\n            return Gst.PadProbeReturn.OK\n        return _caps_event\n\n    display_tiler.get_static_pad("src").add_probe(\n        Gst.PadProbeType.EVENT_DOWNSTREAM,\n        caps_event_probe("display_tiler_src"),\n    )\n    wall_caps.get_static_pad("src").add_probe(\n        Gst.PadProbeType.EVENT_DOWNSTREAM,\n        caps_event_probe("wall_caps_src"),\n    )\n    display_convert.get_static_pad("src").add_probe(\n        Gst.PadProbeType.EVENT_DOWNSTREAM,\n        caps_event_probe("display_convert_src"),\n    )''',
)

source = source.replace(
    'f"parent={os.getppid()} child={os.getpid()} appsink=0 detector=0 tracker=0 gate=0"',
    'f"parent={os.getppid()} child={os.getpid()} display_interp=4 wall_caps=NVMM-1600x1350-noPAR appsink=0 detector=0 tracker=0 gate=0"',
)
source = source.replace(
    '"cameras=6 tee=1 analysis_tiler=1 bgrx=1 appsink=0 detector=0"',
    '"cameras=6 tee=1 display_interp=4 wall_caps=1 wall_par=0 analysis_tiler=1 bgrx=1 appsink=0 detector=0"',
)

# Execute in real module globals so multiprocessing spawn can resolve _child_main.
exec(compile(source, str(source_path), "exec"), globals())
