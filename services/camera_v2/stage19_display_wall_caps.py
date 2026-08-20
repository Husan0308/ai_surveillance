from __future__ import annotations

"""Stage 19 exact-delta diagnostic: production display wall capsfilter.

Derives from hardware-proven Stage 12 and keeps proven Stage 18's display
interpolation-method=4. Adds only the production-shaped geometry capsfilter:

    display tiler -> NVMM wall caps(1600x1350, PAR 1/1) -> display convert

No production wall_queue, detector, gate, metadata injection, or tracker is added.
"""

from pathlib import Path


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"stage19 transform expected exactly one match for {old!r}, got {count}"
        )
    return text.replace(old, new, 1)


source_path = Path(__file__).with_name("stage12_process_analysis_convert.py")
source = source_path.read_text(encoding="utf-8")

source = source.replace("STAGE12", "STAGE19")
source = source.replace("Stage12", "Stage19")
source = source.replace("Stage 12", "Stage 19")
source = source.replace("stage12", "stage19")

# Add the production geometry capsfilter immediately after the display tiler.
source = _replace_once(
    source,
    '    display_tiler = make("nvmultistreamtiler", "stage19_display_tiler")\n    display_convert = make("nvvideoconvert", "stage19_display_convert")',
    '    display_tiler = make("nvmultistreamtiler", "stage19_display_tiler")\n    wall_caps = make("capsfilter", "stage19_wall_geometry")\n    display_convert = make("nvvideoconvert", "stage19_display_convert")',
)

# Preserve Stage 18's production Lanczos property and add exact wall caps.
source = _replace_once(
    source,
    '    _set_if(display_tiler, "gpu-id", gpu_id)\n\n    _set_if(display_convert, "gpu-id", gpu_id)',
    '''    _set_if(display_tiler, "gpu-id", gpu_id)\n    _set_if(display_tiler, "interpolation-method", 4)\n    wall_caps.set_property(\n        "caps",\n        Gst.Caps.from_string(\n            "video/x-raw(memory:NVMM),width=1600,height=1350,pixel-aspect-ratio=1/1"\n        ),\n    )\n\n    _set_if(display_convert, "gpu-id", gpu_id)''',
)

# Insert wall_caps into the pipeline element set.
source = _replace_once(
    source,
    '        display_tiler,\n        display_convert,',
    '        display_tiler,\n        wall_caps,\n        display_convert,',
)

# Replace direct tiler->convert with tiler->wall_caps->convert.
source = _replace_once(
    source,
    '        (display_tiler, display_convert, "display_tiler -> display_convert"),\n        (display_convert, display_caps, "display_convert -> RGBA"),',
    '        (display_tiler, wall_caps, "display_tiler -> wall_caps"),\n        (wall_caps, display_convert, "wall_caps -> display_convert"),\n        (display_convert, display_caps, "display_convert -> RGBA"),',
)

# Add an independent counter after geometry caps negotiation.
source = _replace_once(
    source,
    '        "display_tiler": 0,\n        "display_sink": 0,',
    '        "display_tiler": 0,\n        "wall_caps": 0,\n        "display_sink": 0,',
)
source = _replace_once(
    source,
    '    display_tiler.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("display_tiler"))\n    sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe("display_sink"))',
    '    display_tiler.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("display_tiler"))\n    wall_caps.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("wall_caps"))\n    sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe("display_sink"))',
)

# Make the diagnostic contract visible in child startup logs.
source = source.replace(
    'f"parent={os.getppid()} child={os.getpid()} appsink=0 detector=0 tracker=0 gate=0"',
    'f"parent={os.getppid()} child={os.getpid()} display_interp=4 wall_caps=NVMM-1600x1350 appsink=0 detector=0 tracker=0 gate=0"',
)
source = source.replace(
    '"cameras=6 tee=1 analysis_tiler=1 bgrx=1 appsink=0 detector=0"',
    '"cameras=6 tee=1 display_interp=4 wall_caps=1 analysis_tiler=1 bgrx=1 appsink=0 detector=0"',
)

exec(compile(source, str(source_path), "exec"), globals())
