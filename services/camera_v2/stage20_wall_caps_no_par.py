from __future__ import annotations

"""Stage 20: Stage 19 wall caps test with only pixel-aspect-ratio removed.

Exact delta from failed Stage 19:

    Stage 19: video/x-raw(memory:NVMM),width=1600,height=1350,pixel-aspect-ratio=1/1
    Stage 20: video/x-raw(memory:NVMM),width=1600,height=1350

Everything else remains the same as Stage 19 / proven Stage 18. This isolates
whether forcing PAR=1/1 is what breaks downstream caps negotiation on this
DeepStream/Pascal stack.
"""

from pathlib import Path


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"stage20 transform expected exactly one match for {old!r}, got {count}"
        )
    return text.replace(old, new, 1)


source_path = Path(__file__).with_name("stage19_display_wall_caps.py")
source = source_path.read_text(encoding="utf-8")

source = source.replace("STAGE19", "STAGE20")
source = source.replace("Stage19", "Stage20")
source = source.replace("Stage 19", "Stage 20")
source = source.replace("stage19", "stage20")

source = _replace_once(
    source,
    '"video/x-raw(memory:NVMM),width=1600,height=1350,pixel-aspect-ratio=1/1"',
    '"video/x-raw(memory:NVMM),width=1600,height=1350"',
)

source = source.replace(
    "wall_caps=NVMM-1600x1350",
    "wall_caps=NVMM-1600x1350-noPAR",
)

# Log one negotiated CAPS event at the relevant display pads. Event probes do not
# retain buffers and do not modify negotiation; they only expose what each element
# actually agreed to before the first buffer arrives.
needle = '''    wall_caps.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("wall_caps"))\n    sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe("display_sink"))'''
replacement = '''    wall_caps.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe("wall_caps"))\n    sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe("display_sink"))\n\n    caps_logged = set()\n\n    def caps_event_probe(label: str):\n        def _caps_event(_pad, info):\n            event = info.get_event()\n            if event is not None and event.type == Gst.EventType.CAPS and label not in caps_logged:\n                caps_logged.add(label)\n                try:\n                    caps = event.parse_caps()\n                    print(f"STAGE20_CAPS point={label} caps={caps.to_string()}", flush=True)\n                except Exception as exc:\n                    print(f"STAGE20_CAPS point={label} error={type(exc).__name__}:{exc}", flush=True)\n            return Gst.PadProbeReturn.OK\n        return _caps_event\n\n    display_tiler.get_static_pad("src").add_probe(\n        Gst.PadProbeType.EVENT_DOWNSTREAM, caps_event_probe("display_tiler_src")\n    )\n    wall_caps.get_static_pad("src").add_probe(\n        Gst.PadProbeType.EVENT_DOWNSTREAM, caps_event_probe("wall_caps_src")\n    )\n    display_convert.get_static_pad("src").add_probe(\n        Gst.PadProbeType.EVENT_DOWNSTREAM, caps_event_probe("display_convert_src")\n    )'''
source = _replace_once(source, needle, replacement)

exec(compile(source, str(source_path), "exec"), globals())
