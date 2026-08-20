from __future__ import annotations

"""Stage 18 exact-delta diagnostic: display tiler interpolation-method=4 only.

Derives from hardware-proven Stage 12. The only graph/property change is adding
production's display nvmultistreamtiler interpolation-method=4. No production
wall_caps/wall_queue, detector, gate, metadata injection, or tracker is added.
"""

from pathlib import Path


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"stage18 transform expected exactly one match for {old!r}, got {count}"
        )
    return text.replace(old, new, 1)


source_path = Path(__file__).with_name("stage12_process_analysis_convert.py")
source = source_path.read_text(encoding="utf-8")

source = source.replace("STAGE12", "STAGE18")
source = source.replace("Stage12", "Stage18")
source = source.replace("Stage 12", "Stage 18")
source = source.replace("stage12", "stage18")

source = _replace_once(
    source,
    '    _set_if(display_tiler, "gpu-id", gpu_id)\n\n    _set_if(display_convert, "gpu-id", gpu_id)',
    '    _set_if(display_tiler, "gpu-id", gpu_id)\n    _set_if(display_tiler, "interpolation-method", 4)\n\n    _set_if(display_convert, "gpu-id", gpu_id)',
)

source = source.replace(
    'f"analysis=tiler-{ANALYSIS_COLUMNS}x{ANALYSIS_ROWS}-BGRx-fakesink "',
    'f"analysis=tiler-{ANALYSIS_COLUMNS}x{ANALYSIS_ROWS}-BGRx-fakesink "',
)
source = source.replace(
    'f"parent={os.getppid()} child={os.getpid()} appsink=0 detector=0 tracker=0 gate=0"',
    'f"parent={os.getppid()} child={os.getpid()} display_interp=4 appsink=0 detector=0 tracker=0 gate=0"',
)
source = source.replace(
    '"cameras=6 tee=1 analysis_tiler=1 bgrx=1 appsink=0 detector=0"',
    '"cameras=6 tee=1 display_interp=4 analysis_tiler=1 bgrx=1 appsink=0 detector=0"',
)

exec(compile(source, str(source_path), "exec"), globals())
