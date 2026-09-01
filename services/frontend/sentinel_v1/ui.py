from __future__ import annotations

# The Sentinel UI source is stored in line-preserving parts so the exact supplied
# interface can be carried without dropping pages/controls while CAM-01 is wired
# incrementally. The parts are concatenated and executed as this module.
from pathlib import Path as _Path

_parts_dir = _Path(__file__).with_name("ui_parts")
_source = "".join(path.read_text(encoding="utf-8") for path in sorted(_parts_dir.glob("part_*.pyfrag")))
exec(compile(_source, str(_parts_dir / "sentinel_ui_combined.py"), "exec"), globals(), globals())
del _source, _parts_dir, _Path
