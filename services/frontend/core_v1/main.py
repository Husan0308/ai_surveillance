from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path


_BUNDLE_SHA256 = "65487172d3e63f96dbd59539c6da1cf050002b77a43f60799ed651b5fd65518e"
_PARTS = tuple(f"sentinel_ui_bundle.b64.{index:02d}" for index in range(5))


def _materialize_ui_bundle() -> Path:
    here = Path(__file__).resolve().parent
    payload = "".join((here / name).read_text(encoding="ascii").strip() for name in _PARTS)
    archive = base64.b64decode(payload, validate=True)
    digest = hashlib.sha256(archive).hexdigest()
    if digest != _BUNDLE_SHA256:
        raise RuntimeError(
            f"Sentinel UI bundle integrity failure: expected {_BUNDLE_SHA256}, got {digest}"
        )

    repo_root = Path(__file__).resolve().parents[3]
    runtime_dir = repo_root / ".runtime" / "sentinel_ui"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    bundle = runtime_dir / "sentinel_ui_bundle.zip"
    if not bundle.exists() or hashlib.sha256(bundle.read_bytes()).hexdigest() != digest:
        temporary = bundle.with_suffix(".zip.tmp")
        temporary.write_bytes(archive)
        temporary.replace(bundle)
    return bundle


def main() -> int:
    bundle = _materialize_ui_bundle()
    if str(bundle) not in sys.path:
        sys.path.insert(0, str(bundle))
    import sentinel_live

    return int(sentinel_live.run())


if __name__ == "__main__":
    raise SystemExit(main())
