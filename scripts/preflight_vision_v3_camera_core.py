from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ROOT / "config" / "cameras.yaml"
CORE = ROOT / "config" / "vision_v3_camera.yaml"
EXPECTED = {"CAM-01", "CAM-02", "CAM-03", "CAM-04", "CAM-05", "CAM-06"}
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def fail(message: str) -> None:
    print(f"VISION_V3_CAMERA_PREFLIGHT=FAIL {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def load_yaml(path: Path) -> dict:
    if not path.exists():
        fail(f"missing {path.relative_to(ROOT)}")
    with path.open("r", encoding="utf-8") as fh:
        value = yaml.safe_load(fh) or {}
    if not isinstance(value, dict):
        fail(f"invalid YAML root: {path.relative_to(ROOT)}")
    return value


def expand_env(value: object) -> str:
    text = str(value or "").strip()
    return ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), text)


def main() -> int:
    camera_data = load_yaml(CAMERAS)
    core_data = load_yaml(CORE)
    rows = [
        row for row in (camera_data.get("cameras") or [])
        if isinstance(row, dict)
        and row.get("enabled", True) is not False
        and row.get("online", True) is not False
    ]
    ids = {str(row.get("id") or row.get("camera_id") or "") for row in rows}
    if ids != EXPECTED or len(rows) != 6:
        fail(f"expected exactly {sorted(EXPECTED)}, got {sorted(ids)}")

    auth_count = 0
    feeds: list[str] = []
    for row in rows:
        cid = str(row.get("id") or row.get("camera_id"))
        uri = str(row.get("display_source") or row.get("source") or row.get("uri") or "")
        if not uri.startswith("rtsp://"):
            fail(f"{cid} missing usable display/source RTSP URI")
        feeds.append(f"{cid}={uri.rsplit('/', 1)[-1]}")
        username = expand_env(row.get("username")) or os.environ.get("SURVEILLANCE_RTSP_USERNAME", "")
        password = expand_env(row.get("password")) or os.environ.get("SURVEILLANCE_RTSP_PASSWORD", "")
        if username and password:
            auth_count += 1

    if auth_count != 6:
        fail(
            "RTSP credentials unresolved for one or more cameras; set "
            "SURVEILLANCE_RTSP_USERNAME and SURVEILLANCE_RTSP_PASSWORD in the local .env"
        )

    cfg = dict(core_data.get("camera_core") or {})
    if int(cfg.get("queue_buffers", 0)) != 1:
        fail("queue_buffers must remain 1")
    if bool(cfg.get("sync_inputs", True)):
        fail("sync_inputs must remain false")
    if bool(cfg.get("sink_sync", True)):
        fail("sink_sync must remain false")
    if bool(cfg.get("sink_qos", True)):
        fail("sink_qos must remain false")
    if int(cfg.get("tiler_columns", 0)) * int(cfg.get("tiler_rows", 0)) < 6:
        fail("tiler grid does not fit all six cameras")

    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
    except Exception as exc:
        fail(f"GStreamer Python bindings unavailable: {type(exc).__name__}: {exc}")

    required = ("nvurisrcbin", "nvstreammux", "nvmultistreamtiler", "nveglglessink", "queue")
    missing = [name for name in required if Gst.ElementFactory.find(name) is None]
    if missing:
        fail("missing plugins: " + ", ".join(missing))

    print(
        "VISION_V3_CAMERA_PREFLIGHT=PASS "
        f"cameras=6 auth={auth_count}/6 "
        f"working={int(cfg.get('working_width', 1280))}x{int(cfg.get('working_height', 720))} "
        f"wall={int(cfg.get('wall_width', 1920))}x{int(cfg.get('wall_height', 720))} "
        + " ".join(feeds),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
