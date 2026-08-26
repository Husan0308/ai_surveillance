from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env", override=False)


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    uri: str
    username: str
    password: str


@dataclass(frozen=True)
class CameraServiceSettings:
    cameras: tuple[CameraConfig, ...]
    gpu_id: int
    rtsp_transport: str
    latency_ms: int
    extra_surfaces: int
    source_fps: int
    display_width: int
    display_height: int
    wall_width: int
    wall_height: int
    startup_stagger_sec: float


def _credential(camera_id: str, suffix: str, global_name: str, row_value: str = "") -> str:
    per_camera = os.getenv(f"{camera_id.replace('-', '_')}_RTSP_{suffix}")
    if per_camera:
        return per_camera
    global_value = os.getenv(global_name)
    if global_value:
        return global_value
    return str(row_value or "")


def load_settings(path: str | Path | None = None) -> CameraServiceSettings:
    config_path = Path(path or os.getenv("CAMERA_CONFIG", "config/cameras.yaml"))
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    cameras: list[CameraConfig] = []
    seen: set[str] = set()
    for row in raw.get("cameras") or []:
        if not bool(row.get("enabled", True)):
            continue
        cid = str(row["id"]).strip()
        uri = str(row.get("uri", "")).strip()
        if not cid or cid in seen:
            raise ValueError(f"invalid/duplicate camera id: {cid!r}")
        if not uri.startswith("rtsp://"):
            raise ValueError(f"{cid}: source must be rtsp://")
        seen.add(cid)
        cameras.append(
            CameraConfig(
                camera_id=cid,
                uri=uri,
                username=_credential(cid, "USERNAME", "SURVEILLANCE_RTSP_USERNAME", row.get("username", "")),
                password=_credential(cid, "PASSWORD", "SURVEILLANCE_RTSP_PASSWORD", row.get("password", "")),
            )
        )
    if not cameras:
        raise ValueError("camera_service needs at least one enabled camera")

    deepstream = raw.get("deepstream") or {}
    return CameraServiceSettings(
        cameras=tuple(cameras),
        gpu_id=int(os.getenv("CAMERA_SERVICE_GPU_ID", deepstream.get("gpu_id", 0))),
        rtsp_transport=os.getenv("CAMERA_SERVICE_RTSP_TRANSPORT", "tcp").strip().lower(),
        latency_ms=max(40, int(os.getenv("CAMERA_SERVICE_RTSP_LATENCY_MS", "80"))),
        extra_surfaces=max(2, min(16, int(os.getenv("CAMERA_SERVICE_EXTRA_SURFACES", "8")))),
        source_fps=max(1, int(os.getenv("CAMERA_SERVICE_SOURCE_FPS", "20"))),
        display_width=max(640, int(os.getenv("CAMERA_SERVICE_DISPLAY_WIDTH", "1280"))),
        display_height=max(360, int(os.getenv("CAMERA_SERVICE_DISPLAY_HEIGHT", "720"))),
        wall_width=max(960, int(os.getenv("CAMERA_SERVICE_WALL_WIDTH", "1920"))),
        wall_height=max(360, int(os.getenv("CAMERA_SERVICE_WALL_HEIGHT", "720"))),
        startup_stagger_sec=max(0.1, float(os.getenv("CAMERA_SERVICE_STARTUP_STAGGER_SEC", "0.50"))),
    )
