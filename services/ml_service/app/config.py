from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env", override=False)

MAX_CAMERAS = 16


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    uri: str
    username: str
    password: str
    name: str = ""
    room: str = ""
    enabled: bool = True


@dataclass(frozen=True)
class DeepStreamConfig:
    gpu_id: int
    cudadec_memtype: int
    rtsp_transport: str
    latency_ms: int
    drop_on_latency: bool
    decoder_extra_surfaces: int
    udp_buffer_size: int
    postdecode_queue_buffers: int
    capture_timeout_ms: int
    startup_grace_sec: float
    reconnect_delay_sec: float
    reconnect_delay_max_sec: float
    startup_stagger_sec: float
    display_width: int
    display_height: int
    display_fps: int
    jpeg_quality: int


@dataclass(frozen=True)
class Settings:
    cameras: tuple[CameraConfig, ...]
    deepstream: DeepStreamConfig


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _credential(
    camera_id: str,
    suffix: str,
    global_name: str,
    row_value: str = "",
) -> str:
    per_camera_name = f"{camera_id.replace('-', '_')}_RTSP_{suffix}"
    per_camera = os.getenv(per_camera_name)
    if per_camera is not None and per_camera != "":
        return per_camera
    global_value = os.getenv(global_name)
    if global_value is not None and global_value != "":
        return global_value
    return str(row_value or "")


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path or os.getenv("CAMERA_CONFIG", "config/cameras.yaml"))
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Camera config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    camera_rows = raw.get("cameras") or []
    if not camera_rows:
        raise ValueError("At least one camera must be configured")
    if len(camera_rows) > MAX_CAMERAS:
        raise ValueError(f"At most {MAX_CAMERAS} cameras are supported, got {len(camera_rows)}")

    cameras: list[CameraConfig] = []
    seen: set[str] = set()
    for row in camera_rows:
        if not _as_bool(row.get("enabled", True)):
            continue

        camera_id = str(row["id"]).strip()
        if not camera_id:
            raise ValueError("Camera id cannot be empty")
        if camera_id in seen:
            raise ValueError(f"Duplicate camera id: {camera_id}")
        seen.add(camera_id)

        # Codec is negotiated by DeepStream nvurisrcbin from the RTSP stream.
        # Keeping codec out of config avoids H264/H265 drift between NVR and app.
        uri = str(row.get("uri", "")).strip()
        if not uri.startswith("rtsp://"):
            raise ValueError(f"{camera_id}: source must start with rtsp://")

        cameras.append(
            CameraConfig(
                camera_id=camera_id,
                uri=uri,
                username=_credential(
                    camera_id,
                    "USERNAME",
                    "SURVEILLANCE_RTSP_USERNAME",
                    str(row.get("username", "")),
                ),
                password=_credential(
                    camera_id,
                    "PASSWORD",
                    "SURVEILLANCE_RTSP_PASSWORD",
                    str(row.get("password", "")),
                ),
                name=str(row.get("name", camera_id)).strip() or camera_id,
                room=str(row.get("room", "")).strip(),
                enabled=True,
            )
        )

    if not cameras:
        raise ValueError("At least one camera must be enabled")

    ds = raw.get("deepstream") or {}
    display = raw.get("display") or {}

    transport = str(ds.get("rtsp_transport", "auto")).strip().lower()
    if transport not in {"auto", "tcp"}:
        raise ValueError("deepstream.rtsp_transport must be auto or tcp for nvurisrcbin")

    gpu_id = int(ds.get("gpu_id", 0))
    cudadec_memtype = int(ds.get("cudadec_memtype", 0))
    latency_ms = int(ds.get("latency_ms", 150))
    decoder_extra_surfaces = int(ds.get("decoder_extra_surfaces", 4))
    udp_buffer_size = int(ds.get("udp_buffer_size", 1_048_576))
    postdecode_queue_buffers = int(ds.get("postdecode_queue_buffers", 1))
    capture_timeout_ms = int(ds.get("capture_timeout_ms", 1200))
    startup_grace_sec = float(ds.get("startup_grace_sec", 12.0))
    reconnect_delay_sec = float(ds.get("reconnect_delay_sec", 2.0))
    reconnect_delay_max_sec = float(ds.get("reconnect_delay_max_sec", 10.0))
    startup_stagger_sec = float(ds.get("startup_stagger_sec", 0.5))

    if gpu_id < 0:
        raise ValueError("deepstream.gpu_id must be >= 0")
    if cudadec_memtype not in {0, 1, 2}:
        raise ValueError("deepstream.cudadec_memtype must be 0, 1, or 2")
    if latency_ms < 1:
        raise ValueError("deepstream.latency_ms must be >= 1")
    if decoder_extra_surfaces < 1:
        raise ValueError("deepstream.decoder_extra_surfaces must be >= 1")
    if udp_buffer_size < 1:
        raise ValueError("deepstream.udp_buffer_size must be >= 1")
    if postdecode_queue_buffers < 1:
        raise ValueError("deepstream.postdecode_queue_buffers must be >= 1")
    if capture_timeout_ms < 100:
        raise ValueError("deepstream.capture_timeout_ms must be >= 100")
    if startup_grace_sec <= 0:
        raise ValueError("deepstream.startup_grace_sec must be > 0")
    if reconnect_delay_sec <= 0 or reconnect_delay_max_sec < reconnect_delay_sec:
        raise ValueError("deepstream reconnect delay range is invalid")
    if startup_stagger_sec < 0:
        raise ValueError("deepstream.startup_stagger_sec must be >= 0")

    width = int(display.get("width", 736))
    height = int(display.get("height", 416))
    fps = int(display.get("fps", 20))
    quality = int(display.get("jpeg_quality", 70))
    if width <= 0 or height <= 0:
        raise ValueError("display width/height must be positive")
    if not 1 <= fps <= 60:
        raise ValueError("display.fps must be 1..60")
    if not 20 <= quality <= 95:
        raise ValueError("display.jpeg_quality must be 20..95")

    return Settings(
        cameras=tuple(cameras),
        deepstream=DeepStreamConfig(
            gpu_id=gpu_id,
            cudadec_memtype=cudadec_memtype,
            rtsp_transport=transport,
            latency_ms=latency_ms,
            drop_on_latency=_as_bool(ds.get("drop_on_latency", True)),
            decoder_extra_surfaces=decoder_extra_surfaces,
            udp_buffer_size=udp_buffer_size,
            postdecode_queue_buffers=postdecode_queue_buffers,
            capture_timeout_ms=capture_timeout_ms,
            startup_grace_sec=startup_grace_sec,
            reconnect_delay_sec=reconnect_delay_sec,
            reconnect_delay_max_sec=reconnect_delay_max_sec,
            startup_stagger_sec=startup_stagger_sec,
            display_width=width,
            display_height=height,
            display_fps=fps,
            jpeg_quality=quality,
        ),
    )
