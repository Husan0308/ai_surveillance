from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import yaml


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    uri: str


@dataclass(frozen=True)
class FrameTransportConfig:
    enabled: bool
    directory: str
    width: int
    height: int
    max_fps: int


@dataclass(frozen=True)
class DeepStreamConfig:
    gpu_id: int
    latency_ms: int
    drop_on_latency: bool
    reconnect_interval_sec: int
    reconnect_attempts: int
    decoder_extra_surfaces: int
    cudadec_memtype: int
    udp_buffer_size: int
    rtp_protocol: int
    batch_size: int
    mux_width: int
    mux_height: int
    live_source: bool
    batched_push_timeout_us: int
    sync_inputs: bool
    display_enabled: bool
    display_rows: int
    display_columns: int
    display_width: int
    display_height: int
    frame_transport: FrameTransportConfig


@dataclass(frozen=True)
class Settings:
    cameras: tuple[CameraConfig, ...]
    deepstream: DeepStreamConfig


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _inject_rtsp_auth(uri: str, row: dict[str, Any]) -> str:
    """Optionally inject per-camera RTSP credentials from environment.

    Authentication is opt-in: unless username_env is explicitly configured for a
    camera, the URI is left exactly as written in cameras.yaml. This avoids
    silently changing camera URLs that already work without URI credentials.
    """
    username_env = str(row.get("username_env", "")).strip()
    if not username_env:
        return uri

    parts = urlsplit(uri)
    if parts.scheme.lower() != "rtsp" or "@" in parts.netloc:
        return uri

    password_env = str(row.get("password_env", "")).strip()
    username = os.getenv(username_env, "").strip()
    password = os.getenv(password_env, "") if password_env else ""
    if not username:
        return uri

    userinfo = quote(username, safe="")
    if password:
        userinfo += ":" + quote(password, safe="")
    return urlunsplit(
        (
            parts.scheme,
            f"{userinfo}@{parts.netloc}",
            parts.path,
            parts.query,
            parts.fragment,
        )
    )


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path or os.getenv("CAMERA_CONFIG", "config/cameras.yaml"))
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Camera config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    camera_rows = raw.get("cameras") or []
    if len(camera_rows) != 6:
        raise ValueError(f"Expected exactly 6 cameras, got {len(camera_rows)}")

    cameras: list[CameraConfig] = []
    seen_ids: set[str] = set()
    for row in camera_rows:
        camera_id = str(row["id"]).strip()
        if camera_id in seen_ids:
            raise ValueError(f"Duplicate camera id: {camera_id}")
        seen_ids.add(camera_id)

        env_uri = str(row.get("env_uri", "")).strip()
        uri = (
            os.getenv(env_uri, str(row["uri"]).strip())
            if env_uri
            else str(row["uri"]).strip()
        )
        if not uri.startswith("rtsp://"):
            raise ValueError(f"{camera_id}: only rtsp:// sources are allowed")
        uri = _inject_rtsp_auth(uri, row)
        cameras.append(CameraConfig(camera_id=camera_id, uri=uri))

    ds = raw.get("deepstream") or {}
    mux = ds.get("streammux") or {}
    display = ds.get("display") or {}
    frame_transport = ds.get("frame_transport") or {}

    batch_size = int(mux.get("batch_size", len(cameras)))
    if batch_size != len(cameras):
        raise ValueError(
            f"streammux.batch_size must equal camera count ({len(cameras)}), got {batch_size}"
        )

    frame_width = int(frame_transport.get("width", 640))
    frame_height = int(frame_transport.get("height", 360))
    frame_max_fps = int(frame_transport.get("max_fps", 15))
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_transport width/height must be positive")
    if not 1 <= frame_max_fps <= 60:
        raise ValueError("frame_transport.max_fps must be between 1 and 60")

    return Settings(
        cameras=tuple(cameras),
        deepstream=DeepStreamConfig(
            gpu_id=int(ds.get("gpu_id", 0)),
            latency_ms=int(ds.get("latency_ms", 150)),
            drop_on_latency=_as_bool(ds.get("drop_on_latency", True)),
            reconnect_interval_sec=int(ds.get("reconnect_interval_sec", 10)),
            reconnect_attempts=int(ds.get("reconnect_attempts", -1)),
            decoder_extra_surfaces=int(ds.get("decoder_extra_surfaces", 4)),
            cudadec_memtype=int(ds.get("cudadec_memtype", 0)),
            udp_buffer_size=int(ds.get("udp_buffer_size", 1_048_576)),
            rtp_protocol=int(ds.get("rtp_protocol", 4)),
            batch_size=batch_size,
            mux_width=int(mux.get("width", 1280)),
            mux_height=int(mux.get("height", 720)),
            live_source=_as_bool(mux.get("live_source", True)),
            batched_push_timeout_us=int(mux.get("batched_push_timeout_us", 50_000)),
            sync_inputs=_as_bool(mux.get("sync_inputs", False)),
            display_enabled=_as_bool(display.get("enabled", False)),
            display_rows=int(display.get("rows", 2)),
            display_columns=int(display.get("columns", 3)),
            display_width=int(display.get("width", 1920)),
            display_height=int(display.get("height", 1080)),
            frame_transport=FrameTransportConfig(
                enabled=_as_bool(frame_transport.get("enabled", True)),
                directory=os.getenv(
                    "FRAME_BUS_DIR",
                    str(frame_transport.get("directory", "/dev/shm/ai_surveillance")),
                ),
                width=frame_width,
                height=frame_height,
                max_fps=frame_max_fps,
            ),
        ),
    )
