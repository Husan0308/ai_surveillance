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
    latency_ms: int | None = None
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
class DetectionConfig:
    enabled: bool
    model: str
    device: str
    width: int
    height: int
    batch_size: int
    target_fps_per_camera: float
    confidence: float
    iou: float
    max_detections: int
    half: bool
    overlay: bool
    overlay_max_age_ms: int


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool
    track_high_thresh: float
    track_low_thresh: float
    new_track_thresh: float
    track_buffer_seconds: float
    match_thresh: float
    fuse_score: bool


@dataclass(frozen=True)
class Settings:
    cameras: tuple[CameraConfig, ...]
    deepstream: DeepStreamConfig
    detection: DetectionConfig
    tracking: TrackingConfig


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


def _camera_uri(camera_id: str, row_value: str) -> str:
    env_name = f"{camera_id.replace('-', '_')}_URI"
    override = os.getenv(env_name)
    if override is not None and override.strip():
        return override.strip()
    return str(row_value or "").strip()


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

        uri = _camera_uri(camera_id, str(row.get("uri", "")))
        if not uri.startswith("rtsp://"):
            raise ValueError(f"{camera_id}: source must start with rtsp://")

        camera_latency = row.get("latency_ms")
        if camera_latency is not None:
            camera_latency = int(camera_latency)
            if camera_latency < 1:
                raise ValueError(f"{camera_id}: latency_ms must be >= 1")

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
                latency_ms=camera_latency,
                enabled=True,
            )
        )

    if not cameras:
        raise ValueError("At least one camera must be enabled")

    ds = raw.get("deepstream") or {}
    display = raw.get("display") or {}
    detection = raw.get("detection") or {}
    tracking = raw.get("tracking") or {}

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

    width = int(display.get("width", 960))
    height = int(display.get("height", 540))
    fps = int(display.get("fps", 20))
    quality = int(display.get("jpeg_quality", 88))
    if width <= 0 or height <= 0:
        raise ValueError("display width/height must be positive")
    if not 1 <= fps <= 60:
        raise ValueError("display.fps must be 1..60")
    if not 20 <= quality <= 95:
        raise ValueError("display.jpeg_quality must be 20..95")

    detect_width = int(detection.get("width", 512))
    detect_height = int(detection.get("height", 288))
    detect_batch = int(detection.get("batch_size", 2))
    detect_fps = float(detection.get("target_fps_per_camera", 4.0))
    detect_conf = float(detection.get("confidence", 0.10))
    detect_iou = float(detection.get("iou", 0.55))
    detect_max = int(detection.get("max_detections", 30))
    overlay_max_age_ms = int(detection.get("overlay_max_age_ms", 900))

    if detect_width < 320 or detect_height < 192:
        raise ValueError("detection width/height are too small")
    if not 1 <= detect_batch <= len(cameras):
        raise ValueError("detection.batch_size must be between 1 and enabled camera count")
    if not 0.1 <= detect_fps <= 30.0:
        raise ValueError("detection.target_fps_per_camera must be 0.1..30")
    if not 0.01 <= detect_conf <= 1.0:
        raise ValueError("detection.confidence must be 0.01..1.0")
    if not 0.01 <= detect_iou <= 1.0:
        raise ValueError("detection.iou must be 0.01..1.0")
    if not 1 <= detect_max <= 300:
        raise ValueError("detection.max_detections must be 1..300")
    if overlay_max_age_ms < 0:
        raise ValueError("detection.overlay_max_age_ms must be >= 0")

    track_high = float(tracking.get("track_high_thresh", 0.25))
    track_low = float(tracking.get("track_low_thresh", 0.10))
    new_track = float(tracking.get("new_track_thresh", 0.25))
    track_buffer_seconds = float(tracking.get("track_buffer_seconds", 2.5))
    match_thresh = float(tracking.get("match_thresh", 0.80))
    if not 0.0 <= track_low <= track_high <= 1.0:
        raise ValueError("tracking thresholds must satisfy 0 <= low <= high <= 1")
    if not 0.0 <= new_track <= 1.0:
        raise ValueError("tracking.new_track_thresh must be 0..1")
    if not 0.1 <= track_buffer_seconds <= 30.0:
        raise ValueError("tracking.track_buffer_seconds must be 0.1..30")
    if not 0.0 <= match_thresh <= 1.0:
        raise ValueError("tracking.match_thresh must be 0..1")

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
        detection=DetectionConfig(
            enabled=_as_bool(detection.get("enabled", True)),
            model=str(os.getenv("ML_DETECT_MODEL", detection.get("model", "yolo26m.pt"))).strip(),
            device=str(os.getenv("ML_DETECT_DEVICE", detection.get("device", f"cuda:{gpu_id}"))).strip(),
            width=detect_width,
            height=detect_height,
            batch_size=detect_batch,
            target_fps_per_camera=detect_fps,
            confidence=detect_conf,
            iou=detect_iou,
            max_detections=detect_max,
            half=_as_bool(detection.get("half", False)),
            overlay=_as_bool(detection.get("overlay", True)),
            overlay_max_age_ms=overlay_max_age_ms,
        ),
        tracking=TrackingConfig(
            enabled=_as_bool(tracking.get("enabled", True)),
            track_high_thresh=track_high,
            track_low_thresh=track_low,
            new_track_thresh=new_track,
            track_buffer_seconds=track_buffer_seconds,
            match_thresh=match_thresh,
            fuse_score=_as_bool(tracking.get("fuse_score", True)),
        ),
    )
