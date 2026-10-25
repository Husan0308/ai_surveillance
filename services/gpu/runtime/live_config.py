"""Build a private, six-source DeepStream 9.1 live-viewer config.

The returned application INI includes RTSP authentication. Keep it in memory
and write it only to a container tmpfs during an approved bounded launch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit
import configparser
import re

CAMERAS = ("CAM-01", "CAM-02", "CAM-03", "CAM-04", "CAM-05", "CAM-06")
CHANNELS = (101, 201, 301, 601, 501, 401)
TRACKER_CFG = (
    "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/"
    "config_tracker_NvDCF_perf.yml"
)
TRACKER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
REQUIRED_MODELS = ("model-engine-file", "onnx-file", "labelfile-path")


@dataclass(frozen=True)
class Camera:
    camera_id: str
    uri: str
    username: str
    password: str


def camera_uri(camera: Camera, expected_channel: int) -> tuple[str, str, int]:
    """Return private authenticated URI, NVR host, and port; never log URI."""
    if not camera.username or not camera.password:
        raise ValueError(f"{camera.camera_id}: missing authorized credentials")
    original = urlsplit(camera.uri)
    if original.scheme.lower() != "rtsp" or not original.hostname:
        raise ValueError(f"{camera.camera_id}: expected RTSP source")
    if original.username is not None or original.password is not None:
        raise ValueError(f"{camera.camera_id}: credentials must be supplied separately")
    if any(c in camera.uri for c in ("\n", "\r", "\x00", " ", "\t")):
        raise ValueError(f"{camera.camera_id}: malformed URI")
    try:
        port = original.port or 554
    except ValueError as exc:
        raise ValueError(f"{camera.camera_id}: invalid NVR port") from exc
    route = re.fullmatch(r"/Streaming/Channels/([0-9]+)", original.path)
    if not route or int(route.group(1)) != expected_channel:
        raise ValueError(f"{camera.camera_id}: unexpected NVR channel")
    if original.fragment:
        raise ValueError(f"{camera.camera_id}: URI fragment not allowed")
    auth = quote(camera.username, safe="") + ":" + quote(camera.password, safe="")
    hostpart = original.netloc
    uri = urlunsplit((original.scheme, auth + "@" + hostpart,
                       original.path, original.query, ""))
    return uri, original.hostname, port


def validate_infer_config(content: str) -> dict[str, str]:
    """Require the already-validated PeopleNet FP16 batch-6 config.

    The selected engine and model files must be bind-mounted below /models.
    Do not silently substitute the older batch-2 diagnostic configuration.
    """
    cfg = configparser.ConfigParser(interpolation=None, strict=True)
    cfg.read_string(content)
    if "property" not in cfg:
        raise ValueError("missing nvinfer [property] section")
    p = cfg["property"]
    if p.get("batch-size") != "6":
        raise ValueError("inference config must use batch-size=6")
    if p.get("network-mode") != "2" or p.get("network-type") != "0":
        raise ValueError("expected FP16 PeopleNet detector")
    if p.get("num-detected-classes") != "3":
        raise ValueError("expected PeopleNet Person/Bag/Face classes")
    try:
        excluded = {int(x.strip()) for x in p["filter-out-class-ids"].split(";")}
    except (KeyError, ValueError) as exc:
        raise ValueError("Person-only class filter is required") from exc
    if excluded != {1, 2}:
        raise ValueError("only Person class 0 may be published")
    paths: dict[str, str] = {}
    for key in REQUIRED_MODELS:
        value = p.get(key, "")
        if not value.startswith("/models/") or ".." in Path(value).parts:
            raise ValueError(f"{key}: expected a file inside /models")
        paths[key] = value
    return paths


def validate_model_files(config_text: str, models_root: Path) -> dict[str, Path]:
    paths = validate_infer_config(config_text)
    result: dict[str, Path] = {}
    base = models_root.resolve(strict=True)
    for key, path in paths.items():
        rel = Path(path).relative_to("/models")
        candidate = (base / rel).resolve(strict=True)
        if not candidate.is_relative_to(base) or not candidate.is_file():
            raise ValueError(f"{key}: missing or unsafe local model file")
        result[key] = candidate
    return result


def build_app_config(cameras: list[Camera]) -> tuple[str, dict[str, object]]:
    if len(cameras) != 6 or tuple(c.camera_id for c in cameras) != CAMERAS:
        raise ValueError("six sources must map to CAM-01..CAM-06 in order")
    uris = []
    endpoints = set()
    for cam, channel in zip(cameras, CHANNELS):
        uri, host, port = camera_uri(cam, channel)
        uris.append(uri)
        endpoints.add((host, port))
    if len(endpoints) != 1:
        raise ValueError("expected one explicitly reviewed NVR endpoint")

    lines = [
        "[application]", "enable-perf-measurement=1",
        "perf-measurement-interval-sec=5", "",
        "[tiled-display]", "enable=1", "rows=2", "columns=3",
        "width=1920", "height=1080", "gpu-id=0", "",
    ]
    for index, uri in enumerate(uris):
        lines.extend([
            f"[source{index}]", "enable=1", "type=4",
            f"uri={uri}", "gpu-id=0", "select-rtp-protocol=4",
            "rtsp-reconnect-interval-sec=10", "rtsp-reconnect-attempts=3",
            "num-extra-surfaces=8", "",
        ])
    lines.extend([
        "[streammux]", "gpu-id=0", "batch-size=6", "live-source=1",
        "batched-push-timeout=50000", "width=1920", "height=1080",
        "enable-padding=0", "",
        "[primary-gie]", "enable=1", "gpu-id=0",
        "config-file=/run/live/infer.txt", "",
        "[tracker]", "enable=1", "gpu-id=0",
        "tracker-width=960", "tracker-height=544",
        f"ll-lib-file={TRACKER_LIB}", f"ll-config-file={TRACKER_CFG}", "",
        "[osd]", "enable=1", "gpu-id=0", "border-width=2",
        "text-size=12", "display-text=1", "",
        "[sink0]", "enable=1", "type=2", "sync=0", "qos=0",
        "gpu-id=0", "",
        "[tests]", "file-loop=0", "",
    ])
    # Do not put URIs or credentials in the public manifest.
    manifest = {
        "cameras": list(CAMERAS),
        "channels": list(CHANNELS),
        "source_indices": list(range(6)),
        "nvr_host": next(iter(endpoints))[0],
        "nvr_port": next(iter(endpoints))[1],
        "layout": {"rows": 2, "columns": 3, "width": 1920, "height": 1080},
        "model": "PeopleNet 2.6.3 FP16 / Person-only / batch 6",
        "tracker": "NVIDIA stock NvDCF",
        "network": "requires independently verified six-camera egress policy",
    }
    return "\n".join(lines), manifest
